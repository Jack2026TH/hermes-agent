"""MEDIA tags pointing into a remote terminal sandbox are fetched for delivery (#466).

The host cannot see files an agent writes inside an ssh/modal/daytona sandbox, so
``filter_media_delivery_paths`` used to drop them silently. With a remote backend active the path is
pulled through the environment's ``fetch_file`` into the document cache — after the SAME denylist
the host applies, so a sandbox path (or a symlink) at ``~/.ssh/...`` never crosses.
"""

from pathlib import Path

import pytest

import gateway.media_fetch as media_fetch
from gateway.platforms.base import BasePlatformAdapter
from tools.environments.base import BaseEnvironment, FileFetchError


class _FakeRemoteEnv:
    """Stands in for a live SSHEnvironment: a tiny remote filesystem + symlink table."""

    _remote_home = "/home/agent"

    def __init__(self):
        self.files = {"/home/agent/out/report.txt": b"hello from sandbox",
                      "/home/agent/.ssh/id_rsa": b"SECRET"}
        self.links = {"/home/agent/out/innocent.txt": "/home/agent/.ssh/id_rsa"}
        self.fetched: list = []

    def fetch_realpath(self, remote_path):
        return self.links.get(remote_path, remote_path)

    def fetch_file(self, remote_path, local_dest, *, max_bytes):
        self.fetched.append(remote_path)
        if remote_path not in self.files:
            raise FileFetchError("missing")
        Path(local_dest).write_bytes(self.files[remote_path])


@pytest.fixture
def remote_env(monkeypatch, tmp_path):
    env = _FakeRemoteEnv()
    monkeypatch.setattr(media_fetch, "_active_remote_env", lambda: env)
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "cache" / "documents")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path / "cache" / "documents",))
    return env


def test_sandbox_artifact_is_fetched_but_credentials_and_symlinks_to_them_are_not(remote_env):
    media = [("/home/agent/out/report.txt", False), ("/home/agent/out/innocent.txt", False),
             ("/home/agent/.ssh/id_rsa", False), ("~/out/report.txt", False)]
    delivered = BasePlatformAdapter.filter_media_delivery_paths(media)

    assert [Path(p).read_bytes() for p, _ in delivered] == [b"hello from sandbox", b"hello from sandbox"]
    assert all(Path(p).name.endswith("_report.txt") for p, _ in delivered)
    # The credential file and the symlink that resolves to it were refused BEFORE any bytes moved.
    assert remote_env.fetched == ["/home/agent/out/report.txt", "/home/agent/out/report.txt"]


def test_local_backend_and_strict_mode_do_not_fetch(monkeypatch, tmp_path, remote_env):
    """Strict mode keeps its recency gate: a fetched copy would land in an allowlisted root and skip it."""
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    assert BasePlatformAdapter.filter_media_delivery_paths([("/home/agent/out/report.txt", False)]) == []
    monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT")
    monkeypatch.setattr(media_fetch, "_active_remote_env", lambda: None)
    assert BasePlatformAdapter.filter_media_delivery_paths([(str(tmp_path / "nope.txt"), False)]) == []
    assert remote_env.fetched == []


class _ScriptedEnv(BaseEnvironment):
    """Runs the fetch command through a real shell so the transport (marker fencing, in-sandbox
    size bound, base64 round-trip) is exercised end to end."""

    def __init__(self):
        pass

    def cleanup(self):
        pass

    def execute(self, command, cwd="", **kwargs):
        import subprocess
        proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
        return {"output": "echo login-noise\n" + proc.stdout + proc.stderr, "returncode": proc.returncode}

    def _run_bash(self, command, *, login=False, timeout=120, stdin_data=None):
        import subprocess
        assert not login
        return subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)


@pytest.mark.parametrize("poison", ["function", "alias", "path"])
def test_single_link_fetch_ignores_agent_shell_snapshot(tmp_path, poison):
    """The real execute wrapper must not let session shell state forge stat nlink."""
    import os
    import shlex

    secret = tmp_path / "synthetic-secret"
    secret.write_bytes(b"synthetic credential fixture")
    alias = tmp_path / "artifact.png"
    os.link(secret, alias)
    dest = tmp_path / "copy.png"

    class SnapshotShell(_ScriptedEnv):
        def __init__(self):
            BaseEnvironment.__init__(self, str(tmp_path), 5)
            self._snapshot_path = str(tmp_path / "snapshot.sh")
            self._cwd_file = str(tmp_path / "cwd.txt")
            self._snapshot_ready = True

        execute = BaseEnvironment.execute

    env = SnapshotShell()
    if poison == "function":
        snapshot = "stat() { echo 1; }\n"
    elif poison == "alias":
        snapshot = "shopt -s expand_aliases\nalias stat='echo 1 #'\n"
    else:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake = bin_dir / "stat"
        fake.write_text("#!/bin/sh\necho 1\n", encoding="utf-8")
        fake.chmod(0o755)
        snapshot = "export PATH=" + shlex.quote(str(bin_dir)) + ":$PATH\n"
    Path(env._snapshot_path).write_text(snapshot, encoding="utf-8")
    with pytest.raises(FileFetchError):
        env.fetch_file(str(alias), dest, max_bytes=100, require_single_link=True)
    assert not dest.exists(), "Hard-linked bytes must not reach the delivery cache"
    ordinary = tmp_path / "ordinary.png"
    ordinary.write_bytes(b"ordinary artifact")
    env.fetch_file(str(ordinary), dest, max_bytes=100, require_single_link=True)
    assert dest.read_bytes() == ordinary.read_bytes()


def test_fetch_file_round_trips_bytes_and_enforces_the_in_sandbox_size_cap(tmp_path):
    src = tmp_path / "artifact.bin"
    src.write_bytes(bytes(range(256)) * 40)
    dest = tmp_path / "copy.bin"
    env = _ScriptedEnv()

    env.fetch_file(str(src), dest, max_bytes=len(src.read_bytes()))
    assert dest.read_bytes() == src.read_bytes()

    with pytest.raises(FileFetchError, match="exceeds"):
        env.fetch_file(str(src), dest, max_bytes=100)
    with pytest.raises(FileFetchError, match="could not read"):
        env.fetch_file(str(tmp_path), dest, max_bytes=100)  # a directory is not a regular file


@pytest.mark.parametrize("prefix", ["/workspace", "/output", "///workspace"])
def test_docker_fallback_never_copies_hardlinked_credentials(tmp_path, monkeypatch, prefix):
    """PR #11: exercise MEDIA filtering through the real shell transfer, not just local validation."""
    import os

    secret = tmp_path / "home" / ".hermes" / "auth.json"
    secret.parent.mkdir(parents=True)
    secret.write_bytes(b"synthetic credential fixture")
    alias = tmp_path / "remote" / "hard.png"
    alias.parent.mkdir()
    os.link(secret, alias)
    ordinary = alias.with_name("ordinary.png")
    ordinary.write_bytes(b"ordinary remote artifact")

    class DockerShell(_ScriptedEnv):
        _remote_home = None

        def fetch_realpath(self, remote_path):
            return str(alias.parent / Path(remote_path).name)

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)
    monkeypatch.setattr(media_fetch, "_active_remote_env", lambda: DockerShell())
    cache = tmp_path / "cache" / "documents"
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", cache)
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (cache,))

    media, _ = BasePlatformAdapter.extract_media(f"MEDIA:{prefix}/hard.png")
    assert BasePlatformAdapter.filter_media_delivery_paths(media) == []
    assert not list(cache.glob("*")), "Credential bytes must never reach the host cache"
    # Reject ambiguous slash spelling even for an ordinary file; preserve normal Docker fetches.
    if prefix.startswith("//"):
        assert media_fetch.fetch_remote_media(prefix + "/ordinary.png") is None
    delivered = BasePlatformAdapter.filter_media_delivery_paths([("/workspace/ordinary.png", False)])
    assert len(delivered) == 1
    assert Path(delivered[0][0]).read_bytes() == ordinary.read_bytes()
