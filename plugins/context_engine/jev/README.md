# Jev context shadow mode

This context engine runs a bounded TypeSafe Jev evaluation over selected older
tool results in long sessions. It is observation-only: Jev's decisions are
reported as shadow metrics and never remove, rewrite, or reorder messages sent
to the configured Hermes model. Hermes' normal `ContextCompressor` remains
active with the host's configured compression policy.

## Enable

Store the TypeSafe credential in Hermes' secret environment (`~/.hermes/.env`):

```dotenv
TYPESAFE_API_KEY=...
```

Then select the engine in `~/.hermes/config.yaml`:

```yaml
context:
  engine: jev
```

Without the key, the plugin loads without network access and reports itself as
unavailable. It does not expose a Jev model tool or a direct CLI entry point.

## Data and behavior boundary

When the old tool-output threshold is met, the plugin sends the latest user
task and up to eight older tool-result excerpts (at most 1,500 characters each)
to `https://api.typesafe.ai/v1/systemone`. Sensitive text is redacted before
egress, values in sensitive credential fields are replaced, request and response sizes are
bounded, redirects are refused, and the call has a two-second timeout. No
transcript text or Jev response body is written to the plugin's status metrics.

The engine keeps candidate indices, typed decisions, and token usage
internally for local diagnostics, but the gateway status boundary exposes
only the following content-free fields: `mode`, `attempted`, `ok`, `model`,
`latency_ms`, `candidates`, `would_drop_count`, and `would_reclaim_chars`.
Prompt text, tool-result text, credentials, raw provider errors, and the
internal decision/usage payload never appear in `/status`. Every call returns
`None` from `select_context()`, so Hermes keeps using the original request.
Enabling active pruning requires a separate reviewed change and evaluation; it
is not part of this plugin's shadow mode.
