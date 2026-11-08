# `zoo_build`

Generate the network zoo.

**Who can call it:** human only (UI / HTTP API / CLI).

## Inputs

| Name | Type | Default |
|---|---|---|
| `scale` | `str` | `"full"` |
| `fresh` | `bool` | `false` |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

```bash
wsp zoo build --scale full   # human-only; via HTTP: POST /api/tools/zoo_build is refused
```
