# Configuration

Defaults live in `wsp_toolkit/default_config.toml` (every key is commented there). Override them with your own
TOML file via `WSP_CONFIG=/path/to/config.toml` or `--config`. Keys you set are merged over the defaults.
`WSP_DATA_DIR` overrides `paths.data_dir`.

Key settings:

| Key | Default | Meaning |
|---|---|---|
| `paths.data_dir` | `~/.wsp_toolkit` | all state: `wsp.sqlite`, `bases/`, `patches/`, `exports/`, `fixtures/`, `zoo/`, `features/` |
| `limits.max_response_bytes` | 16000 | hard cap on any tool response (bigger → handle + `result_page`) |
| `limits.max_slice_elements` | 512 | raw values per `tensor_get` / `set_slice` |
| `limits.download_confirm_gb` | 1.0 | downloads above this need `confirm_download=true` |
| `session.access_level` | `weights+activations+outputs` | **ceiling** for agent sessions |
| `session.forward_pass_budget` | 50 | **ceiling** budget per agent session |
| `session.inputs_per_pass` | 1 | inputs covered by one budgeted pass (paper: 512) |
| `session.heldout_rate_limit_per_hour` | 5 | held-out evaluations per session per hour |
| `policy.max_param_change_pct` | 5.0 | max % of parameters one patch may change |
| `policy.allowed_primitives` | 5 primitives | which edit ops agents may use |
| `policy.required_suites` | `["regression"]` | suites needed before a patch is `tested` |
| `policy.auto_regression_on_apply` | true | run required suites automatically on `edit_apply` |
| `policy.sensitive_direction_cosine` | 0.3 | alignment threshold for sensitive-direction flags |
| `policy.forbidden_models` | `[]` | sources / hashes that may never be opened (e.g. the agent's own weights) |
| `eval.isolation` | `thread` | `thread` (per-thread network guard) or `subprocess` |
| `ui.reviewer_token` | `""` | if set, approvals need this token |

Environment: `WSP_HF_DEVICE` (`cpu`/`cuda`/`auto`) and `WSP_HF_DTYPE` (`float32`/`bfloat16`) for HF models.

Example experiment config reproducing the paper's 50-pass Qwen condition:

```toml
[session]
access_level = "weights+activations+outputs"
forward_pass_budget = 50
inputs_per_pass = 512
```
