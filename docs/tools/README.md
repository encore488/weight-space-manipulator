# Tool reference

Every tool returns JSON capped at `limits.max_response_bytes` (default 16000 bytes); bigger results come back as `{truncated, handle, pages, preview}` — fetch the rest with [`result_page`](result_page.md). Errors are `{error, code, hint?}` (over MCP they set `isError`).

## M1 inspect & probe

| Tool | Purpose | Who |
|---|---|---|
| [`workspace_open`](workspace_open.md) | Open a model (fixture:<name>, zoo:<id>, hf:<repo>, or a local path) into a new workspace with a working copy, and start a budgeted session on it. | agents (MCP) and humans |
| [`session_start`](session_start.md) | Start a new budgeted session on an existing workspace. | agents (MCP) and humans |
| [`workspace_info`](workspace_info.md) | Describe the session's model, working-copy state (applied patches, hash) and budget. | agents (MCP) and humans |
| [`budget_status`](budget_status.md) | Remaining forward-pass budget and held-out evaluation allowance for the session. | agents (MCP) and humans |
| [`tensor_list`](tensor_list.md) | List tensors with shapes, dtypes, layer grouping and head layout. Paged; regex filter. | agents (MCP) and humans |
| [`tensor_get`](tensor_get.md) | Raw values of a small tensor slice (numpy-style 'a:b, c:d'; optional head). Size-capped. | agents (MCP) and humans |
| [`tensor_stats`](tensor_stats.md) | Summary statistics of a tensor or slice: norms, distribution, sparsity, row/col/head norms. | agents (MCP) and humans |
| [`tensor_svd`](tensor_svd.md) | Top-k singular values/vectors of a 2-D tensor, slice or head. Each singular vector is stored as a reusable direction id. | agents (MCP) and humans |
| [`tensor_compare`](tensor_compare.md) | Compare two tensors (same or another session's workspace, e.g. base vs patched). | agents (MCP) and humans |
| [`direction_readout`](direction_readout.md) | Project a direction onto the vocabulary: which tokens it promotes / suppresses. | agents (MCP) and humans |
| [`direction_info`](direction_info.md) | Resolve a direction spec: dimension, norm-1 vector top entries, and a stored id. | agents (MCP) and humans |
| [`direction_cosine`](direction_cosine.md) | Cosine similarity between two directions of the same dimension. | agents (MCP) and humans |
| [`run_forward`](run_forward.md) | Run the model on inputs; returns top-k next-token predictions (LMs) or outputs. Needs 'outputs' access; costs 1 pass per input. | agents (MCP) and humans |
| [`generate`](generate.md) | Generate a continuation for each input (LMs). Needs 'outputs' access; 1 pass per input. | agents (MCP) and humans |
| [`probe_activations`](probe_activations.md) | Capture activations at hook points for inputs; returns summaries (norms, top units). Needs 'activations' access; 1 pass per input. | agents (MCP) and humans |
| [`direction_from_activations`](direction_from_activations.md) | Difference of mean activations (positive minus negative inputs) at a hook, stored as a direction. Needs 'activations' access; 1 pass per input. | agents (MCP) and humans |
| [`probe_attention`](probe_attention.md) | Attention-pattern summaries per head: entropy, attention by relative offset, top keys per query. Needs 'activations' access; 1 pass per input. | agents (MCP) and humans |
| [`result_page`](result_page.md) | Fetch page N of a large result that was returned as a handle. | agents (MCP) and humans |
| [`session_log`](session_log.md) | The session's event log (what has been done, in order). | agents (MCP) and humans |

## M2 edits & patches

| Tool | Purpose | Who |
|---|---|---|
| [`edit_preview`](edit_preview.md) | Show what a list of edit operations would change (tensors, #params, % of model, policy and sensitive-component flags) without applying. | agents (MCP) and humans |
| [`edit_apply`](edit_apply.md) | Apply edit operations to the working copy as a new draft patch (with rationale). Regression suites run automatically. | agents (MCP) and humans |
| [`patch_list`](patch_list.md) | List patches (optionally for this session's workspace or by status). | agents (MCP) and humans |
| [`patch_show`](patch_show.md) | Full manifest of a patch: operations, per-tensor hashes, params changed, rationale, linked evaluations, status, approvals. | agents (MCP) and humans |
| [`patch_diff`](patch_diff.md) | Compare two patches: tensors touched by each, overlapping elements, delta cosine. | agents (MCP) and humans |
| [`patch_revert`](patch_revert.md) | Revert an applied patch. mode='cascade' (default) also reverts patches above it; 'rebase' re-derives the patches above it. | agents (MCP) and humans |
| [`patch_reapply`](patch_reapply.md) | Re-apply an existing patch (whose parent is the current top of the stack). | agents (MCP) and humans |
| [`patch_export`](patch_export.md) | Export the patched model (base + patch chain) to safetensors. Blocked unless every patch in the chain is approved by a human. | agents (MCP) and humans |

## M3 harness

| Tool | Purpose | Who |
|---|---|---|
| [`suite_list`](suite_list.md) | List test suites for this model: practice (examples visible), held-out and regression (names and sizes only). | agents (MCP) and humans |
| [`suite_examples`](suite_examples.md) | Page through the examples of a PRACTICE suite (held-out / regression examples are never returned). | agents (MCP) and humans |
| [`suite_run_practice`](suite_run_practice.md) | Run a practice suite on the current working copy. Costs 1 pass per example (x samples). Per-example results need 'outputs' access. | agents (MCP) and humans |
| [`evaluate_heldout`](evaluate_heldout.md) | Aggregate score on a held-out suite for the current working copy. Never returns examples; rate-limited; logged. | agents (MCP) and humans |
| [`run_baseline`](run_baseline.md) | Run a non-edit baseline: the BASE model on a practice or held-out suite with different generation settings (e.g. temperature, repetition_penalty). Aggregate only for held-out. | agents (MCP) and humans |
| [`compare_runs`](compare_runs.md) | Per-metric comparison of two evaluation runs (with settings-mismatch warnings). | agents (MCP) and humans |
| [`patch_test`](patch_test.md) | Run the required suites (regression etc.) on parent vs patched; marks the patch 'tested'. | agents (MCP) and humans |

## M4 safety

| Tool | Purpose | Who |
|---|---|---|
| [`sensitive_list`](sensitive_list.md) | List registered sensitive components for this session's model. | agents (MCP) and humans |
| [`patch_approve`](patch_approve.md) | Approve a tested patch (human only). | human only (UI / HTTP API / CLI) |
| [`patch_reject`](patch_reject.md) | Reject a patch (human only). | human only (UI / HTTP API / CLI) |
| [`sensitive_register`](sensitive_register.md) | Mark a tensor, direction or feature as safety-relevant (human only). | human only (UI / HTTP API / CLI) |

## M5 zoo

| Tool | Purpose | Who |
|---|---|---|
| [`zoo_list_tasks`](zoo_list_tasks.md) | List zoo networks and their task types (identify / localize / repair). | agents (MCP) and humans |
| [`zoo_get_task`](zoo_get_task.md) | Task card for a zoo network: instructions, answer format, candidate labels, and the source string to open it with workspace_open. | agents (MCP) and humans |
| [`zoo_submit`](zoo_submit.md) | Submit an answer for a zoo task. Public split returns the score; hidden split is scored server-side and only aggregated. | agents (MCP) and humans |
| [`zoo_scores`](zoo_scores.md) | Aggregate zoo scores per submitter, split, family and task type. | agents (MCP) and humans |
| [`zoo_build`](zoo_build.md) | Generate the network zoo. | human only (UI / HTTP API / CLI) |

## M6 features

| Tool | Purpose | Who |
|---|---|---|
| [`feature_search`](feature_search.md) | Search the feature index by meaning (label text). Returns fidelity and the dictionaries' unexplained fraction. | agents (MCP) and humans |
| [`features_at_location`](features_at_location.md) | Which features do these weights write or read? (tensor + optional slice) | agents (MCP) and humans |
| [`neuron_features`](neuron_features.md) | Features linked to one unit of a hooked space, with decoder and encoder weights (both directions of the neuron<->feature overlap). | agents (MCP) and humans |
| [`features_active_on`](features_active_on.md) | Features active on given inputs, per position (needs 'activations'; 1 pass per input). Reports the unexplained fraction on these inputs. | agents (MCP) and humans |
| [`feature_get`](feature_get.md) | Full feature record: label, fidelity, top examples, vocabulary readout, neurons, zoom links, up/downstream features. | agents (MCP) and humans |
| [`feature_neighbors`](feature_neighbors.md) | Zoom parent/children, cross-layer links and similar features. | agents (MCP) and humans |
| [`feature_dictionaries`](feature_dictionaries.md) | List feature dictionaries (hook, size, unexplained fraction). | agents (MCP) and humans |

## M7 audit

| Tool | Purpose | Who |
|---|---|---|
| [`diff_models`](diff_models.md) | Weight diff between two sessions' models: per-tensor change size, low-rank structure, most-changed tensors and heads. Free (no forward passes). | agents (MCP) and humans |
| [`diff_report`](diff_report.md) | Full fine-tune audit (weights + features + behaviour), stored and viewable in the UI. Needs 'outputs'; costs 2 passes per probe. | agents (MCP) and humans |
| [`audit_get`](audit_get.md) | Markdown text of a stored audit report. | agents (MCP) and humans |

## Challenges

| Tool | Purpose | Who |
|---|---|---|
| [`challenge_list`](challenge_list.md) | List the WSP challenges (short, auto-scored tasks for comparing agents/models). | agents (MCP) and humans |
| [`challenge_start`](challenge_start.md) | Start a challenge run: builds a seeded instance, opens its sessions with the challenge's access level and budget, and returns the brief, parameters and answer format. agent_model = the model you are (e.g. 'claude-fable-5-2'). Leave seed empty for a random instance. | agents (MCP) and humans |
| [`challenge_submit`](challenge_submit.md) | Submit your answer for a challenge run (once). Returns the score. | agents (MCP) and humans |
| [`challenge_scores`](challenge_scores.md) | Challenge leaderboard by model and harness. | agents (MCP) and humans |

## Direction specs

Many tools take a `direction`: see [`wsp_toolkit/core/directions.py`](../../wsp_toolkit/core/directions.py) — `{"id": "dir_…"}`, `{"tensor", "row"|"col"|"svd"}`, `{"token", "matrix"}`, `{"vector"}`, `{"combine"}`.

## Edit operations

`{"op": "project_out_direction", "tensor", "direction", "side": "output"|"input", "indices"?, "strength"?}` · `{"op": "scale", "tensor", "head"|"slice"?, "factor"}` · `{"op": "zero", ...}` · `{"op": "set_slice", "tensor", "slice", "values"}` · `{"op": "add_low_rank", "tensor", "u": [dir], "v": [dir], "coeffs"?}` or explicit `U`/`V` · `graft_subspace` (stub).
