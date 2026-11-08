# Patch semantics ("git for weights")

## What a patch is
- **Manifest** (SQLite `patches.manifest_json`): patch ID, workspace, base model hash, parent patch, resolved
  operations (directions stored by ID), per-tensor `hash_before`/`hash_after`, elements changed, % of model,
  model hash before/after, rationale, author, timestamp, linked evaluation runs. Status and flags sit in their
  own columns.
- **Delta** (`<data>/patches/<id>.safetensors`): for each changed tensor `T`, `T::idx` (flat int64 indices of
  changed elements), `T::old` and `T::new` (exact values, original dtype).

## Why values, not differences
Writing exact old/new values makes `apply → revert` and `replay on a fresh base` **bit-exact**. Adding and
subtracting float deltas would not be. Each apply/revert checks the tensor hash before and after; a mismatch
aborts with `hash_mismatch`, so a patch never lands on the wrong state.

## Stack
A workspace has an ordered stack `[p1, p2, …]`; each patch's `parent` is the patch below it when it was created.
The working copy is always `base + stack` (any process can rebuild it from SQLite + deltas).

| Operation | Result |
|---|---|
| `edit_apply` | computes the ops on the current working copy, stores the delta, pushes the new patch |
| `patch_revert(P)` (cascade, default) | pops P **and every patch above it**; the working copy equals the state before P. Always exact. |
| `patch_revert(P, mode="rebase")` | pops P and everything above, then **re-derives** each patch above P by re-running its *operations* on the new state. They get new IDs (their parent changed); the originals become `superseded`. Ops whose directions depended on the old state (e.g. an SVD direction of a tensor P changed) are re-derived against the new state; that is the intended meaning of "rebase". |
| `patch_reapply(P)` | re-pushes an existing patch whose parent is the current top (e.g. after a cascade revert) |

## Status machine
`draft` → (`patch_test` / automatic regression run) → `tested` → human `approved` | `rejected`.
`superseded` = replaced by a rebase. Only `tested` patches can be approved; export requires every patch in the
chain to be `approved` with a logged approval row.

## IDs
`p_` + sha256 over (workspace, base hash, parent, resolved ops, per-tensor before/after hashes), first 16 hex.
The same edit on the same parent in the same workspace gives the same ID, and the row is reused.
