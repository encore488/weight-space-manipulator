# Decisions & assumptions log (for owner review)

> **Status (2026-11-09): reviewed and approved by the owner.** Ownership confirmed as a **personal project**
> (not PAI3). License: Apache-2.0 (`LICENSE` added).

You asked me to build all milestones without stopping and to write down every assumption and business-logic
decision. This file is that list. Each entry says what I decided, why, and what to change if you disagree.
Entries marked **[REVIEW]** are the ones most likely to need your input.

---

## 0. Open questions from the roadmap (§8): what I assumed

| # | Question | What I assumed | How to change |
|---|---|---|---|
| 1 | Name & license | Kept the working name **WSP Toolkit** (package `wsp_toolkit`, CLI `wsp`). License **Apache-2.0** (`LICENSE` file + `pyproject.toml`). Approved. | Rename the package/CLI if you want a different name. |
| 2 | Personal vs PAI3 | **Personal project** (confirmed by the owner). Repo: `encore488/weight-space-manipulator`. | n/a |
| 3 | Hardware | **No GPU** in the build environment (4 CPU cores, 15 GB RAM). Everything except the real Qwen demo runs on CPU. `WSP_HF_DEVICE=cuda` and `WSP_HF_DTYPE=bfloat16` switch HF models to GPU. | — |
| 4 | UI stack | **FastAPI + Jinja2 + HTMX**, as suggested. HTMX 1.9.12 is vendored in `ui/static/` (no CDN at runtime → local-first). | — |
| 5 | Agent for demos | No LLM API key was available to the toolkit. Demos use (a) **scripted agents** that only call the tools (deterministic, used as acceptance tests), and (b) one **LLM-driven session**: I (Claude, the model building this) used the tools through `wsp call`, which has exactly the MCP agent permissions. That session is logged and in the reports. **[REVIEW]** | Point any MCP client (Claude Code, Codex) at `wsp mcp`. |

---

## 1. Architecture & dependencies

1. **Layout.** The roadmap's `core/ mcp_server/ api/ ui/ fixtures/` live inside one installable package
   `wsp_toolkit/` (so `pip install -e .` works), with `tests/`, `docs/` and `scripts/` at the repo root. Scripted
   agents are in `wsp_toolkit/agents/`.
2. **Thin layers.** All behaviour is in `wsp_toolkit/core/service.py::Toolkit` and its modules. MCP and HTTP both
   call `Toolkit.call(tool, args, actor=...)`. MCP tool schemas are generated from the Toolkit method signatures,
   so the two can't drift.
3. **MCP SDK.** The installed SDK is **mcp 2.3.0**, where `FastMCP` became `mcp.server.mcpserver.MCPServer`.
   I checked the real API rather than relying on memory (principle 7). The `mcp<2` API will not work. Tool
   errors are raised as `ToolError`, so MCP clients see `isError: true` with the JSON error body.
4. **nnsight / TransformerLens: neither adopted.** The roadmap asked to evaluate both in M1. I used **plain
   PyTorch forward hooks** behind a `ModelAdapter` interface (`core/workspace/adapters.py`). Reasons: (a) neither
   was installed, and both pin heavy and fast-moving dependencies; (b) our fixtures are built with hook points
   (TransformerLens style) and HF models expose clean module paths; (c) we need the same interface over native
   fixtures, HF models and raw tensor bags. Swapping in nnsight later only means writing another adapter.
   **[REVIEW]**
5. **Storage.** SQLite (WAL) for metadata, logs, runs, patches, features and audits. Safetensors for model bases,
   patch deltas and SAE weights. There is no general blob store and no vector index. Feature search is lexical
   (see §6). `sqlite-vec`/LanceDB were not needed at this scale.
6. **Python ≥ 3.11**, tested on 3.13 with torch 2.15 (CPU), transformers 5.22. HF support is an optional extra
   (`pip install -e '.[hf]'`).

## 2. Workspaces, hashing, sessions (M0/M1)

1. **Model hash** = sha256 over sorted `(tensor name, sha256(dtype, shape, raw bytes))`. It doesn't depend on
   file format, so a re-saved model with the same values has the same hash. IDs shown to agents are 16-hex
   prefixes; full hashes are stored.
2. **Never modify originals.** Small native models are copied into a content-addressed `bases/<digest>/` folder
   with files chmod 0444. HF models are used **in place from the HF cache** (copying GBs is wasteful). They are
   protected by re-verifying the hash every time a workspace is rebuilt.
3. **Working copy = in-memory model.** Persistent state = base hash + ordered patch stack in SQLite. Any process
   (MCP server, UI) rebuilds the same working copy and re-syncs when another process changes the stack.
4. **Sessions bind an agent to a workspace** with an access level and a forward-pass budget. **The ceiling comes
   from the human's config / `wsp mcp --access --budget`; an agent may request less, never more.** **[REVIEW]**
5. **Access levels** map to capability sets: `weights_only` = {weights};
   `weights+activations` = {weights, activations} (activations **and attention patterns**);
   `weights+activations+outputs` adds model outputs (logits, generations, per-example practice results).
   Attention counts as "activations" (the paper's Gemma experiment added attention on top of weights).
   **[REVIEW]**
6. **Budget unit:** by default **1 pass = 1 input** (generation of one sample = 1 pass, whatever its length).
   The paper counted one pass as a batch of up to 512 problems, so `session.inputs_per_pass` (default 1)
   reproduces that (set 512). Budget is charged atomically *after* inputs validate and *before* running.
   An over-budget call fails with no partial spend. Weight-only tools are free. **[REVIEW]**
7. **Held-out evaluations do not spend the probe budget.** They are rate-limited instead
   (`heldout_rate_limit_per_hour`, default 5 per session) and logged. **[REVIEW]**
8. **Regression suites run on every patch automatically** (`policy.auto_regression_on_apply`). They are system
   actions and are not charged to the agent.
9. **Downloads** above `limits.download_confirm_gb` (1 GB) need `confirm_download=true`. I did **not** download
   Qwen3.5-0.8B (~1.6 GB per checkpoint) under the roadmap's "ask before >1 GB" rule (see §4).
10. **Never edit the calling agent's weights (hard rule).** `workspace_open` refuses when the source equals the
    agent's declared `agent_model`, or when the source/hash is in `policy.forbidden_models`. This can only check
    what it's told: an agent that lies about its identity isn't caught. **[REVIEW]**

## 3. Patches (M2)

1. **Delta format:** per changed tensor, the flat indices of changed elements plus their exact **old and new
   values** (safetensors). Applying writes new values; reverting writes old ones. That makes apply→revert and
   replay-on-fresh-base **bit-exact** (both are tested). Additive deltas would not be bit-exact in floating
   point.
2. **Patch ID** = hash of (workspace, base hash, parent, resolved operations, per-tensor before/after hashes).
   I added the workspace after finding that identical edits in two workspaces collided on one patch row.
3. **Stacking semantics:** `revert(P, mode="cascade")` (default) pops P and every patch above it.
   `mode="rebase"` pops them, then **re-derives** the patches above P from their operation lists on the new
   state. They get new IDs and the originals are marked `superseded`. See `docs/design/patch_semantics.md`.
   **[REVIEW]**
4. **Rationale is mandatory** (≥ 3 characters) on `edit_apply`.
5. **Status machine:** `draft → tested → approved | rejected` (plus `superseded`). A patch becomes `tested` when
   the required suites ran (automatically on apply). **Only `tested` patches can be approved**, and only by a
   human. **Export requires every patch in the chain to be approved** with a logged approval row.
6. **Op semantics:** for an `nn.Linear` weight `[out, in]`, `side:"output"` = axis 0 (rows), and
   `project_out_direction` does `W[:, idx] -= s · d (dᵀ W[:, idx])`. This matches the Qwen fix ("remove a
   direction from 64 output columns of down_proj"). `head` selects a head's rows (q/k/v) or columns (o_proj),
   handling GQA and Qwen3.5's gated `q_proj`.
7. **`graft_subspace` is a stub** with a design note (`docs/design/graft_subspace.md`).

## 4. Evaluation harness (M3)

1. **Split enforcement in code:** practice examples are visible (`suite_list` preview, paged `suite_examples`).
   Held-out and regression examples are never returned by any agent tool. Held-out runs return aggregates only,
   even through `patch_show`.
2. **Every run records** model hash, generation settings (temperature, top-p, top-k, max tokens, repetition and
   presence penalty, thinking, seed, samples), library versions and the network-guard mode. Runs are cached by
   (model hash, suite, settings, limit).
3. **Network isolation:** evaluation runs under a **Python-level guard** that blocks outbound connects and DNS
   *for the evaluating thread* (default), or in a **subprocess** with the guard on for the whole process
   (`eval.isolation = "subprocess"`). Native code that opens sockets directly is not covered. For adversarial
   settings use OS isolation (`unshare -n`, a network-less container). **[REVIEW]**
4. **Presence penalty** (used by the paper) isn't built into `transformers`; I added a logits processor
   (vLLM/OpenAI semantics, prompt excluded).
5. **Loop detection:** a generation "looped" if its tail is one block of period ≤ 64 tokens repeated ≥ 3 times
   over ≥ 24 tokens. Token-limit hits are tracked separately (`hit_token_limit_rate`), as in the paper.
6. **Benchmarks:** GSM8K (`openai/gsm8k`), MATH-500 (`HuggingFaceH4/MATH-500`), MATH train
   (`EleutherAI/hendrycks_math`) are loaded on demand and cached. Scoring: last number (GSM8K) / last
   `\boxed{}` with light normalisation (MATH). **This is not `math-verify`**, which the authors used, so expect
   small differences. **[REVIEW]**
7. **Qwen M3 demo not executed here.** Reasons: no GPU; ~3.2 GB of downloads (roadmap says ask first); and
   32k-token thinking generations through HF `generate` on 4 CPU cores would take days. `scripts/qwen_m3_demo.py`
   is ready (`--confirm-download --device cuda --limit 50`). What I verified offline: the real Qwen3.5 config
   (fetched `config.json` only, a few KB) instantiated at tiny size loads through the adapter with the paper's
   tensor names, hybrid linear/full attention, gated q_proj, zero-centred RMSNorm and tied embeddings, and the
   loop-fix-style edit applies and reverts exactly (`tests/test_qwen35_arch.py`). **[REVIEW]**
7a. **GPU route (decided 2026-11-09): Kaggle, "GPU T4 x2".** The owner's MacBook Pro (base Apple M5) has an
    integrated GPU that PyTorch can use (`--device mps`, untested here) — fine for smoke tests, far too slow for
    paper-scale 32k-token runs. Claude cannot rent GPUs, so the run is packaged as
    `notebooks/kaggle_qwen_m3_demo.ipynb` for the owner to execute. Choices made for it: **float32** (T4 has no
    bfloat16 and Qwen can overflow in float16); **batched generation** (`eval.batch_size`, added for this) with SDPA
    attention; base + 2 baselines on GPU 0, edited + 1 baseline on GPU 1; a measured-throughput smoke step that
    shrinks `LIMIT` (default 30 MATH-500 problems, 1 sample) to fit a 9-hour budget. This is a **subset** of the
    paper's evaluation (500 problems × 4 samples, vLLM), so numbers will be noisier. **[REVIEW]**
8. **"Recommended temperature" baseline:** Qwen's recommended thinking-mode sampling *is* the paper's setting
   (T=0.6, top-p 0.95, top-k 20), so the demo's baselines vary from it: repetition penalty 1.1, presence
   penalty 2.0, temperature 1.0.
9. **Fixture M3 demo** (CPU, runs here): the agent fixes `induction_tf_broken` under a 50-pass budget with
   practice data only. Because the fixture is tiny (38K params; one head's `o_proj` block is 2.7% of the model),
   the demo raises `max_param_change_pct` to 20%. Real models keep 5%. **[REVIEW]**

## 5. Safety (M4)

1. **Approval is human-only by construction:** `patch_approve`, `patch_reject`, `sensitive_register` and
   `zoo_build` are not exposed over MCP, and are refused through `/api/tools/*`. They work only from the UI forms,
   `/api/patches/{id}/approve|reject`, or `wsp approve`. Approvals are an append-only table (reviewer, time,
   note, flag acknowledgement).
2. **Local-first limitation:** an agent with shell access on the same machine could call the HTTP endpoint or
   edit SQLite. Set `[ui] reviewer_token` so approvals need a token the agent doesn't have. True separation
   needs OS-level isolation (different user / container). **[REVIEW]**
3. **The event log is append-only**, enforced by SQLite triggers that reject UPDATE/DELETE.
4. **Policy defaults:** max 5% of parameters per patch; primitives
   `project_out_direction, scale, zero, set_slice, add_low_rank`; required suites = all regression suites;
   held-out 5 per hour; budget ceiling 50. Per-session policy can only tighten. **[REVIEW]**
5. **Sensitive registry:** a patch is flagged when it touches a registered tensor (exact name or `re:` regex), or
   when a registered direction/feature carries ≥ `sensitive_direction_cosine` (0.3) of the delta's energy along
   that axis, or when an op's direction has that cosine with it. Flagged patches need an explicit acknowledgement
   checkbox to approve. **[REVIEW]** (the threshold is a guess)

## 6. Network zoo (M5)

1. **v0 zoo = 392 networks** (102 hidden split): 240 boolean MLPs (10 functions × 3 seeds × {healthy, silenced
   input, shifted threshold, backdoor} × {original, symmetry copy}), 12 hand-compiled MLPs (exact weights; I
   used hand compilation instead of **Tracr**, which needs JAX), 54 CNNs (3 image tasks), 32 one-layer
   transformers (4 algorithmic tasks), 54 RL policy MLPs (behaviour-cloned from analytic controllers, not RL
   trained). Builds in ~40 s on CPU. **[REVIEW]**
2. **Network IDs are opaque** (`z` + hash). The first version used descriptive IDs, which **leaked the answers**
   to any agent calling `zoo_list_tasks`. Found and fixed before the final runs.
3. **Hidden split** = deterministic 25% by hash. Hidden submissions return no per-item score; only the
   leaderboard aggregates them. Answers live in `answers.json` next to each network. That is "server-side" for an
   MCP agent, but a local agent with file access could read them. **[REVIEW]**
4. **Scoring:** identify = exact label; localize = 0.5 for the right flaw type + 0.5 × location match (Jaccard
   for backdoor triggers; 1.0 for a correct "none"); repair = `max(0, (after − before) / (1 − before))` on the
   held-out suite of the intended behaviour, with the per-patch % limit **not** applied (tiny networks).
5. **Repair is only offered when the flaw actually hurts** (intended accuracy < 0.97). Single dead CNN channels
   turned out to be functionally silent (4 redundant channels), so they are localize-only.
6. **Practice data for zoo nets** shows the *intended* behaviour on 16–32 inputs. With `outputs` access that makes
   "identify" easy; the leaderboard records each submission's access level. Weights-only conditions are the real
   test.
7. **Baseline agent** = a scripted heuristic agent (behavioural probing + weight statistics), not an LLM.
   Its scores are in `docs/reports/milestones.md`.

## 7. Feature index (M6)

1. **No Gemma Scope ingestion run.** It needs Gemma 2 weights (license gating, multi-GB). I trained SAEs on the
   fixture model instead. An ingestion path for Gemma-Scope-style `params.npz` / safetensors SAEs (JumpReLU
   thresholds supported) exists but is **untested on real files**. **[REVIEW]**
2. **SAEs:** ReLU SAE, unit-norm decoder, L1 = 0.3 (on inputs normalised to unit mean norm), dead-feature
   resampling. The first version (L1 = 0.003) gave dense codes with a deceptively low unexplained fraction. I
   noticed and fixed it before reporting.
3. **Labels come from a deterministic predicate labeler**, not an LLM: it scores ~200 token-level hypotheses
   (current/previous/next token, position, "repeats an earlier token", sequence type) and picks the best by F1.
   **Fidelity = that label's F1 at predicting firing on a held-out split.** An LLM labeler can plug into the same
   interface. **[REVIEW]**
4. **Neurons** in the neuron↔feature mapping are the basis units of the hooked space (residual dims for
   `resid_post`, MLP neurons for `mlp_hidden`). The mapping stores decoder weights (feature→unit) and encoder
   weights (unit→feature).
5. **Semantic search is lexical** (IDF-weighted token overlap on labels). There is no embedding model, to keep
   it offline and dependency-free.
6. **"What uses these weights?"** scores each feature direction by how much the selected weights write along it
   (residual-writing tensors) or read it (residual-reading tensors).

## 8. Fine-tune audit (M7)

1. **Three lenses:** exact weight deltas (with SVD rank structure); feature mapping of top delta directions; and
   behaviour (suite deltas + a divergence scan ranking tokens/inputs over-represented where the models disagree,
   plus each token's mean log-probability shift).
2. **Singular directions have no sign.** The report shows "poles", not promotes/suppresses. My first validator
   wrongly counted a sign-ambiguous readout as identifying a suppressed token. I caught that and replaced it with
   the sign-robust mean log-prob shift.
3. **Validation set:** 5 planted fine-tunes of the induction fixture (trigger, "hint-following"
   sycophancy-analogue, token suppression, capability loss, control) + 6 backdoor fine-tunes of zoo MLPs.
   Result: **11/11 identified**. Caveat: these planted behaviours are strong and the probe set covers them
   (uniform random tokens / all 256 binary inputs). Subtle or rare-trigger behaviours would be harder. A real
   sycophantic-vs-honest LLM pair (like the paper's Gemma experiment) was **not** tested. **[REVIEW]**
4. **PDF export** via reportlab (optional extra). Markdown is the canonical format.

## 9. Things I deliberately did not do

- No downloads > 1 GB, no Gemma (license), no GPU runs.
- No backlog items (§5 of the roadmap), and nothing from "out of scope" (§6).
- No pull request (not requested). Work is pushed to `chris/vibrant-hamilton-286z89`.

---

## 10. v0.2 launch preparation (2026-11-10): new decisions for review

1. **Challenges** (`wsp_toolkit/core/challenges.py`): five fixed, auto-scored tasks with a per-model leaderboard.
   The answers are fixed, so a determined person can share them. That's acceptable for a demo leaderboard, not a
   benchmark. The `mystery-finetune` suspect uses a different trigger (t19 → t4) from the one published in
   `docs/reports/milestones.md` (t7 → t13). **[REVIEW]**
2. **Harness labels instead of trust.** Runs via `wsp agent run` are labelled `wsp-agent-runner`; runs via MCP are
   `mcp (self-reported)`; servers started without `--challenge` mark runs *unlocked*. Nothing stops a shell-capable
   agent from reading answer files, so the labels state what each number can be trusted for.
3. **Access-control tightening.** Evaluation results (practice, held-out, baselines, `compare_runs`, regression
   summaries from `edit_apply` / `patch_test`, metrics in `patch_show`) now require the `outputs` capability. Before
   this, a `weights_only` session could learn behavioural scores through those tools, which contradicted the paper's
   weights-only condition. Humans (UI/CLI) still see everything.
4. **Server lock** (`session.allowed_sources`, set by `wsp mcp --challenge`): agents may only open the listed models.
5. **Per-patch % limit exempts tiny models** (`policy.pct_limit_min_params = 1,000,000`). In a 38K-parameter model one
   attention head is ~2.7% of the model, so the 5% limit made reasonable fixes impossible. Real models keep the limit.
   Replaces the earlier per-demo config override and the zoo-scoring workaround. **[REVIEW]**
6. **API runner defaults** (`wsp agent run`):
   - **Model and effort:** default model `claude-fable-5-2`, effort `high` (the recommended starting point for
     Fable 5.2).
   - **Thinking:** adaptive thinking with summarized display, so transcripts show reasoning summaries.
   - **Refusals:** server-side refusal fallback (`fallbacks: "default"`) on models that support it, so a classifier
     decline doesn't end a run.
   - **Limits:** max 80 turns and $20 estimated cost per run. **Cost estimates use list prices and are approximate**:
     fallback turns may bill at another model's rates. **[REVIEW]**
7. **`wsp challenge prepare`** builds all challenge assets up front (fixtures, the 392-network zoo, the mystery
   fine-tune), so the first MCP tool call doesn't spend a minute building them and time out.
8. **Install path:** `uv tool install "wsp-toolkit[agent] @ git+https://github.com/encore488/weight-space-manipulator"`
   (uv provides Python 3.11+ on machines that lack it). Verified from a clean environment. No PyPI release yet.
   **[REVIEW]**

## 11. WSP Bench, website and public leaderboard (2026-11-10)

1. **Seeded instances.** Every challenge run builds its instance from `(challenge, seed)`. This replaces the fixed
   instances of v0.2, whose answers could be shared. Heads are permuted (function-preserving) per seed; damage kind
   (shrink / sign-flip / mixed), banned and trigger tokens, the backdoor's trigger pattern and the zoo networks all
   vary. Because instances are seeded, a submission can be re-scored by anyone.
2. **Four new challenges** (nine in total): `token-ban` and `install-a-rule` (*write* tasks: program behaviour, not
   just repair), `name-that-function` (weights-only reading, including permuted copies, the critique paper's point
   about priors vs. tracing), and `backdoor-removal`. Each was checked: doing nothing ≈ 0, and a solution built from
   the hidden truth reaches ≈ 1 (≈ 0.67 for my own `install-a-rule` attempt, which leaves headroom). `token-ban` turned
   out to have a one-edit solution, so it is level 1. **[REVIEW]**
3. **WSP-Bench score** = mean of the nine challenge means; missing challenges count 0 (shown as *partial*). Equal
   weights per challenge, no level weighting. **[REVIEW]**
4. **Website on GitHub Pages**, a static site built by `.github/workflows/site.yml` on every push to `main`. It
   re-scores every submission at build time, so the "re-scored ✓" badge always reflects the current code. **Needs
   one manual step from you**: Settings → Pages → Source: *GitHub Actions*. Until then the deploy job fails.
5. **Contributions:** an issue form, with a bot that re-scores the pasted submission and comments, and a
   maintainer `/accept` comment that commits it and rebuilds the site. The alternative is a pull request checked by
   `leaderboard-pr.yml`, which runs main's verifier on the PR's data files only.
   - **Why no PR auto-creation:** that would need the "Allow GitHub Actions to create pull requests" setting.
   - **Why the extra rebuild trigger:** pushes made with the default token don't trigger other workflows, so
     intake triggers the site build explicitly.
   **[REVIEW]**
6. **What verification can't do:** it proves a score follows from the submitted answer and edits on that seed's
   instance. It can't prove which model produced them, or that budgets were respected; someone could compute an
   answer from the public code. The site says this plainly. A stronger design would need trusted execution (e.g. a
   hosted runner with its own API key), which is out of scope for a local-first project.
7. **Score tolerance ±0.05** in verification: instances include small trained models whose weights can differ across
   platforms and PyTorch versions. Locally, re-scoring from scratch reproduced all 27 baseline runs exactly.
8. The scripted baseline (seeds 1–3) is the first leaderboard entry, labelled as a reference, not an AI.
