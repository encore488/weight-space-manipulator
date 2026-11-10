"""M3 end-to-end demo on Qwen3.5-0.8B: base vs. the authors' WSP edit, plus non-edit baselines.

Designed for a GPU box (Kaggle "GPU T4 x2" works; see notebooks/kaggle_qwen_m3_demo.ipynb). Steps:
  1. Load base Qwen/Qwen3.5-0.8B (pinned to the commit the edit was made from) and
     RikiyaT/Qwen3.5-0.8B-weight-space-programming. Evaluate both under the paper's settings on a MATH-500 / GSM8K
     subset: accuracy, token-limit hit rate, loop rate.
  2. Non-edit baselines on the base model (repetition penalty, presence penalty, temperature).
  3. (Separate, interactive) an agent with a 50-pass budget on practice data: `wsp mcp --budget 50`.

Modes
  --smoke                 one batch of problems, 512 new tokens, prints measured tokens/second (sizes the real run)
  --runs base,edited,...  which runs to do in this process (split across GPUs with CUDA_VISIBLE_DEVICES)
  --report-from A.jsonl B.jsonl [--with-diff]   merge result files into docs/reports/m3_qwen_demo.md

Each finished run is appended to --results-jsonl immediately, so a session that dies keeps completed runs.
Generation uses Hugging Face generate with batching (not vLLM like the paper); numbers are reported as measured.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

# overridable for testing the pipeline on small local models
BASE = os.environ.get("WSP_DEMO_BASE", "hf:Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17")
EDITED = os.environ.get("WSP_DEMO_EDITED", "hf:RikiyaT/Qwen3.5-0.8B-weight-space-programming")
PAPER = {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
         "max_new_tokens": 32768}
BASELINES = {"repetition_penalty_1.1": {"repetition_penalty": 1.1},
             "presence_penalty_2.0": {"presence_penalty": 2.0},
             "temperature_1.0": {"temperature": 1.0}}
ALL_RUNS = "base,edited," + ",".join(f"baseline:{k}" for k in BASELINES)


def run_evals(a) -> None:
    os.environ.setdefault("WSP_HF_DEVICE", a.device)
    os.environ.setdefault("WSP_HF_DTYPE", a.dtype)
    os.environ.setdefault("WSP_HF_ATTN", a.attn)
    from wsp_toolkit.config import Config
    from wsp_toolkit.core.service import Toolkit

    tk = Toolkit(Config.load(overrides={"eval": {"batch_size": a.batch_size}}))
    t_run = [time.time()]

    def progress(done, total):
        print(f"    {done}/{total} problems, {time.time() - t_run[0]:.0f}s elapsed", flush=True)
    tk.harness.progress = progress

    runs = a.runs.split(",")
    need = {"base": any(r != "edited" for r in runs), "edited": "edited" in runs}
    sessions = {}
    for name, src in (("base", BASE), ("edited", EDITED)):
        if need[name]:
            o = tk.call("workspace_open", {"source": src, "label": f"Qwen3.5-0.8B {name}",
                                           "confirm_download": a.confirm_download}, actor="human")
            if "error" in o:
                raise SystemExit(json.dumps(o, indent=1))
            sessions[name] = o
    # smoke: one full batch of problems at 512 tokens, to measure batched throughput on this hardware
    limit, max_new = (a.batch_size, 512) if a.smoke else (a.limit, a.max_new_tokens)
    settings = {**PAPER, "max_new_tokens": max_new, "samples": a.samples}
    out = Path(a.results_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.smoke:  # warm-up so kernel compilation / autotuning is not counted as generation time
        ws = tk.wsm.get(sessions["base" if need["base"] else "edited"]["workspace_id"])
        tk.harness.run(ws, a.suites.split(",")[0], kind="baseline", settings={**settings, "max_new_tokens": 32},
                       limit=limit, use_cache=False)
    for suite in a.suites.split(","):
        for run in runs:
            model = "edited" if run == "edited" else "base"
            extra = BASELINES[run.split(":", 1)[1]] if run.startswith("baseline:") else {}
            ws = tk.wsm.get(sessions[model]["workspace_id"])
            print(f"[{suite}] {run}: {limit} problems, max_new_tokens={max_new}, batch={a.batch_size}", flush=True)
            t_run[0] = time.time()
            r = tk.harness.run(ws, suite, kind="baseline" if extra else "heldout", settings={**settings, **extra},
                               limit=limit, session_id=sessions[model]["session_id"], use_cache=not a.smoke)
            secs = time.time() - t_run[0]
            m = r["metrics"]
            toks = m.get("mean_new_tokens", 0) * m["n"] * a.samples
            # decode steps/s is what predicts run time with batching: a batch runs until its longest member stops
            steps = max_new if m.get("hit_token_limit_rate", 0) > 0 else m.get("mean_new_tokens", 0)
            row = {"suite": suite, "run": run, "model": model, "run_id": r["run_id"], **m, "seconds": round(secs, 1),
                   "tokens_per_second": round(toks / secs, 1) if secs else None,
                   "decode_steps_per_second": round(steps * math.ceil(limit / a.batch_size) / secs, 2) if secs else None,
                   "settings": r["settings"],
                   "env": r["env"], "smoke": a.smoke}
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("suite", "run", "score", "hit_token_limit_rate",
                                                     "loop_rate", "mean_new_tokens", "seconds",
                                                     "tokens_per_second", "decode_steps_per_second")}), flush=True)


def report(a) -> None:
    rows = []
    for p in a.report_from:
        rows += [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    rows = [r for r in rows if not r.get("smoke")]
    L = ["# M3 demo: Qwen3.5-0.8B base vs. WSP edit", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Hardware: {sorted({r['env'].get('gpu', 'cpu') for r in rows})}. "
         "Hugging Face generate with batching (the paper used vLLM).", ""]
    if a.with_diff:
        from wsp_toolkit.core.service import Toolkit
        tk = Toolkit()
        s1 = tk.call("workspace_open", {"source": BASE, "confirm_download": True}, actor="human")
        s2 = tk.call("workspace_open", {"source": EDITED, "confirm_download": True}, actor="human")
        d = tk.call("diff_models", {"session_id": s1["session_id"], "other_session_id": s2["session_id"]},
                    actor="human")
        L += ["## Weight diff", "", f"{d['params_changed']} parameters changed in {d['tensors_changed']} tensors "
              "(model card: 131,072 weights in layers 15 and 19 down_proj).", ""]
        for t in d["most_changed_tensors"]:
            L.append(f"- `{t['tensor']}`: {t['changed']} values, rank for 90% energy {t.get('rank_90')}, "
                     f"most-changed columns {[c[0] for c in t.get('top_cols', [])]}")
        L.append("")
    L += ["## Results", "", "| Benchmark | Run | n | Accuracy | ±SE | Hit token limit | Loop rate | Mean new tokens | "
          "max_new_tokens |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["suite"], ["base", "edited"].index(r["run"]) if r["run"] in
                                         ("base", "edited") else 2, r["run"])):
        L.append(f"| {r['suite']} | {r['run']} | {r['n']} | {r['score']:.3f} | {r['stderr']:.3f} | "
                 f"{r.get('hit_token_limit_rate', 0):.3f} | {r.get('loop_rate', 0):.3f} | "
                 f"{r.get('mean_new_tokens', 0):.0f} | {r['settings']['max_new_tokens']} |")
    L += ["", "Paper (blog), for reference only: MATH-500 7.9 → 48.1, GSM8K 13.8 → 67.1, hit-token-limit 93.0 → 33.1 "
          "(full benchmarks, vLLM, multiple samples, math-verify grading). Differences are reported, not adjusted."]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm-download", action="store_true", help="allow the ~1.6 GB downloads per model")
    ap.add_argument("--device", default="cuda", help="cuda | mps (Apple silicon) | cpu")
    ap.add_argument("--dtype", default="float32", help="float32 (safe on T4/P100) | float16 | bfloat16 (A100/H100/Mac)")
    ap.add_argument("--attn", default="sdpa", help="sdpa (fast) | eager")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=30, help="problems per benchmark")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=PAPER["max_new_tokens"])
    ap.add_argument("--suites", default="math500_heldout")
    ap.add_argument("--runs", default=ALL_RUNS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--results-jsonl", default="m3_results.jsonl")
    ap.add_argument("--report-from", nargs="*")
    ap.add_argument("--with-diff", action="store_true")
    ap.add_argument("--out", default="docs/reports/m3_qwen_demo.md")
    a = ap.parse_args()
    if a.report_from:
        report(a)
    else:
        run_evals(a)


if __name__ == "__main__":
    main()
