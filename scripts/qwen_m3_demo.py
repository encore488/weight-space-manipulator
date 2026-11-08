"""M3 end-to-end demo on Qwen3.5-0.8B (needs ~3.2 GB of downloads and, for meaningful runs, a GPU).

Steps (roadmap M3):
  1. Load base Qwen/Qwen3.5-0.8B and the authors' edited RikiyaT/Qwen3.5-0.8B-weight-space-programming.
     Confirm with diff_models that only the two down_proj tensors differ (131,072 weights per the model card).
     Evaluate both under matching settings on MATH-500 / GSM8K subsets: accuracy and token-limit hit rate.
  2. Run non-edit baselines on the base model (repetition penalty, presence penalty, temperature).
  3. (Separate) give an agent a 50-pass budget on practice data via MCP:
        wsp mcp --budget 50 --access weights+activations+outputs
     and ask it to open hf:Qwen/Qwen3.5-0.8B and use gsm8k_practice / math_practice. For the paper's
     budget unit (1 pass = up to 512 problems) set session.inputs_per_pass = 512 in a config file.

Usage:
  python scripts/qwen_m3_demo.py --confirm-download --device cuda --dtype bfloat16 --limit 50
  python scripts/qwen_m3_demo.py --confirm-download --limit 3 --max-new-tokens 512      # CPU smoke run

Generation uses Hugging Face generate (not vLLM, unlike the paper), so full 32k-token runs are slow; numbers will
differ from the paper's and are reported as measured. Paper settings: thinking on, temperature 0.6, top-p 0.95,
top-k 20, presence penalty 1.5, max 32,768 new tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

BASE = "hf:Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17"
EDITED = "hf:RikiyaT/Qwen3.5-0.8B-weight-space-programming"
PAPER = {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
         "max_new_tokens": 32768}
BASELINES = {"repetition_penalty_1.1": {"repetition_penalty": 1.1},
             "presence_penalty_2.0": {"presence_penalty": 2.0},
             "temperature_1.0": {"temperature": 1.0}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-download", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=20, help="problems per benchmark (subset first; full needs a GPU)")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=PAPER["max_new_tokens"])
    ap.add_argument("--suites", default="math500_heldout,gsm8k_heldout")
    ap.add_argument("--out", default="docs/reports/m3_qwen_demo.md")
    a = ap.parse_args()
    os.environ["WSP_HF_DEVICE"], os.environ["WSP_HF_DTYPE"] = a.device, a.dtype

    from wsp_toolkit.core.service import Toolkit
    tk = Toolkit()
    call = lambda tool, **kw: tk.call(tool, kw, actor="human")  # noqa: E731
    base = call("workspace_open", source=BASE, label="Qwen3.5-0.8B base", confirm_download=a.confirm_download)
    edit = call("workspace_open", source=EDITED, label="Qwen3.5-0.8B WSP edit", confirm_download=a.confirm_download)
    for o in (base, edit):
        if "error" in o:
            raise SystemExit(json.dumps(o, indent=1))
    diff = call("diff_models", session_id=base["session_id"], other_session_id=edit["session_id"])
    settings = {**PAPER, "max_new_tokens": a.max_new_tokens, "samples": a.samples}
    rows = []
    for suite in a.suites.split(","):
        for label, sess, extra, model in [("base", base, {}, "base"), ("edited", edit, {}, "edited")] + \
                [(f"baseline {k}", base, v, "base") for k, v in BASELINES.items()]:
            t0 = time.time()
            r = tk.harness.run(tk.wsm.get(sess["workspace_id"]), suite, kind="heldout" if not extra else "baseline",
                               settings={**settings, **extra}, limit=a.limit, session_id=sess["session_id"])
            rows.append({"suite": suite, "run": label, "model": model, "run_id": r["run_id"], **r["metrics"],
                         "seconds": round(time.time() - t0, 1), "settings": r["settings"]})
            print(json.dumps(rows[-1]))
    L = ["# M3 demo: Qwen3.5-0.8B base vs. WSP edit", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} on "
         f"{a.device}/{a.dtype}; {a.limit} problems per benchmark, {a.samples} sample(s), max_new_tokens "
         f"{a.max_new_tokens}.", "", "## Weight diff", "",
         f"{diff['params_changed']} parameters changed in {diff['tensors_changed']} tensors:", ""]
    for t in diff["most_changed_tensors"]:
        L.append(f"- `{t['tensor']}`: {t['changed']} values, rank-90 {t.get('rank_90')}, cols {t.get('top_cols')}")
    L += ["", "## Results", "", "| Benchmark | Run | Accuracy | ±SE | Hit token limit | Loop rate | Mean new tokens |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['suite']} | {r['run']} | {r['score']:.3f} | {r['stderr']:.3f} | "
                 f"{r.get('hit_token_limit_rate', 0):.3f} | {r.get('loop_rate', 0):.3f} | {r.get('mean_new_tokens', 0):.0f} |")
    L += ["", "Paper (blog) for reference: MATH-500 7.9 -> 48.1, GSM8K 13.8 -> 67.1, hit-token-limit 93.0 -> 33.1 "
          "(full benchmarks, vLLM, multiple samples). Differences are expected and are not adjusted."]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
