"""Benchmark loaders (network needed the first time; cached under <data_dir>/benchmarks).

Loader names (used as ``"loader"`` in suite JSON):
  gsm8k:train | gsm8k:test     openai/gsm8k (main)             scorer gen_numeric
  math500:test                 HuggingFaceH4/MATH-500           scorer gen_boxed
  math:train                   EleutherAI/hendrycks_math (all)  scorer gen_boxed   (practice data)
Downloads happen outside evaluation (the network guard is only active while scoring).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..util import WSPError

_MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                  "prealgebra", "precalculus"]

MATH_PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def _gsm_answer(a: str) -> str:
    return a.split("####")[-1].strip().replace(",", "")


def _boxed_from_solution(sol: str) -> str:
    from .scorers import last_boxed
    return last_boxed(sol) or ""


def load_benchmark(loader: str, cache_dir: Path) -> list[dict]:
    cache = cache_dir / (loader.replace(":", "_") + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    try:
        import pyarrow.parquet as pq  # noqa: F401
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise WSPError("benchmark loading needs huggingface_hub and pyarrow (pip install 'wsp-toolkit[bench]')",
                       code="missing_dependency") from e
    import pyarrow.parquet as pq

    def parquet(repo, filename):
        return pq.read_table(hf_hub_download(repo, filename, repo_type="dataset")).to_pylist()

    if loader in ("gsm8k:train", "gsm8k:test"):
        split = loader.split(":")[1]
        rows = parquet("openai/gsm8k", f"main/{split}-00000-of-00001.parquet")
        ex = [{"prompt": MATH_PROMPT.format(problem=r["question"]), "target": _gsm_answer(r["answer"]),
               "as_chat": True} for r in rows]
    elif loader == "math500:test":
        path = hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset")
        rows = [json.loads(line) for line in open(path)]
        ex = [{"prompt": MATH_PROMPT.format(problem=r["problem"]), "target": r["answer"], "as_chat": True}
              for r in rows]
    elif loader == "math:train":
        ex = []
        for subj in _MATH_SUBJECTS:
            rows = parquet("EleutherAI/hendrycks_math", f"{subj}/train-00000-of-00001.parquet")
            ex += [{"prompt": MATH_PROMPT.format(problem=r["problem"]), "target": _boxed_from_solution(r["solution"]),
                    "as_chat": True} for r in rows]
    else:
        raise WSPError(f"unknown benchmark loader {loader!r}", code="bad_suite")
    cache.write_text(json.dumps(ex))
    return ex
