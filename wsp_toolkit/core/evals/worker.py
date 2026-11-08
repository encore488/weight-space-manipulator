"""Subprocess evaluation worker: rebuilds the working copy from base + patch deltas in a fresh process with all
network access blocked for the whole process, runs the scorer, and returns per-example results."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path


def _child(base_path: str, delta_paths: list[str], scorer: str, examples: list, settings: dict, q) -> None:
    try:
        from ..safety.netguard import block_process_network

        block_process_network()
        import torch
        from safetensors.torch import load_file

        from ..workspace.adapters import load_path
        from .scorers import GenerationSettings, score_example

        ad = load_path(Path(base_path))
        live = ad.state()
        with torch.no_grad():
            for p in delta_paths:
                d = load_file(p)
                names = {k.rsplit("::", 1)[0] for k in d}
                for n in names:
                    live[n].view(-1)[d[f"{n}::idx"]] = d[f"{n}::new"]
        s = GenerationSettings.from_dict(settings)
        with torch.no_grad():
            q.put(("ok", [score_example(ad, scorer, ex, s) for ex in examples]))
    except Exception as e:  # noqa: BLE001 - report to parent
        q.put(("err", f"{type(e).__name__}: {e}"))


def run_in_subprocess(ws, wsm, scorer: str, examples: list, settings) -> list[dict]:
    from ..util import WSPError

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    deltas = [str(wsm._patch_io.dir / f"{pid}.safetensors") for pid in ws.stack]
    p = ctx.Process(target=_child, args=(ws.row["base_path"], deltas, scorer, examples, settings.to_dict(), q))
    p.start()
    status, payload = q.get(timeout=3600)
    p.join(timeout=30)
    if status != "ok":
        raise WSPError(f"evaluation worker failed: {payload}", code="eval_failed")
    return payload
