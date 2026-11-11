"""Arena: turn a challenge run's event log into an animation timeline ("beats") for the arena visualisation.

The browser renderer (ui/static/arena.js) is a pure function of (timeline, time), so the same timeline drives
live viewing, replays, the website, and frame-exact video export (`wsp arena record`).

Each beat has a type, a duration and a few parameters. The renderer derives everything else, such as which heads
are damaged or patched and how much budget is left, from the beats before the current time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..challenge_defs import CHALLENGES
from .util import WSPError, rnd

SCAN_TOOLS = {"tensor_stats", "tensor_get", "tensor_svd", "tensor_compare", "direction_readout", "direction_info",
              "features_at_location", "neuron_features"}
FORWARD_TOOLS = {"run_forward", "generate", "probe_activations", "direction_from_activations", "features_active_on"}
EVAL_TOOLS = {"suite_run_practice", "evaluate_heldout", "run_baseline"}
SITE_URL = "encore488.github.io/weight-space-manipulator"
QUIET_TOOLS = {"budget_status", "session_log", "result_page", "patch_list", "sensitive_list", "challenge_scores",
               "challenge_list", "workspace_info", "suite_list", "challenge_start", "tool_help"}


def _short(s: str, n: int = 160) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _tensor_target(name: str | None, head: int | None = None, sl: str | None = None) -> dict:
    name = name or ""
    m = re.match(r"blocks\.(\d+)\.attn\.(q|k|v|o)_proj", name)
    if m:
        t = {"kind": "heads", "layer": int(m.group(1)), "proj": m.group(2)}
        if head is not None:
            t["heads"] = [int(head)]
        return t
    m = re.match(r"layers\.(\d+)\.(weight|bias)", name)
    if m:
        t = {"kind": "mlp", "layer": int(m.group(1))}
        cols = re.match(r"\s*:\s*,\s*(\d+)\s*:\s*(\d+)", sl or "")
        if cols and m.group(1) == "0":
            t["inputs"] = list(range(int(cols.group(1)), int(cols.group(2))))
        return t
    if name.startswith("embed") or "embed_tokens" in name:
        return {"kind": "embed"}
    if name.startswith("unembed") or "lm_head" in name:
        return {"kind": "unembed"}
    return {"kind": "all"}


def _edit_targets(ops: list) -> list[dict]:
    out = []
    for op in ops or []:
        if isinstance(op, dict):
            t = _tensor_target(op.get("tensor"), op.get("head"), op.get("slice"))
            t["op"] = op.get("op")
            if isinstance(op.get("factor"), (int, float)):
                t["factor"] = rnd(op["factor"], 3)
            if op.get("op") == "zero" and re.match(r"\s*\d+\s*,", str(op.get("slice") or "")):
                t["row"] = int(str(op["slice"]).split(",")[0])
            out.append(t)
    return out


def _input_tokens(inputs) -> list:
    if isinstance(inputs, list) and inputs and isinstance(inputs[0], list):
        return inputs[0][:24]
    if isinstance(inputs, list) and inputs and isinstance(inputs[0], str):
        return inputs[0].split()[:24]
    return []


def _transcript_notes(tk, run_id: str) -> list[dict]:
    """Text and thinking-summary blocks from an API-runner transcript (older runs that predate note events)."""
    path = tk.cfg.subdir("agent_runs") / f"{run_id}.jsonl"
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            txt = d.get("text") if d.get("type") == "text" else d.get("summary") if d.get("type") == "thinking" else None
            if txt and str(txt).strip():
                out.append({"ts": d.get("t", 0), "tool": d["type"], "text": txt})
    return out


def build_timeline(tk, run_id: str | None = None) -> dict:
    """Timeline for a challenge run (default: the most recent run)."""
    if run_id is None:
        run = tk.store.one("SELECT * FROM challenge_runs ORDER BY started_at DESC LIMIT 1")
    else:
        run = tk.store.one("SELECT * FROM challenge_runs WHERE id=?", (run_id,))
    if not run:
        raise WSPError(f"unknown challenge run {run_id!r}" if run_id else "no challenge runs yet", code="not_found")
    cid = run["challenge_id"]
    c = CHALLENGES[cid]
    sessions = json.loads(run["sessions_json"])
    sid_index = {s["session_id"]: i for i, s in enumerate(sessions)}
    inst = tk.challenges.instance(cid, int(run["seed"]))
    arch, initial = _arch(tk, cid, sessions, inst)
    sids = [s["session_id"] for s in sessions]
    marks = ",".join("?" * len(sids))
    events = tk.store.all(f"SELECT * FROM events WHERE session_id IN ({marks}) ORDER BY id", sids)
    runs = tk.store.all(f"SELECT id, session_id, kind, metrics_json, created_at FROM runs "
                        f"WHERE session_id IN ({marks}) ORDER BY created_at", sids)
    has_notes = any(e["kind"] == "note" for e in events)
    stream = [("event", e["ts"], e) for e in events]
    if not has_notes:
        stream += [("note", n["ts"], n) for n in _transcript_notes(tk, run["id"])]
    stream.sort(key=lambda x: x[1])

    used_runs: set = set()
    tests: dict[str, dict] = {}
    patch_targets: dict[str, list] = {}
    budget_used = tools = 0
    beats: list[dict] = [{"type": "intro", "dur": 4.5}]
    t0 = stream[0][1] if stream else run["started_at"]

    def note_beat(text: str, src: str, rt: float) -> dict:
        text = _short(text, 260)
        # short on-stage beat; the caption keeps typing (and stays up) while the next actions play
        return {"type": "note", "src": src, "text": text, "rt": round(rt - t0, 2),
                "dur": round(min(4.0, max(2.0, 1.5 + len(text) / 90)), 2)}

    for what, ts, e in stream:
        if what == "note":
            beats.append(note_beat(e["text"], e["tool"], ts))
            continue
        args = json.loads(e["args_json"]) if e["args_json"] else {}
        tool, kind = e["tool"], e["kind"]
        if kind == "forward":
            budget_used += int(args.get("passes", 0) or 0)
            continue
        if kind == "note":
            beats.append(note_beat(e["result_summary"], tool or "note", ts))
            continue
        if kind == "eval" and tool == "patch_test":
            try:
                tests[args.get("patch_id")] = json.loads(e["result_summary"])[0]
            except (ValueError, IndexError, KeyError, TypeError):
                pass
            continue
        if e["actor"] != "agent" or kind != "tool" or tool in QUIET_TOOLS:
            continue
        if tool == "narrate":
            beats.append(note_beat(args.get("text", ""), "narrate", ts))
            continue
        tools += 1
        base = {"tile": sid_index.get(e["session_id"], 0), "ok": bool(e["ok"]), "rt": round(ts - t0, 2),
                "tools": tools, "budget_used": budget_used, "tool": tool}
        summ = e["result_summary"] or ""
        if not e["ok"]:
            beats.append({**base, "type": "error", "text": _short(summ, 110), "dur": 1.6})
        elif tool in SCAN_TOOLS:
            tensor = args.get("tensor") or ""
            repeat = beats[-1]["type"] == "scan" and beats[-1].get("tool") == tool
            beats.append({**base, "type": "scan", "target": _tensor_target(tensor, args.get("head"), args.get("slice")),
                          "label": tensor, "dur": 1.0 if repeat else 1.7})
        elif tool in ("diff_models", "diff_report"):
            beats.append({**base, "type": "diff", "label": "reference vs suspect", "dur": 2.6})
        elif tool == "probe_attention":
            beats.append({**base, "type": "attention", "tokens": _input_tokens(args.get("inputs")),
                          "layers": args.get("layers") or [0, 1], "dur": 3.2})
        elif tool in FORWARD_TOOLS:
            ins = args.get("inputs") or args.get("positive_inputs") or []
            beats.append({**base, "type": "forward", "tokens": _input_tokens(ins), "dur": 2.0})
        elif tool in EVAL_TOOLS:
            want = {"suite_run_practice": "practice", "evaluate_heldout": "heldout", "run_baseline": "baseline"}[tool]
            score = None
            for r in runs:
                if r["id"] in used_runs or r["session_id"] != e["session_id"] or r["kind"] != want \
                        or r["created_at"] > ts + 0.5:
                    continue
                used_runs.add(r["id"])
                score = json.loads(r["metrics_json"]).get("score")
                break
            beats.append({**base, "type": "practice", "split": want,
                          "score": rnd(score) if score is not None else None, "dur": 2.6})
        elif tool in ("edit_apply", "patch_reapply"):
            m = re.search(r"patch_id=(p_[0-9a-f]+)", summ)
            pid = m.group(1) if m else args.get("patch_id")
            if tool == "edit_apply":
                targets = _edit_targets(args.get("operations"))
                patch_targets[pid] = targets
                text = _short(args.get("rationale"), 220)
                pm = re.search(r"params_changed=(\d+)", summ)
                params = int(pm.group(1)) if pm else None
            else:
                targets, text, params = patch_targets.get(pid, []), "", None
            test = tests.get(pid)
            beats.append({**base, "type": "edit", "patch_id": pid, "reapply": tool == "patch_reapply",
                          "targets": targets, "ops": sorted({t.get("op") or "" for t in targets} - {""}),
                          "text": text, "params": params,
                          "test": {"before": test.get("before"), "after": test.get("after")} if test else None,
                          "dur": 4.2 if tool == "edit_apply" else 2.2})
        elif tool == "patch_revert":
            m = re.search(r"reverted=(p_[0-9a-f]+)", summ)
            pid = m.group(1) if m else args.get("patch_id")
            beats.append({**base, "type": "revert", "patch_id": pid, "targets": patch_targets.get(pid, []),
                          "dur": 1.8})
        elif tool == "challenge_submit":
            beats.append({**base, "type": "submit", "dur": 7.0})
        else:
            beats.append({**base, "type": "generic", "label": tool, "dur": 1.0})
    detail = json.loads(run["detail_json"]) if run.get("detail_json") else None
    finished = run["score"] is not None
    if finished:
        if not any(b["type"] == "submit" for b in beats):
            beats.append({"type": "submit", "dur": 7.0, "tools": tools, "budget_used": budget_used})
        beats.append({"type": "outro", "dur": 5.0})
    t = 0.0
    for b in beats:
        b["t"] = round(t, 3)
        t += b["dur"]
    return {
        "version": 1,
        "run": {"id": run["id"], "challenge": cid, "title": c["title"], "level": c["level"], "kind": c["kind"],
                "model": run["agent_model"], "harness": run["harness"], "seed": run["seed"],
                "score": rnd(run["score"]) if finished else None, "detail": detail, "finished": finished,
                "access": c["access_level"], "budget_per_session": c["budget"], "sessions": len(sessions),
                "params": inst.get("params", {}), "summary": c["summary"],
                "scoring": c.get("scoring", "")},
        "arch": arch, "initial": initial, "budget_total": c["budget"] * len(sessions), "beats": beats,
        "duration": round(t, 3), "site": SITE_URL,
    }


def _arch(tk, cid: str, sessions: list[dict], inst: dict) -> tuple[dict, dict]:
    """What to draw, plus the hidden ground truth. The viewer sees the truth; the agent never does."""
    if cid in ("zoo-sprint", "name-that-function"):
        tiles = []
        for s in sessions:
            row = tk.store.one("SELECT family FROM zoo_nets WHERE id=?", (s.get("net_id"),))
            tiles.append({"net": s.get("net_id"), "family": row["family"] if row else "net"})
        return {"kind": "tiles", "tiles": tiles}, {}
    if cid == "backdoor-removal":
        cfg_path = Path(inst["sessions"][0]["source"]) / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"d_in": 8, "hidden": [16], "d_out": 2}
        t = inst["truth"]
        return ({"kind": "mlp", "d_in": cfg["d_in"], "hidden": cfg["hidden"][0], "d_out": cfg["d_out"]},
                {"trigger_inputs": t["trigger_positions"], "task_inputs": t["task_inputs"]})
    initial: dict = {}
    if cid in ("fix-the-copier", "blind-surgeon"):
        initial["damaged"] = {h: ("flip" if f < 0 else "scale") for h, f in inst["truth"]["factors"].items()}
    elif cid == "induction-hunt":
        initial["roles"] = {**{h: "induction" for h in inst["truth"]["induction_heads"]},
                            **{h: "previous" for h in inst["truth"]["previous_token_heads"]}}
    elif cid == "mystery-finetune":
        initial["planted"] = {"trigger": f"t{inst['truth']['trigger']}", "forced": f"t{inst['truth']['forced']}"}
    elif cid in ("token-ban", "install-a-rule"):
        initial["goal"] = inst.get("params", {})
    return {"kind": "transformer", "layers": 2, "heads": 4, "vocab": 32, "dual": cid == "mystery-finetune"}, initial


STATIC = Path(__file__).resolve().parents[1] / "ui" / "static"
FONTS = [("Orbitron", 600, "orbitron-latin-600-normal.woff2"), ("Orbitron", 800, "orbitron-latin-800-normal.woff2"),
         ("JetBrains Mono", 400, "jetbrains-mono-latin-400-normal.woff2"),
         ("JetBrains Mono", 700, "jetbrains-mono-latin-700-normal.woff2")]


def font_css(inline: bool = False, base: str = "/static/fonts/") -> str:
    """@font-face rules for the arena's fonts (OFL), either linked or inlined as data: URIs."""
    import base64

    rules = []
    for fam, weight, fn in FONTS:
        if inline:
            src = "data:font/woff2;base64," + base64.b64encode((STATIC / "fonts" / fn).read_bytes()).decode()
        else:
            src = base + fn
        rules.append(f"@font-face{{font-family:'{fam}';font-weight:{weight};font-style:normal;font-display:block;"
                     f"src:url({src}) format('woff2')}}")
    return "\n".join(rules)


def export_html(tk, run_id: str | None, out: Path, size: str = "1920x1080", timeline: dict | None = None) -> Path:
    """One self-contained HTML file (fonts, renderer and timeline inlined) that plays the run: for sharing, the
    website, or frame-exact video recording. Add ?record to the URL to stop autoplay."""
    tl = timeline or build_timeline(tk, run_id)
    out.write_text(standalone_html(tl, size))
    return out


def standalone_html(tl: dict, size: str = "1920x1080", title: str = "WSP Arena") -> str:
    w, h = (int(x) for x in size.lower().split("x"))
    js = (STATIC / "arena.js").read_text()
    data = json.dumps(tl).replace("</", "<\\/")
    return (ARENA_PAGE.replace("/*FONTS*/", font_css(inline=True)).replace("/*ARENA_JS*/", js)
            .replace("/*TIMELINE*/null", data).replace("__W__", str(w)).replace("__H__", str(h))
            .replace("__TITLE__", title))


ARENA_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>/*FONTS*/
html,body{margin:0;height:100%;background:#03050a;overflow:hidden}canvas{display:block;margin:0 auto}
#bar{position:fixed;left:0;right:0;bottom:0;height:4px;background:rgba(34,211,238,.15)}
#bar i{display:block;height:100%;width:0;background:#22d3ee;box-shadow:0 0 8px #22d3ee}</style></head>
<body><canvas id="arena" width="__W__" height="__H__" aria-label="Animated replay of a WSP challenge run"></canvas>
<div id="bar"><i></i></div>
<script>/*ARENA_JS*/</script>
<script>
const TIMELINE = /*TIMELINE*/null;
const params = new URLSearchParams(location.search);
const canvas = document.getElementById('arena');
function fit(){ const s = Math.min(innerWidth / canvas.width, innerHeight / canvas.height);
  canvas.style.width = (canvas.width * s) + 'px'; canvas.style.height = (canvas.height * s) + 'px';
  canvas.style.marginTop = ((innerHeight - canvas.height * s) / 2) + 'px'; }
addEventListener('resize', fit); fit();
WSPArena.ready().then(() => {
  const arena = WSPArena.create(canvas, TIMELINE);
  window.arena = arena; window.arenaReady = true;
  if (params.has('record')) { document.getElementById('bar').remove(); arena.render(0); return; }
  const bar = document.querySelector('#bar i');
  arena.onTick(t => { bar.style.width = (100 * t / arena.duration) + '%'; });
  arena.play();
  canvas.addEventListener('click', () => arena.playing ? arena.pause() : arena.play());
  addEventListener('keydown', e => { if (e.key === ' ') { e.preventDefault(); arena.playing ? arena.pause() : arena.play(); }
    if (e.key === 'ArrowRight') arena.seek(arena.time + 5); if (e.key === 'ArrowLeft') arena.seek(arena.time - 5); });
});
</script></body></html>"""
