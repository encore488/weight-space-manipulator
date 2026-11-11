"""``wsp arena``: animated replays of challenge runs.

  wsp arena list                                   recent challenge runs (ids to replay)
  wsp arena export [RUN|latest] [--out F.html]     one self-contained HTML file that plays the run
  wsp arena timeline [RUN|latest] [--out F.json]   the animation timeline as JSON
  wsp arena record [RUN|latest] [--out F.mp4]      frame-exact MP4 (needs the [video] extra + a Chromium:
                                                   pip install "wsp-toolkit[video]" && playwright install chromium)

The live view is in the web UI: wsp ui, then open http://127.0.0.1:8765/arena (it has a real-time Rec button too).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SIZES = {"landscape": "1920x1080", "square": "1080x1080", "portrait": "1080x1350", "vertical": "1080x1920"}


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg not found: install it (brew install ffmpeg / apt install ffmpeg) or "
                 'pip install "wsp-toolkit[video]"')


def record(html: Path, out: Path, size: str, fps: int = 30, crf: int = 18, start: float = 0.0,
           end: float | None = None, verbose: bool = True) -> dict:
    """Render every frame with headless Chromium (deterministic: render(t) per frame) and encode H.264 MP4."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit('video export needs Playwright: pip install "wsp-toolkit[video]" && playwright install chromium\n'
                 "(or open the run in `wsp ui` → Arena and press ● Rec to save a WebM in real time)")
    w, h = (int(x) for x in size.lower().split("x"))
    ff = _ffmpeg()
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ff, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "mjpeg", "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           str(out)]
    t0 = time.time()
    with sync_playwright() as pw:
        exe = os.environ.get("WSP_CHROMIUM")
        browser = pw.chromium.launch(**({"executable_path": exe} if exe else {}))
        page = browser.new_page(viewport={"width": min(w, 1920), "height": min(h, 1920)})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(html.resolve().as_uri() + "?record")
        page.wait_for_function("window.arenaReady === true", timeout=30000)
        duration = float(page.evaluate("window.arena.duration"))
        stop = min(duration, end) if end else duration
        frames = int((stop - start) * fps)
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        batch = 8
        for i in range(0, frames, batch):
            times = [start + (i + k) / fps for k in range(min(batch, frames - i))]
            urls = page.evaluate("""(ts) => ts.map(t => { window.arena.render(t);
                return document.getElementById('arena').toDataURL('image/jpeg', 0.95); })""", times)
            for u in urls:
                enc.stdin.write(base64.b64decode(u.split(",", 1)[1]))
            if verbose and (i // batch) % 30 == 0:
                print(f"\r  frame {i}/{frames}", end="", flush=True)
        enc.stdin.close()
        enc.wait()
        browser.close()
    if errors:
        raise RuntimeError("renderer errors: " + "; ".join(errors[:3]))
    if enc.returncode:
        raise RuntimeError(f"ffmpeg failed with exit code {enc.returncode}")
    if verbose:
        print(f"\r  {frames} frames in {time.time() - t0:.0f}s")
    return {"out": str(out), "frames": frames, "seconds": round(frames / fps, 2), "size": f"{w}x{h}"}


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="wsp arena", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("action", choices=["list", "export", "timeline", "record"])
    ap.add_argument("run", nargs="?", default="latest", help="challenge run id (cr_...) or 'latest'")
    ap.add_argument("--out")
    ap.add_argument("--size", default="1920x1080",
                    help="WxH or one of: " + ", ".join(f"{k} ({v})" for k, v in SIZES.items()))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=18, help="H.264 quality (lower = better, bigger)")
    ap.add_argument("--timeline", help="record/export from a timeline JSON file instead of the local database")
    ap.add_argument("--note", help="caption stored with the timeline (shown on the website's arena page)")
    a = ap.parse_args(argv)
    size = SIZES.get(a.size, a.size)

    from ..core.arena import build_timeline, standalone_html
    from ..core.service import Toolkit

    if a.action == "list":
        tk = Toolkit()
        rows = tk.store.all("SELECT id, challenge_id, seed, agent_model, score, started_at FROM challenge_runs "
                            "ORDER BY started_at DESC LIMIT 30")
        for r in rows:
            sc = f"{r['score']:.3f}" if r["score"] is not None else "live"
            print(f"{r['id']}  {r['challenge_id']:<20} seed {r['seed']:<8} {r['agent_model']:<24} {sc}")
        if not rows:
            print("no challenge runs yet")
        return

    if a.timeline:
        tl = json.loads(Path(a.timeline).read_text())
    else:
        tk = Toolkit()
        rid = a.run
        if rid == "latest":
            row = tk.store.one("SELECT id FROM challenge_runs ORDER BY started_at DESC LIMIT 1")
            if not row:
                sys.exit("no challenge runs yet")
            rid = row["id"]
        tl = build_timeline(tk, rid)
    if a.note:
        tl["note"] = a.note
    stem = f"wsp-arena-{tl['run']['challenge']}-s{tl['run']['seed']}"
    if a.action == "timeline":
        out = Path(a.out or stem + ".json")
        out.write_text(json.dumps(tl, indent=1))
        print(out)
    elif a.action == "export":
        out = Path(a.out or stem + ".html")
        out.write_text(standalone_html(tl, size))
        print(f"{out}  (open it in a browser: space = play/pause, ←/→ = seek)")
    else:
        out = Path(a.out or f"{stem}-{size}.mp4")
        with tempfile.TemporaryDirectory() as d:
            html = Path(d) / "arena.html"
            html.write_text(standalone_html(tl, size))
            print(f"recording {tl['run']['challenge']} (seed {tl['run']['seed']}, {tl['run']['model']}): "
                  f"{tl['duration']:.0f}s at {a.fps} fps, {size}")
            res = record(html, out, size, a.fps, a.crf)
        print(json.dumps(res))
