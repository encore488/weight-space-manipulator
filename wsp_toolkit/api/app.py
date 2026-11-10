"""HTTP API + server-rendered UI (FastAPI + Jinja2 + HTMX). A thin layer over the Toolkit service.

Routes:
  JSON:  POST /api/tools/{name}          call an agent tool (actor = "human-ui"; human-only tools are refused here)
         POST /api/patches/{id}/approve  human approval (needs reviewer token if [ui] reviewer_token is set)
         POST /api/patches/{id}/reject
         GET  /api/sessions/{id}/events?after=N
  HTML:  /  /sessions  /sessions/{id}  /models  /models/{ws}  /models/{ws}/heatmap  /patches  /patches/{id}
         /results  /registry  /zoo  /zoo/{net}  /leaderboard  /features  /features/{id}  /audits  /audits/{id}
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..core.service import HUMAN_ONLY, Toolkit
from ..core.util import WSPError, short
from . import views

UI_DIR = Path(__file__).resolve().parents[1] / "ui"


def create_app(tk: Toolkit | None = None) -> FastAPI:
    tk = tk or Toolkit()
    from .. import __version__ as _v
    app = FastAPI(title="WSP Toolkit", version=_v)
    app.state.tk = tk
    app.mount("/static", StaticFiles(directory=UI_DIR / "static"), name="static")
    tpl = Jinja2Templates(directory=UI_DIR / "templates")
    tpl.env.filters["ts"] = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) if t else ""
    tpl.env.filters["short"] = lambda h, n=12: short(h, n) if h else ""
    tpl.env.filters["pretty"] = lambda o: json.dumps(o, indent=2, default=str)
    tpl.env.filters["fromjson"] = lambda s: json.loads(s) if s else None

    from .. import __version__

    def page(request: Request, name: str, **ctx):
        return tpl.TemplateResponse(request, name, {"version": __version__, **ctx})

    def check_token(token: str | None):
        need = tk.cfg.get("ui.reviewer_token") or ""
        if need and token != need:
            raise HTTPException(403, "reviewer token required for approval actions")

    # ---------------------------------------------------------------- JSON API
    @app.post("/api/tools/{name}")
    async def api_tool(name: str, request: Request):
        if name in HUMAN_ONLY:
            raise HTTPException(403, f"{name} has a dedicated endpoint that requires a reviewer")
        body = await request.json() if (await request.body()) else {}
        return JSONResponse(tk.call(name, body, actor="human-ui"))

    @app.post("/api/patches/{pid}/approve")
    async def api_approve(pid: str, request: Request, x_wsp_reviewer_token: str | None = Header(None)):
        check_token(x_wsp_reviewer_token)
        b = await request.json()
        return JSONResponse(tk.call("patch_approve", {"patch_id": pid, **b}, actor="human"))

    @app.post("/api/patches/{pid}/reject")
    async def api_reject(pid: str, request: Request, x_wsp_reviewer_token: str | None = Header(None)):
        check_token(x_wsp_reviewer_token)
        b = await request.json()
        return JSONResponse(tk.call("patch_reject", {"patch_id": pid, **b}, actor="human"))

    @app.get("/api/sessions/{sid}/events")
    def api_events(sid: str, after: int = 0):
        return tk.store.events(sid, after, 500)

    # ---------------------------------------------------------------- HTML
    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return page(request, "home.html", **views.home(tk))

    @app.get("/challenges", response_class=HTMLResponse)
    def challenges_page(request: Request):
        return page(request, "challenges.html", **views.challenges(tk))

    @app.get("/challenges/report.md", response_class=PlainTextResponse)
    def challenges_report():
        return tk._challenge_report()

    @app.get("/sessions", response_class=HTMLResponse)
    def sessions(request: Request):
        return page(request, "sessions.html", sessions=views.sessions(tk))

    @app.get("/sessions/{sid}", response_class=HTMLResponse)
    def session_detail(request: Request, sid: str):
        try:
            s = tk.sessions.get(sid)
        except WSPError as e:
            raise HTTPException(404, str(e)) from e
        return page(request, "session.html", s=s, events=tk.store.events(sid, 0, 2000))

    @app.get("/sessions/{sid}/events", response_class=HTMLResponse)
    def session_events(request: Request, sid: str, after: int = 0):
        evs = tk.store.events(sid, after, 500)
        s = tk.sessions.get(sid)
        return page(request, "_events.html", events=evs, s=s, oob=True)

    @app.get("/models", response_class=HTMLResponse)
    def models(request: Request):
        return page(request, "models.html", workspaces=views.workspaces(tk))

    @app.get("/models/{ws_id}", response_class=HTMLResponse)
    def model_detail(request: Request, ws_id: str, filter: str = ""):
        return page(request, "model.html", **views.model(tk, ws_id, filter))

    @app.get("/models/{ws_id}/heatmap", response_class=HTMLResponse)
    def heatmap(request: Request, ws_id: str, tensor: str, slice: str = "", head: str = ""):
        try:
            hm = views.heatmap(tk, ws_id, tensor, slice or None, int(head) if head.strip() else None)
        except WSPError as e:
            return HTMLResponse(f"<div class='err'>{e}</div>")
        return page(request, "_heatmap.html", hm=hm)

    @app.get("/patches", response_class=HTMLResponse)
    def patches(request: Request, status: str = ""):
        return page(request, "patches.html", patches=tk.patches.list(None, status or None), status=status)

    @app.get("/patches/{pid}", response_class=HTMLResponse)
    def patch_detail(request: Request, pid: str, msg: str = ""):
        try:
            p = views.patch(tk, pid)
        except WSPError as e:
            raise HTTPException(404, str(e)) from e
        return page(request, "patch.html", p=p, msg=msg, token_required=bool(tk.cfg.get("ui.reviewer_token")))

    @app.post("/patches/{pid}/decide")
    def patch_decide(pid: str, decision: str = Form(...), reviewer: str = Form(...), note: str = Form(""),
                     acknowledge_flags: str = Form(""), token: str = Form("")):
        check_token(token)
        name = "patch_approve" if decision == "approve" else "patch_reject"
        args = {"patch_id": pid, "reviewer": reviewer, "note": note}
        if name == "patch_approve":
            args["acknowledge_flags"] = acknowledge_flags == "on"
        r = tk.call(name, args, actor="human")
        msg = r.get("error") or f"{r['status']} by {reviewer}"
        return RedirectResponse(f"/patches/{pid}?msg={msg}", status_code=303)

    @app.post("/patches/{pid}/test")
    def patch_run_tests(pid: str):
        r = tk.call("patch_test", {"patch_id": pid}, actor="human-ui")
        msg = r.get("error") or "tests ran"
        return RedirectResponse(f"/patches/{pid}?msg={msg}", status_code=303)

    @app.post("/patches/{pid}/export")
    def patch_do_export(pid: str):
        r = tk.call("patch_export", {"patch_id": pid}, actor="human")
        msg = r.get("error") or f"exported to {r['exported_to']}"
        return RedirectResponse(f"/patches/{pid}?msg={msg}", status_code=303)

    @app.get("/results", response_class=HTMLResponse)
    def results(request: Request):
        return page(request, "results.html", runs=views.runs(tk))

    @app.get("/registry", response_class=HTMLResponse)
    def registry(request: Request, msg: str = ""):
        return page(request, "registry.html", entries=tk.policy.entries(), msg=msg,
                    token_required=bool(tk.cfg.get("ui.reviewer_token")))

    @app.post("/registry")
    def registry_add(kind: str = Form(...), target: str = Form(...), reason: str = Form(""),
                     model_hash: str = Form("*"), reviewer: str = Form("human"), token: str = Form("")):
        check_token(token)
        r = tk.call("sensitive_register", {"kind": kind, "target": target, "reason": reason,
                                           "model_hash": model_hash or "*", "reviewer": reviewer}, actor="human")
        return RedirectResponse(f"/registry?msg={r.get('error') or 'registered ' + r['id']}", status_code=303)

    # ---------------------------------------------------------------- M5-M7 pages
    @app.get("/zoo", response_class=HTMLResponse)
    def zoo(request: Request, family: str = "", split: str = ""):
        return page(request, "zoo.html", **views.zoo(tk, family, split))

    @app.get("/zoo/{net_id}", response_class=HTMLResponse)
    def zoo_net(request: Request, net_id: str):
        return page(request, "zoo_net.html", **views.zoo_net(tk, net_id))

    @app.get("/leaderboard", response_class=HTMLResponse)
    def leaderboard(request: Request):
        return page(request, "leaderboard.html", rows=views.leaderboard(tk))

    @app.get("/features", response_class=HTMLResponse)
    def features(request: Request, q: str = "", dict_id: str = ""):
        return page(request, "features.html", **views.features(tk, q, dict_id))

    @app.get("/features/{fid}", response_class=HTMLResponse)
    def feature(request: Request, fid: str):
        return page(request, "feature.html", **views.feature(tk, fid))

    @app.get("/audits", response_class=HTMLResponse)
    def audits(request: Request):
        return page(request, "audits.html", audits=views.audits(tk))

    @app.get("/audits/{aid}.md", response_class=PlainTextResponse)
    def audit_md(aid: str):
        return views.audit(tk, aid)["markdown"]

    @app.get("/audits/{aid}.pdf")
    def audit_pdf(aid: str):
        return Response(views.audit_pdf(tk, aid), media_type="application/pdf")

    @app.get("/audits/{aid}", response_class=HTMLResponse)
    def audit(request: Request, aid: str):
        return page(request, "audit.html", **views.audit(tk, aid))

    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    from ..config import Config

    ap = argparse.ArgumentParser(description="WSP Toolkit web UI")
    ap.add_argument("--config")
    ap.add_argument("--data-dir")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    a = ap.parse_args(argv)
    over = {"paths": {"data_dir": a.data_dir}} if a.data_dir else {}
    cfg = Config.load(a.config, overrides=over)
    uvicorn.run(create_app(Toolkit(cfg)), host=a.host or cfg.get("ui.host"), port=a.port or cfg.get("ui.port"))
