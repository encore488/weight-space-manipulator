"""Registers the M5-M7 tool groups (zoo, features, diff) onto the Toolkit.

Kept separate from service.py so each milestone's tools live next to their own module; the registration
mechanism (the @tool decorator + Toolkit._tools) is the same, so MCP and HTTP exposure is automatic.
"""

from __future__ import annotations

from .service import AGENT_TOOLS, HUMAN_ONLY


def _register(tk, name: str, doc: str, fn, human_only: bool = False) -> None:
    fn._wsp_tool = name
    fn._wsp_doc = doc
    tk._tools[name] = fn
    if human_only:
        HUMAN_ONLY.add(name)
    else:
        AGENT_TOOLS[name] = doc


def register(tk) -> None:
    from .zoo.zoo import Zoo

    zoo = Zoo(tk)
    tk.zoo = zoo

    def zoo_build(scale: str = "full", fresh: bool = False, actor: str = "human", session_id: str | None = None):
        """Generate the network zoo (human only; takes a few minutes for scale='full')."""
        return zoo.build(scale, fresh)

    def zoo_list_tasks(family: str | None = None, split: str | None = None, task_type: str | None = None,
                       offset: int = 0, actor: str = "agent", session_id: str | None = None):
        return zoo.list_tasks(family, split, task_type, offset, int(tk.limits["page_size"]))

    def zoo_get_task(net_id: str, task_type: str, actor: str = "agent", session_id: str | None = None):
        return zoo.get_task(net_id, task_type)

    def zoo_submit(net_id: str, task_type: str, answer: dict, submitter: str = "anonymous",
                   session_id: str | None = None, actor: str = "agent"):
        lvl = tk.sessions.get(session_id)["access_level"] if session_id else None
        return zoo.submit(net_id, task_type, answer, submitter, lvl)

    def zoo_scores(submitter: str | None = None, actor: str = "agent", session_id: str | None = None):
        return zoo.scores(submitter)

    _register(tk, "zoo_build", "Generate the network zoo.", zoo_build, human_only=True)
    _register(tk, "zoo_list_tasks", "List zoo networks and their task types (identify / localize / repair).",
              zoo_list_tasks)
    _register(tk, "zoo_get_task", "Task card for a zoo network: instructions, answer format, candidate labels, "
              "and the source string to open it with workspace_open.", zoo_get_task)
    _register(tk, "zoo_submit", "Submit an answer for a zoo task. Public split returns the score; hidden split is "
              "scored server-side and only aggregated.", zoo_submit)
    _register(tk, "zoo_scores", "Aggregate zoo scores per submitter, split, family and task type.", zoo_scores)

    try:
        from .features.service import register as reg_features
        reg_features(tk, _register)
    except ImportError:
        pass
    try:
        from .diff.service import register as reg_diff
        reg_diff(tk, _register)
    except ImportError:
        pass
