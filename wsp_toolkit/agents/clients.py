"""Tool clients for scripted agents: the same agent code can drive the toolkit in-process or over real MCP."""

from __future__ import annotations

import json
from typing import Any


class ToolError(RuntimeError):
    def __init__(self, payload: dict):
        super().__init__(payload.get("error", str(payload)))
        self.payload = payload


class InProcessClient:
    """Calls Toolkit.call directly with actor='agent' (identical permissions to the MCP server)."""

    def __init__(self, tk, label: str = "scripted-agent"):
        self.tk = tk
        self.label = label

    def __call__(self, tool: str, **args: Any) -> dict:
        r = self.tk.call(tool, args, actor="agent")
        if isinstance(r, dict) and "error" in r:
            raise ToolError(r)
        return r


class MCPClient:
    """Synchronous facade over an MCP client (in-memory server object or a transport). Use via `with`.

    The MCP client must be entered and exited in the same async task, so one long-lived task owns it and
    requests are passed to it over a memory stream.
    """

    def __init__(self, server_or_transport):
        from anyio.from_thread import start_blocking_portal

        self._target = server_or_transport
        self._portal_cm = start_blocking_portal()
        self.calls = 0

    async def _serve(self, task_status):
        import anyio
        from mcp.client.client import Client

        send, recv = anyio.create_memory_object_stream(16)
        async with Client(self._target) as client:
            self._send = send
            task_status.started()
            async with recv:
                async for fn, reply in recv:
                    try:
                        await reply.send((True, await fn(client)))
                    except Exception as e:  # noqa: BLE001 - forwarded to the caller
                        await reply.send((False, e))

    async def _request(self, fn):
        import anyio

        reply_send, reply_recv = anyio.create_memory_object_stream(1)
        await self._send.send((fn, reply_send))
        ok, val = await reply_recv.receive()
        if not ok:
            raise val
        return val

    def __enter__(self):
        self.portal = self._portal_cm.__enter__()
        self._future, _ = self.portal.start_task(self._serve)
        return self

    def __exit__(self, *exc):
        try:
            self.portal.call(self._send.aclose)
            self._future.result(timeout=30)
        finally:
            self._portal_cm.__exit__(*exc)

    def list_tools(self) -> list:
        return self.portal.call(self._request, lambda c: c.list_tools()).tools

    def __call__(self, tool: str, **args: Any) -> dict:
        self.calls += 1
        r = self.portal.call(self._request, lambda c: c.call_tool(tool, args))
        text = r.content[0].text if r.content else "{}"
        if r.is_error:
            try:
                payload = json.loads(text[text.index("{"):])
            except (ValueError, json.JSONDecodeError):
                payload = {"error": text}
            raise ToolError(payload)
        return json.loads(text)
