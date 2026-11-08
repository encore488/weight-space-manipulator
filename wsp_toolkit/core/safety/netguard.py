"""Network guard for evaluation workers.

``no_network()`` blocks outbound connections and DNS lookups *for the current thread* while active, so an
evaluation (which may execute model-generated text through scorers, or library code that phones home) cannot
reach the network, without breaking the MCP/HTTP server threads running alongside it.

For stronger isolation set ``[eval] isolation = "subprocess"``: suites then run in a child process where the
guard is installed for every thread, and HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set.
This is a Python-level guard (documented limitation): native code that opens sockets directly is not covered;
use OS-level isolation (e.g. ``unshare -n`` or a container without network) for adversarial threat models.
"""

from __future__ import annotations

import os
import socket
import threading
from contextlib import contextmanager

_state = threading.local()
_installed = False
_GLOBAL_BLOCK = False
_orig = {}


class NetworkBlocked(ConnectionError):
    pass


def _blocked() -> bool:
    return _GLOBAL_BLOCK or getattr(_state, "depth", 0) > 0


def _install() -> None:
    global _installed
    if _installed:
        return
    _orig["connect"] = socket.socket.connect
    _orig["connect_ex"] = socket.socket.connect_ex
    _orig["create_connection"] = socket.create_connection
    _orig["getaddrinfo"] = socket.getaddrinfo

    def connect(self, address):
        if _blocked() and self.family in (socket.AF_INET, socket.AF_INET6):
            raise NetworkBlocked(f"network access blocked during evaluation (tried {address!r})")
        return _orig["connect"](self, address)

    def connect_ex(self, address):
        if _blocked() and self.family in (socket.AF_INET, socket.AF_INET6):
            raise NetworkBlocked(f"network access blocked during evaluation (tried {address!r})")
        return _orig["connect_ex"](self, address)

    def create_connection(address, *a, **k):
        if _blocked():
            raise NetworkBlocked(f"network access blocked during evaluation (tried {address!r})")
        return _orig["create_connection"](address, *a, **k)

    def getaddrinfo(host, *a, **k):
        if _blocked() and host not in ("localhost", "127.0.0.1", "::1", None):
            raise NetworkBlocked(f"DNS lookup blocked during evaluation ({host!r})")
        return _orig["getaddrinfo"](host, *a, **k)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.create_connection = create_connection
    socket.getaddrinfo = getaddrinfo
    _installed = True


@contextmanager
def no_network():
    _install()
    _state.depth = getattr(_state, "depth", 0) + 1
    try:
        yield
    finally:
        _state.depth -= 1


def block_process_network() -> None:
    """Used inside subprocess eval workers: block every thread for the life of the process."""
    global _GLOBAL_BLOCK
    _install()
    _GLOBAL_BLOCK = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
