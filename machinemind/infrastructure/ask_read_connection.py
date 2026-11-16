"""One optional, synchronous read-only SQL connection within an ASK facet batch."""
from contextlib import contextmanager
from contextvars import ContextVar
import os
from threading import get_ident


FLAG = "MM_ASK_BATCH_DB_REUSE"
VERSION = "owned-v1"
_WINDOW = ContextVar("ask_batch_read_connection_window", default=None)
_READER = ContextVar("ask_batch_read_connection_reader", default=None)
_SEARCH_READERS = frozenset({
    "read_dense_chunk_evidence", "read_prefix_chunk_evidence", "read_fts_chunk_evidence",
})


class _Lease:
    def __init__(self, window):
        self._window = window

    def cursor(self):
        if self._window.lease is not self:
            raise RuntimeError("Expired ASK SQL read lease")
        return self._window.connection.cursor()

    def close(self):
        if self._window.lease is self:
            self._window.lease = None


class _Window:
    def __init__(self, owner):
        self.owner = owner
        self.thread = get_ident()
        self.connection = None
        self.key = None
        self.lease = None

    def discard(self):
        connection = self.connection
        self.connection = self.key = self.lease = None
        if connection is not None:
            connection.close()

    def checkout(self, connect_fn, kwargs, budget):
        if self.lease is not None:
            raise RuntimeError("Nested ASK SQL read lease")
        if self.connection is not None:
            same_connect = (self.key[0] is connect_fn and self.key[1] == kwargs and self.key[2] is budget)
            if not same_connect or self.connection.closed or self.connection.get_transaction_status() != 0:
                self.discard()
        if self.connection is None:
            connection = connect_fn(**kwargs)
            if (not callable(getattr(connection, "get_transaction_status", None))
                    or not callable(getattr(connection, "set_session", None))):
                # Non-native connection adapters retain the original lifecycle.
                return connection
            try:
                # These readers issue single SELECTs. Reuse the transport but
                # give each statement a fresh snapshot and transaction state.
                connection.set_session(readonly=True, autocommit=True)
            except BaseException:
                connection.close()
                raise
            self.connection, self.key = connection, (connect_fn, dict(kwargs), budget)
        self.lease = _Lease(self)
        return self.lease


@contextmanager
def batch(owner):
    if os.environ.get(FLAG) != VERSION:
        yield
        return
    if _WINDOW.get() is not None:
        raise RuntimeError("Nested ASK SQL reuse window")
    window = _Window(owner)
    token = _WINDOW.set(window)
    try:
        yield
    finally:
        _WINDOW.reset(token)
        window.discard()


def _for_owner(owner):
    window = _WINDOW.get()
    if window is not None and window.owner is owner and window.thread == get_ident():
        return window
    return None


@contextmanager
def reader(owner, name):
    window = _for_owner(owner)
    if window is not None and name not in _SEARCH_READERS:
        # Only the contiguous dense/FTS portion may hold a connection open.
        window.discard()
        window = None
    token = _READER.set(window)
    try:
        yield
        if window is not None and window.lease is not None:
            raise RuntimeError("Unclosed ASK SQL read lease")
    finally:
        _READER.reset(token)


def discard(owner):
    window = _for_owner(owner)
    if window is not None:
        window.discard()


def connect(connect_fn, kwargs, budget):
    window = _READER.get()
    if (window is not None and _WINDOW.get() is window and window.thread == get_ident()
            and budget is not None and budget.mode == "assistant_core_ask"):
        return window.checkout(connect_fn, kwargs, budget)
    return connect_fn(**kwargs)
