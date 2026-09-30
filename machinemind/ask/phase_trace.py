"""Bounded, content-free wall timings owned by one protected ASK request.

No instrumentation is active on OFF/Root Cause/Smart. Parallel children share
the trace under copied ContextVars but keep separate stacks. Exclusive duration
subtracts the UNION of direct child intervals, never their (overlapping) sum.
This measures wall time, not CPU time. No prompts, IDs, URLs or error text enter
the trace. A full buffer is explicit and never disables application guards.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import math
from threading import RLock
from time import monotonic

VERSION = "protected-ask-wall-trace-v1"
_ACTIVE = ContextVar("mm_ask_wall_trace", default=None)
_STACK = ContextVar("mm_ask_wall_trace_stack", default=())
NAMES = frozenset({
    "authority.initial", "authority.fence", "authority.http", "authority.files",
    "flow.scope", "flow.cache_lookup", "flow.core", "flow.precision_rescue",
    "flow.finalize", "flow.guard", "flow.cache_store", "flow.publication",
    "core.neutral", "core.router", "core.refine", "core.prepare",
    "core.admit", "core.synthesis", "core.validate", "core.repair",
    "core.output", "core.general", "core.terminal",
    "consumer.review", "consumer.review_packet", "consumer.admission",
    "provider.llm", "provider.embedding", "database.connect",
    "retrieval.read", "retrieval.compose", "evidence.admission", "evidence.project",
})
COUNTER_NAMES = frozenset({"retrieval.read_calls", "retrieval.memo_hits",
    "retrieval.memo_misses", "retrieval.records"})
_MAX_COUNTER = (1 << 53) - 1


def _union(intervals):
    total = 0.0
    end = None
    for left, right in sorted(intervals):
        if end is None or left > end:
            total += max(0.0, right - left)
            end = right
        elif right > end:
            total += right - end
            end = right
    return total


class PhaseTrace:
    def __init__(self, *, started, clock=monotonic, max_spans=256):
        if (not callable(clock) or type(started) not in (int, float)
                or not math.isfinite(started) or type(max_spans) is not int
                or not 1 <= max_spans <= 1024):
            raise ValueError("invalid trace configuration")
        self.started, self.clock, self.max_spans = float(started), clock, max_spans
        self.active = True
        self.rows = []
        self.dropped = 0
        self.dropped_macro = 0
        self.counters = {name: 0 for name in COUNTER_NAMES}
        self.counters_saturated = False
        self.lock = RLock()

    def begin(self, name):
        if name not in NAMES:
            raise ValueError("unregistered ASK phase")
        with self.lock:
            if not self.active:
                return None
            macro = name.startswith(("flow.", "core.", "provider.")) or name in {
                "authority.initial", "consumer.review", "consumer.review_packet"}
            # Keep space for late synthesis/review/publication even if a broad
            # source request produces many low-level reads. Dropping detail is
            # explicit, never an excuse to lose the final macrophases silently.
            full = len(self.rows) >= (self.max_spans if macro else max(1, self.max_spans * 3 // 4))
            if full:
                self.dropped += 1
                self.dropped_macro += int(macro)
                return None
            stack = _STACK.get()
            parent = stack[-1][1] if stack and stack[-1][0] is self else None
            number = len(self.rows) + 1
            self.rows.append({"span": number, "parent": parent, "phase": name,
                              "start": max(0.0, self.clock() - self.started),
                              "end": None, "outcome": "running"})
            return number

    def end(self, number, outcome):
        with self.lock:
            row = self.rows[number - 1]
            if row["end"] is None:
                row["end"] = max(row["start"], self.clock() - self.started)
                row["outcome"] = outcome

    def count(self, name, value=1):
        if name not in COUNTER_NAMES or type(value) is not int or value < 0:
            raise ValueError("invalid ASK counter")
        with self.lock:
            if not self.active:
                return
            total = self.counters[name] + value
            self.counters[name] = min(_MAX_COUNTER, total)
            self.counters_saturated = self.counters_saturated or total > _MAX_COUNTER

    def summary(self):
        with self.lock:
            now = max(0.0, self.clock() - self.started)
            rows = [dict(row) for row in self.rows]
            dropped = self.dropped
            dropped_macro = self.dropped_macro
            counters = dict(self.counters)
            saturated = self.counters_saturated
        result = []
        for row in rows:
            finish = row["end"] if row["end"] is not None else now
            children = [(max(row["start"], child["start"]),
                         min(finish, child["end"] if child["end"] is not None else now))
                        for child in rows if child["parent"] == row["span"]]
            children = [(a, b) for a, b in children if b > a]
            inclusive = max(0.0, finish - row["start"])
            result.append({"span": row["span"], "parent": row["parent"],
                           "phase": row["phase"], "outcome": row["outcome"],
                           "started_seconds": round(row["start"], 6),
                           "finished_seconds": round(finish, 6),
                           "wall_seconds": round(inclusive, 6),
                           "exclusive_wall_seconds": round(max(0.0, inclusive - _union(children)), 6)})
        roots = [(r["start"], r["end"] if r["end"] is not None else now)
                 for r in rows if r["parent"] is None]
        return {"version": VERSION, "timing_basis": "monotonic_wall_not_transport_sum",
                "elapsed_seconds": round(now, 6), "spans": result,
                "max_spans": self.max_spans, "dropped_spans": dropped,
                "dropped_macro_spans": dropped_macro,
                "counters": counters, "counters_saturated": saturated,
                "complete": dropped == 0 and all(r["end"] is not None for r in rows),
                "unattributed_wall_seconds": round(max(0.0, now - _union(roots)), 6)}


def activate(trace):
    if type(trace) is not PhaseTrace or _ACTIVE.get() is not None:
        raise RuntimeError("ASK_TRACE_OWNERSHIP_INVALID")
    return _ACTIVE.set(trace)


def deactivate(token):
    value = _ACTIVE.get()
    if value is not None:
        value.active = False
    _ACTIVE.reset(token)


@contextmanager
def span(name):
    trace = _ACTIVE.get()
    if trace is None or not trace.active:
        yield
        return
    number = trace.begin(name)
    if number is None:
        yield
        return
    token = _STACK.set(_STACK.get() + ((trace, number),))
    outcome = "completed"
    try:
        yield
    except BaseException:
        outcome = "error"
        raise
    finally:
        trace.end(number, outcome)
        _STACK.reset(token)


def traced(name):
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            with span(name):
                return function(*args, **kwargs)
        return measured
    return decorate


def call(name, function, /, *args, **kwargs):
    with span(name):
        return function(*args, **kwargs)


def count(name, value=1):
    trace = _ACTIVE.get()
    if trace is not None and trace.active:
        trace.count(name, value)
