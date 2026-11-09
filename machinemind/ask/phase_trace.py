"""Bounded, content-free wall/thread-CPU timings for one protected ASK.

No instrumentation is active on OFF/Root Cause/Smart. Parallel children share
the trace under copied ContextVars but keep separate stacks. Exclusive duration
subtracts the UNION of direct child intervals, never their (overlapping) sum.
CPU durations belong to the executing thread, not the container or request's
other workers. Hot-operation totals are inclusive/nonadditive. No prompts, IDs,
URLs or error text enter the trace. Instrumentation never changes application
guards, outputs or exception behavior.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import math
from threading import RLock, get_ident
from time import monotonic, thread_time_ns

VERSION = "protected-ask-wall-threadcpu-trace-v2"
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
HOT_NAMES = frozenset({"legacy.capture", "receipt.same_value", "receipt.adapt_candidate"})
_MAX_TICKS = (1 << 63) - 1


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
    def __init__(self, *, started, clock=monotonic, max_spans=256,
                 cpu_clock=thread_time_ns, thread_identity=get_ident):
        if (not callable(clock) or type(started) not in (int, float)
                or not math.isfinite(started) or type(max_spans) is not int
                or not 1 <= max_spans <= 1024):
            raise ValueError("invalid trace configuration")
        self.started, self.clock, self.max_spans = float(started), clock, max_spans
        self.cpu_clock, self.thread_identity = cpu_clock, thread_identity
        self.active = True
        self.rows = []
        self.dropped = 0
        self.dropped_macro = 0
        self.counters = {name: 0 for name in COUNTER_NAMES}
        self.counters_saturated = False
        self.hot_rows = {name: {"calls": 0, "completed_calls": 0, "error_calls": 0,
            "wall_ns": 0, "cpu_ns": 0, "cpu_invalid_calls": 0,
            "wall_invalid_calls": 0} for name in HOT_NAMES}
        self.hot_saturated = False
        self.hot_diagnostics_incomplete = False
        self.lock = RLock()

    def _cpu_sample(self):
        try:
            thread, tick = self.thread_identity(), self.cpu_clock()
            if type(thread) is int and type(tick) is int and tick >= 0:
                return thread, tick
        except Exception:
            pass
        return None, None

    def _hot_sample(self):
        try:
            wall = self.clock()
            if type(wall) not in (int, float) or not math.isfinite(wall):
                wall = None
        except Exception:
            wall = None
        thread, cpu = self._cpu_sample()
        return wall, thread, cpu

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
            cpu_thread, cpu_start = self._cpu_sample()
            self.rows.append({"span": number, "parent": parent, "phase": name,
                              "start": max(0.0, self.clock() - self.started),
                              "end": None, "outcome": "running",
                              "cpu_thread": cpu_thread, "cpu_start": cpu_start,
                              "cpu_end": None, "cpu_valid": False})
            return number

    def end(self, number, outcome):
        with self.lock:
            row = self.rows[number - 1]
            if row["end"] is None:
                row["end"] = max(row["start"], self.clock() - self.started)
                row["outcome"] = outcome
                thread, cpu = self._cpu_sample()
                valid = (thread is not None and thread == row["cpu_thread"]
                    and row["cpu_start"] is not None and cpu is not None and cpu >= row["cpu_start"])
                row["cpu_end"], row["cpu_valid"] = cpu if valid else None, valid

    def record_hot(self, name, started, finished, completed):
        """Three fixed aggregate buckets, never per-record events or content."""
        with self.lock:
            if not self.active:
                return
            row = self.hot_rows[name]
            wall0, thread0, cpu0 = started
            wall1, thread1, cpu1 = finished
            wall_valid = wall0 is not None and wall1 is not None and wall1 >= wall0
            cpu_valid = (thread0 is not None and thread0 == thread1
                and cpu0 is not None and cpu1 is not None and cpu1 >= cpu0)
            values = {"calls": 1, "completed_calls": int(completed), "error_calls": int(not completed),
                "wall_ns": int((wall1 - wall0) * 1_000_000_000) if wall_valid else 0,
                "cpu_ns": cpu1 - cpu0 if cpu_valid else 0,
                "cpu_invalid_calls": int(not cpu_valid), "wall_invalid_calls": int(not wall_valid)}
            for key, value in values.items():
                ceiling = _MAX_TICKS if key.endswith("_ns") else _MAX_COUNTER
                total = row[key] + value
                row[key] = min(ceiling, total)
                self.hot_saturated = self.hot_saturated or total > ceiling

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
            hot_rows = {name: dict(row) for name, row in self.hot_rows.items()}
            hot_saturated = self.hot_saturated
            hot_incomplete = self.hot_diagnostics_incomplete
        result = []
        for row in rows:
            finish = row["end"] if row["end"] is not None else now
            children = [(max(row["start"], child["start"]),
                         min(finish, child["end"] if child["end"] is not None else now))
                        for child in rows if child["parent"] == row["span"]]
            children = [(a, b) for a, b in children if b > a]
            inclusive = max(0.0, finish - row["start"])
            cpu = exclusive_cpu = None
            if row["cpu_valid"]:
                cpu = max(0, row["cpu_end"] - row["cpu_start"])
                cpu_children = [(max(row["cpu_start"], child["cpu_start"]),
                    min(row["cpu_end"], child["cpu_end"])) for child in rows
                    if child["parent"] == row["span"] and child["cpu_valid"]
                    and child["cpu_thread"] == row["cpu_thread"]]
                cpu_children = [(a, b) for a, b in cpu_children if b > a]
                exclusive_cpu = max(0, cpu - _union(cpu_children))
            result.append({"span": row["span"], "parent": row["parent"],
                           "phase": row["phase"], "outcome": row["outcome"],
                           "started_seconds": round(row["start"], 6),
                           "finished_seconds": round(finish, 6),
                           "wall_seconds": round(inclusive, 6),
                           "exclusive_wall_seconds": round(max(0.0, inclusive - _union(children)), 6),
                           "thread_cpu_seconds": round(cpu / 1_000_000_000, 6) if cpu is not None else None,
                           "exclusive_thread_cpu_seconds": round(exclusive_cpu / 1_000_000_000, 6) if exclusive_cpu is not None else None,
                           "thread_cpu_valid": row["cpu_valid"]})
        roots = [(r["start"], r["end"] if r["end"] is not None else now)
                 for r in rows if r["parent"] is None]
        return {"version": VERSION, "timing_basis": "monotonic_wall_not_transport_sum",
                "cpu_timing_basis": "same_thread_cpu_excluding_off_thread_children_not_container_cpu",
                "cpu_incomplete_spans": sum(not r["cpu_valid"] for r in rows),
                "hot_operations": {name: {
                    **{key: value for key, value in row.items() if not key.endswith("_ns")},
                    "inclusive_wall_seconds": round(row["wall_ns"] / 1_000_000_000, 6),
                    "inclusive_thread_cpu_seconds": round(row["cpu_ns"] / 1_000_000_000, 6)}
                    for name, row in sorted(hot_rows.items())},
                "hot_operations_basis": "inclusive_nonadditive_fixed_aggregate_calls",
                "hot_operations_saturated": hot_saturated,
                "hot_operations_incomplete": hot_incomplete,
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


def hot(name):
    """Time one fixed operation, with no clocks/locks on inactive paths."""
    if name not in HOT_NAMES:
        raise ValueError("unregistered ASK hot operation")
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            trace = _ACTIVE.get()
            if trace is None or not trace.active:
                return function(*args, **kwargs)
            try:
                started = trace._hot_sample()
            except Exception:
                trace.hot_diagnostics_incomplete = True
                return function(*args, **kwargs)
            completed = False
            try:
                result = function(*args, **kwargs)
                completed = True
                return result
            finally:
                try:
                    trace.record_hot(name, started, trace._hot_sample(), completed)
                except Exception:
                    trace.hot_diagnostics_incomplete = True
        return measured
    return decorate
