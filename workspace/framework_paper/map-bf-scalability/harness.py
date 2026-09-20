"""Raw evaluator for the map-bf scalability scenario.

A deliberately small harness: it takes a *matrix of systems* and a list of
tools (callables ``system -> bool``), runs every tool on every system and
records, for each pair, whether the system became schedulable and how long the
tool took.

Matrix convention
-----------------
``systems[row][col]``: all systems in a column share a characteristic (the
task count in this scenario), the rows are different systems. Systems are
dispatched column by column, row by row (all systems of the first column, then
the second, ...); ``threads`` systems run in parallel and a free worker takes
the next one.

Results are raw: a list of ``{"system", "row", "column", "tool", "schedulable",
"time"}`` dictionaries. ``row`` is the matrix row (a system has no intrinsic
meaning, but the same row across columns is the same base system) and
``column`` is the caller-provided label of the matrix column. ``time`` is
``None`` when the tool exceeded its budget (a timeout counts as not
schedulable); it is a finite number of seconds otherwise, including when the
tool ran to completion without finding a schedule.

Timeouts
--------
``multiprocessing.Pool`` cannot kill a running worker, so the budget is
enforced *inside* each worker with ``SIGALRM``: a tool that overruns raises
and the pair is recorded as a timeout. This interrupts Python loops (the
brute force iterates in Python) but not a long C call in progress.

Usage::

    records = evaluate(systems, labels, funcs, threads=6, timeouts={...})
    evaluate(systems, labels, funcs, output="raw.json")
"""

import json
import os
import signal
import time
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Sequence

from model.linear_system import LinearSystem

DEFAULT_TIMEOUT = 300.0
DEFAULT_THREADS = 6


class _Timeout(Exception):
    """Raised inside a worker when a tool exceeds its time budget."""


def _raise_timeout(signum, frame):
    raise _Timeout()


def _evaluate_system(task, labels: Sequence[str],
                     funcs: Sequence[Callable], timeouts: Dict[str, float]):
    """Run every tool once on a private copy of ``task = (system, row, column)``."""
    system, row, column = task
    records = []
    for label, func in zip(labels, funcs):
        work = deepcopy(system)
        started = time.perf_counter()
        timed_out = False
        try:
            signal.signal(signal.SIGALRM, _raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, timeouts[label])
            schedulable = bool(func(work))
        except _Timeout:
            timed_out = True
            schedulable = False
        except Exception as exc:  # a tool failure is a failed fix, not a crash
            schedulable = False
            print(f"error tool={label} system={system.name}: {exc!r}", flush=True)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            elapsed = time.perf_counter() - started
        records.append({
            "system": system.name,
            "row": row,
            "column": column,
            "tool": label,
            "schedulable": schedulable,
            "time": None if timed_out else elapsed,
        })
    return system.name, records


def _format(system_records) -> str:
    name = system_records[0]["system"]
    parts = []
    for rec in system_records:
        seconds = "timeout" if rec["time"] is None else f"{rec['time']:.2f}s"
        parts.append(f"{rec['tool']}={'1' if rec['schedulable'] else '0'}({seconds})")
    return f"{name} | " + " ".join(parts)


def _write_json(path: str, records: List[dict]) -> None:
    """Atomically replace ``path`` with the JSON array ``records``."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        json.dump(records, handle, indent=2)
    os.replace(tmp, path)


def evaluate(systems: List[List[LinearSystem]], labels: Sequence[str],
             funcs: Sequence[Callable], threads: int = DEFAULT_THREADS,
             columns: Optional[Sequence[str]] = None,
             timeouts: Optional[Dict[str, float]] = None,
             default_timeout: float = DEFAULT_TIMEOUT,
             output: Optional[str] = None,
             on_column: Optional[Callable[[str, List[dict], List[str]], None]] = None,
             verbose: bool = True) -> List[dict]:
    """Run ``funcs`` on every system and return the raw records.

    ``systems`` is a rectangular ``rows x cols`` matrix. ``columns`` gives a
    label per column (the characteristic the column represents, e.g. the task
    count); it is copied into every record as ``"column"``. ``timeouts`` maps a
    tool label to its budget in seconds (labels missing from the dict use
    ``default_timeout``). If ``output`` is given the raw records are written
    there as a JSON array, rewritten (atomically) after every finished system
    so a partial run still leaves the results obtained so far.

    ``on_column(label, records, finished)`` is called in the main process
    every time a column's last system finishes, with the records accumulated so
    far and the list of already-finished column labels in matrix order (used by
    the runner to refresh the processed Excel with only complete columns).
    """
    if len(labels) != len(funcs):
        raise ValueError("labels and funcs must have the same length")
    if not systems:
        return []
    n_cols = len(systems[0])
    if any(len(row) != n_cols for row in systems):
        raise ValueError("systems matrix must be rectangular")
    if columns is None:
        columns = [str(col) for col in range(n_cols)]
    if len(columns) != n_cols:
        raise ValueError("columns must have one label per matrix column")

    budget = {label: float((timeouts or {}).get(label, default_timeout))
              for label in labels}
    ordered = [(systems[row][col], row, columns[col])
               for col in range(n_cols) for row in range(len(systems))]

    worker = partial(_evaluate_system, labels=labels, funcs=funcs, timeouts=budget)
    order = {label: i for i, label in enumerate(labels)}
    sort_key = lambda rec: (rec["system"], order[rec["tool"]])  # noqa: E731
    remaining = {label: len(systems) for label in columns}
    finished = set()

    records: List[dict] = []
    done = 0
    with Pool(threads) as pool:
        for _, system_records in pool.imap_unordered(worker, ordered):
            records.extend(system_records)
            done += 1
            if output:
                records.sort(key=sort_key)
                _write_json(output, records)
            if on_column:
                column = system_records[0]["column"]
                remaining[column] -= 1
                if remaining[column] == 0:
                    finished.add(column)
                    ordered_finished = [c for c in columns if c in finished]
                    on_column(column, records, ordered_finished)
            if verbose:
                print(f"[{done}/{len(ordered)}] {_format(system_records)}",
                      flush=True)

    if output:
        records.sort(key=sort_key)
        _write_json(output, records)
    return records
