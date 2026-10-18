"""Your workspace: the vectorized local-EDF analysis goes here.

Entry point expected by ``run.py``: a module with an ``analyze`` function.
The default module is this one, ``algorithms/analyze.py``; you can create other
modules and select them with ``python run.py --module algorithms.<name>``.

See ``AGENTS.md`` for the goal and ``frozen/analysis.py`` for the reference.
"""


def analyze(system, deadlines):
    """Return the worst-case response times for a batch of deadline assignments.

    Parameters
    ----------
    system : frozen.model.LinearSystem
        The system under analysis: a set of flows, each an ordered chain of
        tasks mapped onto processors (all processors are EDF here). Its tasks
        already have a ``wcet``, ``period``, ``flow.deadline`` and so on; you
        must not change the task->processor mapping.
    deadlines : sequence of shape ``(K, n_tasks)``
        ``K`` independent local-deadline assignments. ``deadlines[j][i]`` is the
        local deadline of task ``i`` (in ``system.tasks`` order) for assignment
        ``j``. ``K`` can be large; assume you should evaluate all of them
        together rather than one at a time.

    Returns
    -------
    wcrts : shape ``(K, n_tasks)``
        ``wcrts[j][i]`` is the worst-case response time of task ``i`` under
        assignment ``j``, matching what the scalar reference analysis in
        ``frozen/analysis.py`` would produce (per task, in ``system.tasks``
        order).

    Notes
    -----
    - The reference computes a WCRT per task with the holistic local-EDF
      analysis (jitter propagates across the flow; interference depends on the
      absolute-deadline points). Reproduce those values.
    - Do not mutate ``system`` in a way that leaks across calls; return results,
      do not rely on side effects.
    - ``run.py`` compares your output against the scalar reference and reports
      the speedup of one batched call over K scalar calls.

    Delete this body and implement it.
    """
    raise NotImplementedError("implement the vectorized analysis here")
