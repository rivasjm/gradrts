# map-bf-9

Joint task-to-processor mapping + fixed priorities, **9 tasks** on 3 processors
(generator size `(3,3,3)`), **unbalanced** initial mapping.

- Population (`bf.py`): seed 42, 25 systems, periods 100-1000, deadline factors
  0.5-1. The initial mapping is a contended one (`unbalance_contended` in
  `examples/generator.py`) and the utilization sweep uses
  `set_system_utilization`, so at high utilization a processor ends up above 1.
- Methods: `pd`, `hopa`, `gdpa-prio`, `gdpa-100`, `gdpa-200`, `gdpa-500`,
  `bf` (vectorized brute force) and `bf-seq` (scalar brute force, size-9 only).
- Reference: `bf` and `bf-seq` are exhaustive and agree (459/500).
- Headline schedulable out of 500: bf 459, **gdpa-500 453 (gap 6)**, gdpa-200
  435, gdpa-100 425, gdpa-prio 270, hopa 265, pd 223.
- Search space: `9!·C(11,2)` = 19,958,400.

Regenerate: `python workspace/framework_paper/map-bf/bf.py --size 9`
