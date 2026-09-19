# map-bf-10

Joint task-to-processor mapping + fixed priorities, **10 tasks** on 3 processors
(generator size `(2,5,3)`), **unbalanced** initial mapping. One step up from
`map-bf-9` (search space 12x larger).

- Population: same settings as the other map-bf variants (`bf.py`), seed 42,
  25 systems, 20 levels, `set_system_utilization` (contended initial mapping).
- Methods: `pd`, `hopa`, `gdpa-prio`, `gdpa-100`, `gdpa-200`, `gdpa-500`, `bf`
  (vectorized) and `bf-seq` (scalar; both exhaustive and agree at 406/500).
- Headline schedulable out of 500: bf 406, **gdpa-500 393 (gap 13)**, gdpa-200
  380, gdpa-100 360, hopa 257, gdpa-prio 256, pd 207.
- Search space: `10!·C(12,2)` = 239,500,800. This is the largest size where the
  (vectorized) brute force is still practical; 12 tasks is not.

Regenerate: `python workspace/framework_paper/map-bf/bf.py --size 10`
