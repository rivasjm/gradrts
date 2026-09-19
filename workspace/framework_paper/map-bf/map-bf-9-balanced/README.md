# map-bf-9-balanced

Same population and size as `map-bf-9` (9 tasks, generator size `(3,3,3)`) but
with a **balanced** initial mapping (`assign_balanced`) and **`set_utilization`**
(every processor set to exactly the sweep utilization, so no processor is
overloaded at u < 1).

- Methods: `pd`, `hopa`, `gdpa-prio`, `gdpa-100`, `gdpa-200`, `gdpa-500`, `bf`
  (no `bf-seq`).
- Headline schedulable out of 500: bf 493, **gdpa-500 464 (gap 29)**, gdpa-100
  458, gdpa-200 458, gdpa-prio 443, hopa 431, pd 402.
- Balanced systems are easier (bf solves more) but the gap vs brute force is
  larger than in the unbalanced variant; it appears only at u >= 0.8.

Regenerate: `python workspace/framework_paper/map-bf/bf.py --size 9 --balanced`
