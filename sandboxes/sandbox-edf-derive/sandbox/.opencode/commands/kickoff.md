---
description: Start deriving a non-iterative deadline-assignment rule
---

Start working now. Follow `AGENTS.md`.

**Step 0 first**: study `frozen/analysis.py` (and `frozen/model.py`) in detail —
where every local deadline enters, and why an interferer with a later deadline
interferes less — and write it in `DERIVATION.md`. Use `frozen/instrumented.py`
to confirm your reading on real systems. Do this *before* deriving anything.

Then: reproduce the baseline (`python run.py --per-level -q`), and derive the
rule (restricted cases first, then generalise), implementing it as a module in
`algorithms/` and validating it on the pool. Keep `DERIVATION.md` updated with
the maths, assumptions, numbers and failure cases. Do not modify `frozen/` or
`run.py`.
