---
description: Adversarially verify the derivation in DERIVATION.md
---

You are an **independent verifier**. Do **not** trust `DERIVATION.md` — try to
**falsify** it.

Read `AGENTS.md`, `frozen/analysis.py`, `frozen/instrumented.py`,
`DERIVATION.md` and the modules in `algorithms/` (plus probe scripts in `tmp/`).
Then:

- re-do the **derivation** yourself: are the steps valid? Do the assumptions
  hold? Does the final rule actually follow, or are there hidden assumptions?
- **Reproduce the numbers** from scratch (same interpreter: the repo venv,
  Python 3.13; same pool/seed) and report match or mismatch;
- hunt for **counterexamples**: systems/levels where the rule is claimed near-
  optimal but is not, and cases where the derivation's assumptions break;
- check whether a success is **exploiting the approximate analysis** rather than
  the modelled system.

Write **`VERIFICATION.md`** (do **not** edit `DERIVATION.md`): per claim,
*reproduced? counterexample? verdict (confirmed / refuted / uncertain)* and the
evidence. Be blunt about anything unsupported or overstated.
