---
description: Adversarially verify the findings in NOTES.md
---

You are an **independent verifier**. Do **not** trust `NOTES.md` — your job is to
try to **falsify** it.

Read `AGENTS.md`, `frozen/analysis.py`, `NOTES.md` and the modules in
`algorithms/` (plus any probe scripts in `tmp/`). For every claimed result:

- **Reproduce** the key numbers from scratch (same interpreter: the repo venv,
  Python 3.13; same pool/seed) and report match or mismatch;
- hunt for **counterexamples** and edge cases (other sizes/mappings if relevant,
  boundary utilizations, degenerate flows);
- check whether each conclusion actually **follows from the analysis** or is just
  correlation — test the *mechanism* it claims (e.g. does raising an interferer's
  local deadline really reduce its counted interference, and by the stated
  amount?).

Then write **`VERIFICATION.md`** (do **not** edit `NOTES.md`) with, per claim:
*reproduced? counterexample? verdict (confirmed / refuted / uncertain)* and the
evidence. Be blunt about anything unsupported or overstated, and list the claims
you could not check and why.
