---
description: Start studying the local-EDF deadline-assignment sandbox
---

Start working now. Follow `AGENTS.md`.

**Step 0 first**: study `frozen/analysis.py` (and `frozen/model.py`) in detail
and write your understanding of it in the "Analysis" section of `NOTES.md` —
especially where each task's local deadline enters the interference term and the
response time, and why larger local deadlines can help. Do this *before* the
experiments.

Then reproduce the baseline:

```bash
python run.py --per-level -q
```

Then work through the research questions in `AGENTS.md` **in order**, stating the
prediction implied by the analysis before each experiment and keeping `NOTES.md`
up to date with the numbers and conclusions. Add your techniques as new modules
in `algorithms/` (register them with `@register`); for ad-hoc probes use
`harness.evaluate({...})` directly. Do not modify `frozen/` or `run.py`.
