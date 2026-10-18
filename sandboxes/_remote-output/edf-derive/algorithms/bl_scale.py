"""Back-loaded, magnitude-scaled local deadlines (the one-pass rule).

Derived rule (see ``DERIVATION.md``). For a flow with WCET shares
``s_i = C_i · D_flow / Σ C`` (so ``Σ_i s_i = D_flow``) and cumulative prefix
``P_{i-1} = s_0 + … + s_{i-1}``, set

    r_i = s_i + g · P_{i-1}                       (back-loaded shape)
    d_i = r_i · λ · D_flow / Σ_j r_j              (uniform magnitude λ)

with g = 0.3 and λ = 2.0.

Mechanism. In the analysis, an interferer ``i`` is credited with
``m = min(pl, pd)`` activations inside the analysed window, where

    pd = 0                              if D <  d_i
       = floor((j_i + D − d_i)/T_i) + 1 otherwise

so a **later** local deadline ``d_i`` means **fewer** credited activations
(eq 1). Two consequences drive the rule:

* *back-loading* (g > 0): later tasks of a flow carry the larger local
  deadlines. Their own response is dominated by their (already large) release
  jitter ``j_i``, so the marginal cost of a larger deadline is smallest there,
  while the benefit (less interference on every co-processor task) is largest.
* *magnitude* (λ > 1): inflating all deadlines slides the interferers' deadline
  points out of the analysed windows, removing interference. λ is the interior
  optimum: below it the caps do not bite, above it the analysed task's own
  ``+d_a`` term (eq 9) and the deadline cap stop paying.

One pass, no search. Scores 865/1000 on the shipped pool (seed 42).
"""

from algorithms import register


def _shares(flow):
    total = sum(t.wcet for t in flow.tasks)
    out, cum = [], 0.0
    for t in flow.tasks:
        s = t.wcet * flow.deadline / total
        out.append((t, s, cum))
        cum += s
    return out


def bl_shape_scale(system, g=0.3, lam=2.0):
    """Assign the back-loaded, λ-scaled local deadlines in place."""
    for flow in system.flows:
        rows = _shares(flow)
        raw = [(t, s + g * prefix) for (t, s, prefix) in rows]
        scale = lam * flow.deadline / sum(r for _, r in raw)
        for t, r in raw:
            t.deadline = r * scale


@register
def bl_scale(system):
    """Back-loaded shape (g=0.3) with uniform magnitude λ=2.0 (one pass)."""
    bl_shape_scale(system, g=0.3, lam=2.0)
