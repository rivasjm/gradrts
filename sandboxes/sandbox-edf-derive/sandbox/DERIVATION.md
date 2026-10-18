# Derivation

> Main deliverable. Write it for a reader who must be able to **follow the
> derivation and reproduce the numbers**. Keep it honest: separate what you
> *derived* from what you only *observed*.

## 0. Setup and notation

<!-- The model, the analysis, the symbols (C_i, T_i, d_i, j_i, psi, p, w_ab, r_ab).
     Where every local deadline enters (analysed task vs interferer). -->

## 1. Understanding the analysis (Step 0)

<!-- Your reading of frozen/analysis.py, confirmed with frozen/instrumented.py. -->

## 2. Restricted cases (derive the shape)

### 2.1 One processor, two tasks, no jitter
### 2.2 Add jitter / a chain
### 2.3 Several flows per processor

## 3. The rule

<!-- State d = g(system) precisely, with its assumptions. Show it is
     non-iterative (or how many passes). -->

## 4. Validation

<!-- Same pool/seed/interpreter as the sibling sandbox. Table of schedulables:
     pd, pd_global, best gamma/lambda, a search headroom, and YOUR rule.
     Per-utilization breakdown. Exact commands. -->

## 5. Validity domain and failure cases

<!-- Where it is near-optimal, where it fails, and why (use the instrumented
     breakdown to explain). -->

## 6. Discussion

<!-- Is it exploiting the approximate analysis? How would it generalise
     (other sizes/mappings)? Open questions. -->
