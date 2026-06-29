# Informe: Linealización del análisis Holistic FP para optimización diferenciable

## 1. Objetivo

Encontrar una fórmula **lineal** (o diferenciable) cuyo resultado esté correlacionado con
el análisis Holistic FP completo, para poder usarla como *surrogate* dentro de un bucle
de optimización (gradiente descendente) de forma más eficiente que evaluar el análisis
real en cada paso.

---

## 2. Versiones de la linealización

Se implementaron 3 versiones incrementales en `linearized_fp.py`:

| Versión | Fórmula de `w` | Loop externo | Loop `p` | Naturaleza |
|---|---|---|---|---|
| **V1** | `(p·C + ΣJᵧ·uᵧ) / (1 - Σuᵧ)` | Sí (convergencia) | Sí | Iterativa |
| **V2** | Ídem | No (N pasadas fijas) | Sí | Semi-iterativa |
| **V3** | Ídem (p=1) | No (sistema lineal) | No | Algebraica pura |

La linealización reemplaza `⌈(J+w)/T⌉` por `(J+w)/T + α`:
```
w = (p·Cᵢ + α·ΣCᵧ + ΣJᵧ·uᵧ) / (1 - Σuᵧ)
```

---

## 3. Correlación V1/V2/V3 con Holistic FP

Estudio con 100 sistemas aleatorios, prioridades Deadline Monotonic, U ∈ [0.5, 0.95]:

| Métrica | V1 (convergente) | V2 (2-pass) | V3 (one-shot) |
|---|---|---|---|
| `avg_wcrt` Spearman | **0.939** | 0.712 | 0.748 |
| `invslack` Spearman | **0.890** | 0.709 | 0.721 |

**V1 con iteraciones limitadas:**
| max_iter | Spearman(invslack) | Velocidad |
|---|---|---|
| 5 | 0.83 | ~40x |
| 10 | 0.87 | ~20x |
| sin límite | 0.97 | 1x (mediana 208 iters) |

---

## 4. V3 diferenciable en PyTorch

Implementación en `differentiable_v3.py`. La matriz del sistema lineal `(I-M)r = b`
se construye con tensores de PyTorch y se resuelve con `torch.linalg.solve`, que es
diferenciable.

### 4.1 Gradiente sobre WCETs

**Estudio**: `gradient_study.py` — compara `∂(avg_wcrt)/∂(wcet)`:
- V3 → autograd
- Holistic → diferencias finitas centrales

**Resultado (cosine similarity):**
| U | media cos | % > 0.8 | % negativo |
|---|---|---|---|
| 0.5–0.6 | 0.85 | 82% | 0% |
| 0.6–0.7 | 0.78 | 77% | 4.5% |
| 0.7–0.8 | 0.67 | 48% | 8.7% |
| 0.8–0.9 | 0.45 | 45% | 14% |
| **Global** | **0.67** | **61%** | **9%** |

El gradiente V3 es fiable para U < 0.7, pero se degrada con la utilización
(el modelo p=1 subestima interferencia en sistemas muy cargados).

### 4.2 Gradiente sobre prioridades (soft-priority)

**Implementación**: softmax-relaxed priority sets:
```
P(j ∈ hp(i)) = σ((sⱼ - sᵢ) / τ)
```

**Validación con swaps**: `priority_gradient_study.py`
- El gradiente V3 sugiere qué tarea promocionar/degradar
- **61%** de los swaps guiados mejoran el Holistic real frente a swaps aleatorios

---

## 5. Integración con el framework `gradient_descent`

Se implementó `V3SoftPriorityGradient` como `GradientFunction` compatible con
`GradientDescentOptimizer`. Componentes:

- **CostFunction**: `InvslackCost` (Holistic FP real, para condición de parada)
- **GradientFunction**: `V3SoftPriorityGradient` (V3 autograd)
- **ParameterHandler**: `RawPriorityHandler` (prioridades continuas ≥ 0)
- **UpdateFunction**: `NoisyAdam`

**Resultado**: El optimizador minimiza exitosamente la función de coste V3,
pero la validación Holistic muestra **cero mejora** sobre DM.

---

## 6. Experimento de schedulability (métricas globales)

Estudio con 100 sistemas, barrido U ∈ [0.5, 0.9] (20 niveles), usando `SchedRatioEval`:

### 6.1 Primera comparación

| Método | Sched (/2000) | Tiempo total | vs DM |
|---|---|---|---|
| DM | 1230 | 0.4s | — |
| **V3-opt** | **1230** | 93.9s | **+0** |
| HOPA | 1687 | 27.5s | +457 |
| **GDPA** | **1741** | 120.9s | **+511** |

V3-opt no mejora ningún sistema. GDPA (gradiente vectorizado con Holistic real)
es el mejor método.

### 6.2 Gráfica de eficiencia

Se añadió `_efficiency_chart` a `SchedRatioEval` (`examples/evaluation.py:181`)
que genera automáticamente un scatter plot `{name}_efficiency.png` con:
- Eje X: tiempo total de ejecución
- Eje Y: sistemas schedulables totales
- Cada método es un punto, con frontera de Pareto

---

## 7. Fases 1-3: V1 diferenciable

### Fase 1 — Unrolled V1

N iteraciones de V1 desenrolladas en PyTorch (N=10), con soft-priority
y soft p-loop vía softplus. Totalmente diferenciable.

### Fase 2 — Annealing

Temperatura τ programada: τ(t) = max(τ_min, τ₀ · decayᵗ).
Empieza con τ alto (paisaje suave) y decae (prioridades casi discretas).

### Fase 3 — Implicit differentiation

Resuelve r* = F(r*, θ) hasta convergencia y usa el teorema de la función
implícita para el gradiente exacto en el punto fijo.

### Resultado de las 3 fases

| Método | Sched | Tiempo | vs DM |
|---|---|---|---|
| DM | 1230 | 0.4s | — |
| V1-unroll | **1230** | 249s | **+0** |
| V1-anneal | **1230** | 248s | **+0** |
| V1-implicit | **1230** | 2275s | **+0** |
| GDPA | 1741 | 121s | +511 |

**Todas las variantes diferenciables fracasan.** La causa no es la precisión
de la fórmula de respuesta (V1 es más preciso que V3), sino la **relajación
softmax del orden de prioridades**: el paisaje de gradiente en el espacio
continuo de scores no se corresponde con el espacio discreto de asignaciones
de prioridad que evalúa Holistic FP.

---

## 8. V1-FD: gradiente por diferencias finitas sobre V1

### Enfoque

En lugar de hacer diferenciable el modelo, usar V1 (Spearman 0.94) como
*surrogate rápido* para diferencias finitas, igual que GDPA usa Holistic
pero 20x más rápido por evaluación:

```
GDPA:  ∂cost/∂x ≈ [Holistic(x+ε) - Holistic(x-ε)] / 2ε   (vectorizado)
V1-FD: ∂cost/∂x ≈ [V1(x+ε) - V1(x-ε)] / 2ε               (secuencial)
```

### Resultado

| Método | Sched | Tiempo | vs DM | vs GDPA |
|---|---|---|---|---|
| DM | 1230 | 0.4s | — | -511 |
| HOPA | 1687 | 26.8s | +457 | -54 |
| GDPA | 1741 | 117.2s | +511 | — |
| **V1-FD** | **1555** | **495.0s** | **+325** | **-186** |

**V1-FD es el primer método con surrogate que mejora sobre DM (+26.4%).**
El gradiente por diferencias finitas sobre V1 con prioridades **discretas**
(no softmax) sí es informativo.

El tiempo es mayor que GDPA (495s vs 117s) porque V1-FD es secuencial
(24 evaluaciones independientes de V1 por paso), mientras GDPA es
vectorizado (24 escenarios en una sola pasada del análisis Holistic).

---

## 9. Conclusiones

### Lo que funciona

1. **GDPA** (gradiente vectorizado con Holistic real) es el mejor método:
   +511 sistemas sobre DM, 117s de tiempo total.

2. **V1-FD** (diferencias finitas sobre V1 con prioridades discretas) es viable:
   +325 sistemas sobre DM. Necesita vectorización para ser competitivo.

3. **V1** como surrogate tiene Spearman 0.94 con Holistic — excelente para
   ranking y evaluación rápida.

### Lo que NO funciona

4. **Cualquier relajación diferenciable del orden de prioridades** (softmax,
   sigmoid, annealing) produce gradientes que optimizan el surrogate pero
   no transfieren al Holistic real. El problema es la discrepancia entre el
   paisaje continuo de scores y el espacio discreto de asignaciones.

5. **V3** (one-shot, p=1) es demasiado impreciso como surrogate para
   optimización (Spearman 0.75).

### Lecciones

6. La **dirección** del gradiente es más importante que la **precisión**
   del surrogate. V1 (94% correlación) produce gradientes utilizables;
   V3 (75%) no.

7. La **vectorización** es crítica para el rendimiento. GDPA evalúa 24
   escenarios en una pasada; V1-FD necesita 24 pasadas secuenciales.

8. El **gráfico de eficiencia** (schedulability vs tiempo) añadido a
   `SchedRatioEval` permite comparar métodos en una sola imagen.

### Trabajo futuro

- **Vectorizar V1**: implementar una versión vectorizada del análisis V1
  (similar a `VectorHolisticFPAnalysis`) para que V1-FD compita en velocidad
  con GDPA.

- **V1-FD + V1 cost**: usar V1 tanto para el gradiente como para la función
  de coste (condición de parada), eliminando completamente Holistic del bucle
  de optimización y validando solo al final.

---

## 10. Archivos generados

```
workspace/holistic_linearization/
├── INFORME.md                        # Este informe
├── linearized_fp.py                  # V1, V2, V3 (análisis linealizados)
├── differentiable_v3.py              # V3 en PyTorch (WCET + soft-priority)
├── differentiable_v1.py              # Fases 1-3: V1 unrolled/annealed/implicit
├── v3_gradient.py                    # V3SoftPriorityGradient
├── v1_gradients.py                   # V1Unrolled/Annealed/Implicit gradients
├── v1_fd_gradient.py                 # V1FiniteDifferenceGradient ★
├── study.py                          # Correlación V1/V2/V3 vs Holistic
├── gradient_study.py                 # Cosine similarity V3 vs Holistic
├── priority_gradient_study.py        # Validación swaps de prioridad
├── v3_optimization_example.py        # Ejemplo optimización V3
├── schedulability_experiment.py      # Experimento principal (todos vs todos)
├── efficiency_plot.py                # Post-procesado (independiente)
│
├── correlation_results.csv           # Datos correlación
├── correlation_scatter.png           # 9-panel scatter
├── gradient_correlation.csv          # Datos gradiente
├── gradient_correlation.png          # Histograma cosine similarity
├── priority_gradient_results.csv     # Datos swaps
│
├── v3_schedulability_*.png/xlsx      # Experimento V3 vs HOPA vs GDPA
├── priority_optimization_*.png/xlsx  # Experimento 7 métodos
├── v1fd_vs_gdpa_*.png/xlsx           # Experimento V1-FD vs GDPA ★
│
└── priority_optimization_v2_*.png/xlsx  # Experimento 8 métodos (parcial)
```

### Modificación externa

- `examples/evaluation.py` — añadido `_efficiency_chart()` que genera
  automáticamente scatter plot schedulability vs tiempo en todos los estudios.
