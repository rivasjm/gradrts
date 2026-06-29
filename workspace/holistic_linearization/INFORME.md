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

---

## 11. Vectorización de V1 (WP-1)

### Motivación

V1-FD (§8) demostró que V1 con prioridades discretas produce gradientes
informativos (+325 sistemas sobre DM), pero queda por debajo de GDPA
(1741) por ser **secuencial** (24 evaluaciones V1 por paso). Se hipotetizó
que vectorizando V1 se obtendría un surrogate ~10–20× más rápido por
evaluación que Holistic, situando a V1-FD en o por encima de la frontera de
Pareto de GDPA.

### Implementación (`vector_v1.py`)

`VectorLinearizedV1Analysis` replica la API de `VectorHolisticFPAnalysis`
y reutiliza su infraestructura (`ResultsCache`, `successor_matrix`,
`jitter_matrix`, pruning de escenarios sobre-límite, cache bidireccional).

La clave del speedup teórico es sustituir el bucle interno deHolistic-vector
(convergencia de `w` mediante fixed-point iterado) por la fórmula cerrada
de V1:

```
w = (p * C_i + JU_hp) / (1 - U_hp)      # cerrado, sin iteración
r = w - (p-1)*T + j
r_max = max(r_max, r)
```

Bucle externo: hasta que ningún escenario cambie (finalización clásica
de Jacobi). Cada iteración `p` se evalúa para todos los escenarios
activos en una sola llamada matmul (`pm_f @ ju`).

### Validación de equivalencia (`test_vector_v1.py`)

50 sistemas aleatorios 3f × 4t × 3p:

| Métrica                          | valor       |
|----------------------------------|-------------|
| Diferencia WCRT vs V1 secuencial | 8.5e-14     |
| Mismatches (>1e-3 rel)           | 0 / 50      |
| Spearman vs Holistic (pooled)    | 0.978       |

Las implementaciones secuencial y vectorizada son numéricamente idénticas
y V1 mantiene su correlación de ranking con Holistic.

### Benchmark de speed (`benchmark_vector_v1.py`)

Tiempo por evaluación (μs) comparando V1-vectorizado y Holistic-vectorizado
con caches frescas por llamada (régimen de gradiente):

| U     | V1vec  | Holvec | ratio V1/Hol |
|-------|--------|--------|--------------|
| 0.50  |  66 µs | 37 µs  | 1.78×        |
| 0.70  | 181 µs | 86 µs  | 2.11×        |
| 0.85  | 1132 µs | 330 µs | 3.43×       |
| 0.90  | 1346 µs | 527 µs | 2.55×        |

Con cache compartido por sistema (régimen de muchas perturbaciones del
mismo sistema, que es el caso real):

| U     | V1vec  | Holvec |
|-------|--------|--------|
| todos | ~9 µs  | ~10 µs |

### Conclusión

**La vectorización de V1 no aporta el speedup esperado.** En régimen con
cache (el relevante en optimización), V1 y Holistic tienen coste por
evaluación comparable (~10 µs). El "20× más rápido" de V1-secuencial en
el INFORME §8 era un artefacto de comparar V1-secuencial contra
Holistic-secuencial en Python puro, donde el bucle interno `w` de
Holistic es caro. Vectorizado, ese bucle desaparece y ambos convergen.

V1 sigue siendo útil como **ranking surrogate** (Spearman 0.978), pero
no como **evaluación rápida**.

### Archivos generados

```
workspace/holistic_linearization/
├── vector_v1.py              # VectorLinearizedV1Analysis ★
├── test_vector_v1.py         # Equivalencia + benchmark  ★
├── benchmark_vector_v1.py    # V1vec vs Holvec, μs/eval  ★
└── (logs de benchmark inline en consola)
```

---

## 12. V1-TopK: exploración barata + validación Holistic (WP-2)

### Motivación

Visto que V1 es un surrogate de ranking excelente (Spearman 0.94/0.978)
y que V1-vectorizado permite evaluar 24–100 escenarios en una sola pasada
numpy (§11), se hipotetizó una vía **no iterativa**:

1. Generar N candidatos de asignación de prioridad.
2. Ranquearlos con una sola llamada a `VectorLinearizedV1Analysis`.
3. Validar los K mejores con Holistic real y devolver el primer plano.

Coste total por sistema ≈ N V1-evals + K Holistic-evals, frente a GDPA que
gasta ~720 Holistic-evals (30 pasos × 24 escenarios).

### Implementación (`v1_topk.py`, `v1topk_experiment.py`)

`v1_topk_assign(system, n_candidates, k, perturbations)`:

- **Candidato 0**: asignación DM (ancla — garantiza no regresar vs DM).
- **Candidatos 1..N-1**: dos modos de muestreo:
  - `perturbations=0` (uniforme): prioridades aleatorias por procesador.
  - `perturbations>0` (DM-anchored): se aplican `perturbations` swaps
    adyacentes aleatorios dentro de cada procesador, partiendo de DM.
- **Ranking V1**: una llamada a `VectorLinearizedV1Analysis.apply(system,
  scenarios=pm)` con todos los candidatos batched, coste = avg flow WCRT.
- **Validación**: los K mejores candidatos se insertan y validan con
  `HolisticFPAnalysis(limit_factor=1, reset=True)`. Se devuelve el primer
  schedulable.

`V1TopKMethod` es un callable picklable para `SchedRatioEval`.

### Búsqueda de hiperparámetros (piloto, 10 sistemas × 20 U = 200)

Variamos `n_candidates` (N), `k` (top-K) y `perturbations` (swaps DM):

| N   | k  | pert | V1-TopK | Tiempo |
|-----|----|------|---------|--------|
| 100 |  5 | 0    | 109     | 6.4 s  |
| 100 |  5 | 1    |  99     | 2.8 s  |
| 100 |  5 | 2    | 110     | 4.7 s  |
| 100 |  5 | 3    | 117     | 5.5 s  |
| 100 | 10 | 3    | **128** | 5.3 s  |
| 300 |  5 | 3    | 110     | 8.8 s  |

Observaciones:
- Aumentar N apenas mueve la aguja (de 110 a 110 pasando de 100 a 300).
- Aumentar k sí ayuda (de 117 a 128 pasando de 5 a 10).
- El modo DM-anchored (`pert>0`) mejora sobre uniforme hasta `pert≈3`, luego se satura.
- Mejor configuración piloto: **N=100, K=10, pert=3 → 128/200**.

HOPA en el mismo piloto alcanza 153/200; GDPA 156/200. V1-TopK ya está
claramente por debajo.

### Experimento completo (`v1topk_experiment.py`)

Configuración del mejor piloto: N=100, K=10, pert=3, 100 sistemas 3f×4t×3p,
barrido U ∈ [0.5, 0.9] (20 niveles, 2000 evaluaciones totales por método).

| Método      | Sched (/2000) | Tiempo total (s) | vs DM          |
|-------------|---------------|------------------|----------------|
| DM          | 1230          | 0.4              | —              |
| HOPA        | 1687          | 31               | +457           |
| GDPA        | 1741          | 123              | +511           |
| **V1-TopK** | **1398**      | **620**          | **+168**       |

Por nivel de utilización (cada celda = sistemas schedulables de 100):
V1-TopK sigue a HOPA hasta U≈0.68 y luego colapsa; por encima de U=0.79
todos los métodos fallan (incluida GDPA).

### Comparación con V1-FD (§8)

V1-TopK es **peor que el propio V1-FD** del INFORME §8, que ya no era
competitivo frente a GDPA:

| Método      | Sched | Tiempo | vs DM |
|-------------|-------|--------|-------|
| V1-FD       | 1555  | 495 s  | +325  |
| V1-TopK     | 1398  | 620 s  | +168  |

### Conclusión: por qué fracasa V1-TopK

La hipótesis de trabajo era que V1 + ranking resolvería el problema de
generar buenas asignaciones. Dos hallazgos la desmienten:

1. **El coste por evaluación V1 ≈ coste por evaluación Holistic**
   (§11). La "exploración barata" no es barata: V1-vectorizado y
   Holistic-vectorizado convergen a ~10 µs/eval con cache. En el régimen
   sin cache, V1 es incluso 2-3× más lento que Holistic a alta U. Por lo
   tanto N V1-evals + K Holistic-evals ≈ (N+K) Holistic-evals, comparable
   o peor que los 720 Holistic-evals de GDPA en coste absoluto.

2. **El cuello es la cobertura del espacio, no la identificación.**
   El espacio de permutaciones por procesador en un sistema 4t×3p es
   4!³≈13 824 combinaciones. Muestrear N=100 (0.7% del espacio) tiene
   baja probabilidad de contener un punto schedulable en sistemas muy
   cargados. V1 ranquea correctamente esos 100, pero ningún candidato
   es realmente bueno. HOPA y GDPA resuelven esto con estructura:
   HOPA via hill-climbing con heurística ka/kr que CAMBIA de vecino
   según el paisaje; GDPA via gradiente informado que mueve las
   prioridades en la dirección de máximo descenso del coste. Muestreo
   no informado no escala.

La mejora marginal con DM-anchored swaps (`pert=3`: 109→117→128 para
k=5,10) confirma que la **estructura del problema no está en el
espacio de permutaciones lejos de DM**, sino en trayectorias informadas
hacia la frontera. La Spearman 0.97 de V1 mide la calidad del ranker,
no la del muestreador.

### Lesson aprendida

> Un ranking perfecto sobre muestras mediocres sigue dando resultados mediocres.
> La calidad de la exploración (cómo se generan candidatos) domina a la
> calidad de la evaluación (cómo se ranquean) cuando el espacio de búsqueda
> es combinatorio. Esto es lo opuesto a la conclusión de §9 lección 6
> ("la dirección del gradiente es más importante que la precisión del
> surrogate"), donde GDPA sí tenía un muestreador informado (gradiente).

### Archivos generados

```
workspace/holistic_linearization/
├── v1_topk.py                       # v1_topk_assign() + V1TopKMethod  ★
├── v1topk_experiment.py             # DM/HOPA/GDPA/V1-TopK via SchedRatioEval  ★
├── v1topk_vs_baseline_schedulables.{png,xlsx}
├── v1topk_vs_baseline_schedulables_summary.png
├── v1topk_vs_baseline_times.{png,xlsx}
├── v1topk_vs_baseline_times_summary.png
└── v1topk_vs_baseline_efficiency.png
```

---

## 13. Estado final y vías restantes

### Resumen acumulado de métodos

| Método          | Sched (/2000) | Tiempo (s) | vs DM  | Frontera Pareto |
|-----------------|---------------|------------|--------|-----------------|
| DM              | 1230          | 0.4        | —      | eficiente (ref) |
| HOPA            | 1687          | 31         | +457   | eficiente       |
| **GDPA**        | **1741**      | **123**    | **+511** | **dominante** |
| V1-FD (§8)      | 1555          | 495        | +325   | dominado        |
| V1-TopK (§12)   | 1398          | 620        | +168   | dominado        |
| V3-opt (§6)     | 1230          | 94         | +0     | dominado        |
| V1-unroll (§7)  | 1230          | 249        | +0     | dominado        |
| V1-anneal (§7)  | 1230          | 248        | +0     | dominado        |
| V1-implicit (§7)| 1230          | 2275       | +0     | dominado        |

**GDPA** es el único método no dominado. Todos los enfoques con surrogate
V1/V3 (diferenciable o por muestreo) quedan dominados por HOPA o GDPA.

### Conclusiones consolidadas

1. **El surrogate V1 es útil para ranking** (Spearman 0.94/0.978) y para
   gradientes por diferencias finitas sobre prioridades discretas (V1-FD,
   +325 sobre DM), pero **no para exploración aleatoria** con un
   presupuesto fijo de muestras: V1-TopK no supera a HOPA pese a ser un
   ranker casi perfecto, porque el muestreo no informado no cubre el
   espacio combinatorio.

2. **Vectorizar V1 no acelera la optimización**: su coste por evaluación
   es comparable a Holistic-vectorizado (~10 µs/eval con cache, §11). La
   ventaja observada en §8 era frente a Holistic en Python puro. En
   régimen vectorizado ambos convergen.

3. **Lo que funciona es el gradiente informado.** GDPA mueve simultáneamente
   todas las prioridades en la dirección de máximo descenso, usando el
   gradiente calculado por diferencias finitas vectorizadas sobre Holistic
   real. El surrogate (V1, V3, softmax) pierde valor frente a la
   estructura del gradiente.

### Vías restantes

| Vía | Estado | Justificación |
|-----|--------|---------------|
| (A) Optimizar GDPA directo | **abierta** | único método no dominado; init, σ-schedule, stop, batch size |
| (B) V1-TopK (este §12) | **cerrada** | dominado por HOPA |
| (C) V1-TopK como init de GDPA | abierta |might reducir iteraciones de GDPA si V1 encuentra un buen punto de partida, pero trivial dado el fallo de §12 |
| Híbrido: escenario caliente transferido | abierta | el cache de Holistic acumula resultados a lo largo del gradiente; mover escenarios entre pasos podría reducir evaluaciones totales |
| Relajación del orden de prioridades | descartada | §7: la relajación no transfiere al espacio discreto |
