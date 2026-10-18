# HANDOFF — "Less Urgent, More Schedulable" (deadline assignment under local EDF)

Documento de traspaso para continuar en una **sesión limpia**. Estado: 18 de
octubre de 2026, tras la campaña de resultados y la reescritura del paper.

---

## 0. TL;DR

- **El paper**: `code/workspace/edf_wad/paper/main.tex` (IEEEtran conference,
  **8 páginas**). PDF: `main.pdf`. Título: *"Less Urgent, More Schedulable: Local
  Deadline Assignment in Distributed EDF Systems"*.
- **La idea**: bajo el análisis holístico de EDF local (LC-EDF), asignar
  deadlines locales **más grandes** aumenta el número de sistemas que el análisis
  certifica, de forma **no monótona** (óptimo interior). El mecanismo es el
  ***demand cap*** en la cota de interferencia (Prop. 1, probado). Se explota con
  **WAD** (una pasada, forma cerrada) y **WAD-Adapt** (3 pasadas), y se compara
  con **RTA-CD**, una búsqueda por coordenadas que da la cota superior.
- **Matiz importante**: el fenómeno es **del análisis**, no de EDF en general.
  Está dicho así en el abstract, en el enunciado de la observación y en la
  conclusión. El test es suficiente (pesimista), así que lo certificado es
  planificable de verdad, pero el tamaño de la ganancia y la posición del óptimo
  podrían cambiar con un análisis más ajustado.
- **Crédito**: el fenómeno y las dos familias extremas (escalar el deadline =
  `λ`; usar los *global scheduling deadlines* = `γ=1`) están en **rivas15**
  (TPDS 2015). Lo nuevo: el mecanismo probado, la globalización parcial /
  forma back-loaded con WAD/WAD-Adapt, la ablación de forma, y RTA-CD como techo.
- **GDPA ya no aparece en el paper** (era un trabajo no publicado). No hay
  autocita ni cita al paper compañero.

---

## 1. Dónde está todo

```
code/workspace/edf_wad/
  HANDOFF.md  README.md  RESULTS_2026-10-18.md   # mapa, índice y campaña

  # evaluación principal
  edf_wad.py         # Tablas I y II (familia WAD + PD/EQF/EQS); RTA-CD solo en 15
  hospa.py           # HOSPA autocontenido -> hospa-15/ y hospa-25/
  dump_deadlines.py  # corre un método y vuelca conteos, tiempos y deadlines
  deadline_shape.py  # ajuste (gamma, lambda) de los volcados
  flatten_check.py   # control de forma a igual gasto (rampa vs plano)
  probe_rta_cd.py    # sondeo de convergencia de RTA-CD (justifica las 16 pasadas)

  # estudios auxiliares (todos usan sweep.py)
  families.py        # curvas gamma/lambda (Fig. 1) -> families.csv
  sensitivity.py     # rejilla kappa_w x kappa_p (Fig. 5) -> qs_grid.csv
  seeds.py           # robustez de semilla (Tabla III) -> seeds.csv
  stats_full.py      # estadísticas de deadlines (Tabla IV) -> stats_full.csv
  ablation.py        # orden vs magnitud + rejilla gamma x lambda -> ablation*.csv
  counterfactual.py  # variantes del análisis (Tabla del contrafactual)

  sweep.py           # convención única de barrido (ver §2)

  *.csv              # salidas generadas; NO editar a mano
  edf-wad-15/  edf-wad-25/          # resultados del runner principal
  rta_cd-15/  rta_cd-25/            # RTA-CD: schedulables, times, deadlines, diagnostics
  hospa-15/  hospa-25/              # HOSPA + PD
  wad-15/25  wad_adapt-15/25  wad_budget-15/25/   # volcados de forma

  paper/
    main.tex main.pdf make_figures.py figures/

  _archive/                         # no se usa para compilar; trazabilidad
    legacy_gdpa/                    # gdpa_seed.py y gdpa-seed42/7/ (GDPA fuera)
    g0_ablation/                    # wad_adapt_g0* (evidencia de §VII-C)
    paper_backups/                  # main.tex.bak-* (pre-reescritura y pre-fluidez)
    paper_briefs/                   # briefs de reescritura y fluidez
    paper_docs/                     # CHANGES, FLUENCY_STAGE*, informes antiguos
```

Código del método:
```
code/assignment/wad_assignment.py        # WAD (una pasada)
code/assignment/wad_adapt_assignment.py  # WAD-Adapt (3 pasadas; q, s de semilla)
code/assignment/rta_cd_assignment.py     # RTA-CD (16 pasadas deterministas)
```

El remoto (`ssh despacho`, VPN UC con `uni-vpn`) es `ctrpc18.ctr.unican.es`, un
Mac de 8 núcleos. Los runs de la campaña están en `~/edf-runs/` (logs y scripts).

---

## 2. Banco de pruebas y convención de evaluación

- Generador: `examples/generator.py` + `workspace/framework_paper/systems.py`
  (`get_systems`). Periodos log-uniforme [100,1000]; deadline del flujo uniforme
  en `[0.5·n_f·T_f, n_f·T_f]`; utilidades por **UUNIFAST**; mapeo **balanced**
  (round-robin por utilidad decreciente).
- **EDF-15** = 5 flows × 3 tasks en 3 procesadores; **EDF-25** = 5×5×5.
- **50 sistemas × 20 niveles de utilización en [0.5,0.9] = 1000 instancias** por
  tamaño, **seed 42**.
- **Métrica**: nº de sistemas que el análisis certifica como planificables. Juez:
  `analysis/holistic_local_edf_analysis.py`. Es un test **suficiente
  (pesimista)**: lo que certifica es planificable de verdad.
- **Entorno**: `code/.venv` (Python 3.13.15, numpy 2.5.2, pandas 3.0.5,
  matplotlib 3.11.1). Ejecutar desde `code/`.

### La convención (importante)

Todos los scripts usan **`sweep.levels`**: se genera **una** población y se llama
a `set_utilization` **en el sitio** en cada nivel, reutilizando los mismos objetos.
No hagas `deepcopy` del base y escales una sola vez: matemáticamente es el mismo
sistema, pero los WCET difieren ~5·10⁻¹⁶ y el análisis holístico es discontinuo
(`ceil`/`floor`, ψ dependiente del jitter). Esa diferencia cambia 4 de 1000
veredictos (p. ej. PD 706 frente a 710). **No es ruido de coma flotante**: en una
instancia la respuesta de un flujo pasaba de 279 a 681.

Si añades un script nuevo, usa `sweep.levels` o los números no cuadrarán con las
tablas.

---

## 3. Cómo reproducir (desde `code/`)

```bash
.venv/bin/python workspace/edf_wad/edf_wad.py 15 25      # Tablas I y II
.venv/bin/python workspace/edf_wad/hospa.py 15 25        # HOSPA (≈2 h)
.venv/bin/python workspace/edf_wad/dump_deadlines.py rta_cd 15   # 964, 2,3565 s
.venv/bin/python workspace/edf_wad/dump_deadlines.py rta_cd 25   # 883, 20,4336 s
.venv/bin/python workspace/edf_wad/families.py           # Fig. 1
.venv/bin/python workspace/edf_wad/sensitivity.py        # Fig. 5
.venv/bin/python workspace/edf_wad/seeds.py              # Tabla III
.venv/bin/python workspace/edf_wad/stats_full.py         # Tabla IV
.venv/bin/python workspace/edf_wad/ablation.py           # §VII-A
.venv/bin/python workspace/edf_wad/counterfactual.py     # Tabla del contrafactual
.venv/bin/python workspace/edf_wad/flatten_check.py wad_adapt-15 wad-15 ...
.venv/bin/python workspace/edf_wad/paper/make_figures.py # regenera figures/
cd workspace/edf_wad/paper && latexmk -pdf main.tex      # compila (8 páginas)
```

`make_figures.py` lee: `edf-wad-{15,25}` (pd, wad, wad_adapt, wad_budget, eqf,
eqs), `hospa-{15,25}` (hospa) y `rta_cd-{15,25}` (rta_cd). Ya no lee GDPA.

---

## 4. Números clave (estado actual)

**Tabla I (EDF-15 /1000)**: PD 706 · EQF 783 · EQS 734 · HOSPA 840 · WAD 875 ·
WAD-Adapt 928 · WAD-Budget 776 · RTA-CD 964.
Tiempos medios (s/inst): 0.0116 / 0.0050 / 0.0056 / 3.8487 / 0.0075 / 0.0496 /
0.1494 / 2.3565.

**Tabla II (EDF-25 /1000)**: PD 438 · EQF 656 · EQS 552 · HOSPA 592 · WAD 718 ·
WAD-Adapt 798 · WAD-Budget 478 · RTA-CD 883.
Tiempos: 0.0181 / 0.0124 / 0.0116 / 32.4682 / 0.0191 / 0.7332 / 2.6427 / 20.4336.

**RTA-CD**: 16 pasadas deterministas + guardia de 1200 s que no se alcanza.
782/1000 y 654/1000 instancias ya salen en el arranque analítico; solo 1 por
tamaño llega al tope (y es no planificable); 0 timeouts. Cobertura de WAD-Adapt:
96,3 % y 90,4 %; coste 47,5× y 27,9×. Pareado: RTA-CD añade 38/86 y pierde 2/1.

**Tabla III (semillas 1/7/42/123)**: PD 710/767/706/726 · WAD 863/853/875/867 ·
WAD-Adapt 926/913/928/921 · WAD-Budget 764/767/776/760.
Rangos: WAD 853–875; WAD-Adapt 913–928.

**Fig. 1** (familias EDF-15): γ 706, 779, 806, 814, 810, 815, 805, 786, 768, 757,
745; λ 460, 611, 706, 787, 808, 792, 774. El punto γ=0 / λ=1 es PD = 706.

**Fig. 5** (κ_w×κ_p): máximo **875 en (1.6,0.4)**, que es el WAD de la Tabla I
(antes había un empate en 876).

**Contrafactual** (full): 460/706/808/794/775/WAD 875 para λ=0.5/1/2/3/5;
`cutoff_only`: 195/386/572/652/701/644; `no_cap`: 0.

**Ablación** (§VII-A): identidad 875; **19/1000** instancias mejoran con algún
shuffle; **11.204/20.000** empeoran; pico de la rejilla γ×λ: 862 en (0.2,2.0).

**Tabla IV (deadline stats)**: mín d_i/C_i = 2.12 (WAD) / **1.08** (WAD-Adapt) /
0.96 (WAD-Budget); media Σd/D = 2.2 / 3.95 / 1.0 (EDF-15), 2.6 / 4.49 / 1.0 (EDF-25).

**Tabla de forma (γ, λ; mediana)**: WAD 0.27/1.67 (15) y 0.24/1.75 (25);
WAD-Adapt 0.27/3.15 y 0.24/3.25; WAD-Budget 0.27/0.76 y 0.24/0.67; RTA-CD
0.00/1.85 y 0.00/2.62.

**Ablación γ=0**: WAD-Adapt con Q=1.6 y rampa apagada 869/622; con el gasto
inicial de WAD (Q=2.2/2.6) 882/698. Control a igual gasto por flujo: WAD
875→807, WAD-Adapt 928→779, WAD-25 718→645, WAD-Adapt-25 798→644.

---

## 5. Citas (claves y `bibitem`)

- **rivas15** — Rivas et al., *Deadline assignment in EDF schedulers…*, IEEE TPDS
  26(10):2671–2684, 2015 (tu paper). Fenómeno + LC-EDF-DS (=λ) + LC-EDF-GSD (=γ=1).
- **hospa** — Rivas et al., *Schedulability analysis and optimization of
  heterogeneous EDF and FP distributed real-time systems*, ECRTS 2011, 195–204.
  HOSPA + el análisis EDF/FP. Es la cita del **baseline** y del **análisis** (§II).
- **hopa** — Gutiérrez & González Harbour, WPDRTS 1995, 124–132. HOPA (prioridades
  fijas).
- Otras: dinatale (PD/EQF/EQS), holistic (Tindell), palencia (offsets), spuri,
  lehozcky, audsley, davis, liu, uunifast.
- **`gdpaframework` ya no existe** en la bibliografía ni en el texto. No lo
  reintroduzcas: el paper es autocontenido.

---

## 6. Informes

- `RESULTS_2026-10-18.md` — campaña de resultados (RTA-CD, HOSPA, volcados),
  ablación de forma y **§9 con la convención unificada** y los valores vigentes.
  Es la referencia de números.
- `_archive/paper_docs/` — histórico, no necesario para compilar:
  `CHANGES_2026-10-18.md` (cambios aplicados), `FLUENCY_STAGE1/2.md`,
  `VERIFICATION.md`, `READINESS.md`, `WRITING_REVIEW.md`, `TITLES.md` y
  `VERIFICATION_BRIEF.md`. Ojo: los informes antiguos citan cifras anteriores a
  la campaña (958, GDPA, 1.09); no los uses como fuente de números.
- `_archive/paper_briefs/` — briefs de la reescritura y de la fluidez.

---

## 7. Estado / pendiente

**Hecho**:

- Contribuciones + problema formal; reencuadre "suficiente"; mecanismo (canal, no
  dueño del óptimo); apéndice unificado; counterfactual con tabla.
- Estudio con datos del repo; stats completas; RTA-CD con pseudocódigo;
  reproducibilidad; robustez de semilla; `rivas15` + `hospa` + `hopa` integrados.
- **GDPA eliminado por completo**; benchmark autocontenido; RTA-CD como C3.
- **RTA-CD determinista** (16 pasadas) y corrido en **EDF-25** (883).
- **HOSPA re-corrido** en ambos tamaños (840/592), autocontenido.
- **Figuras 2–5 vectoriales** (PDF), sin GDPA y con RTA-CD en ambos tamaños.
- **Ablación de forma** (γ=0 + control plano a igual gasto) en §VII.
- **Convención unificada** en todos los scripts (`sweep.levels`); el 710 y el 813
  desaparecen. Paper en 8 páginas, sin referencias sin resolver.

**Pendiente / opcional**:

- Una lectura final de escritura (el paper se simplificó mucho; conviene releerlo
  entero una vez).
- Decidir si se menciona el guardia de 2 s de WAD-Adapt (hoy no se menciona; los
  conteos no cambian, solo la tercera cifra decimal de algunos deadlines).
- No hace falta anonimizar.

---

## 8. Trampas / notas

- **Convención**: usa `sweep.levels`. Si un script escala desde el base una sola
  vez, sus números no cuadrarán con las tablas (706 vs 710).
- **RTA-CD**: `MAX_SWEEPS=16`, `SAFETY_SECONDS=1200`. Es determinista en la
  práctica (0 timeouts). `edf_wad.py` solo lo corre en tamaño 15; para el 25 usa
  `dump_deadlines.py rta_cd 25`.
- **WAD-Adapt** no es bit-determinista en los deadlines: su análisis lleva
  `max_time=2.0 s`; si expira, ese flujo queda sin corregir. Los conteos salen
  iguales, pero el mínimo d_i/C_i puede variar en la tercera cifra.
- **RTA-CD puede asignar d_i < C_i** en ~0,2 % de las tareas (24/15000 y
  47/25000), con mínimo ≈0 en el benchmark completo. Está divulgado en §VIII.
- **No reutilices** los `.xlsx` de `workspace/framework_paper/edf/` (GDPA/HOSPA
  del paper compañero): el paper ya no los usa.
- **`_archive/paper_backups/main.tex.bak-2026-10-18`** es la copia previa a la reescritura; no la borres
  hasta que el paper esté cerrado.
- Los `.csv/.xlsx` son **salidas**; se regeneran con los scripts de §3.
