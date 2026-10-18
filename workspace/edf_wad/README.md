# edf_wad — mapa del directorio

Paper: `paper/main.tex` (8 páginas). Estado y traspaso: `HANDOFF.md`.
Campaña de resultados y convención: `RESULTS_2026-10-18.md`.

## Scripts y qué produce cada uno

| script | salida en el paper |
|---|---|
| `edf_wad.py` | Tablas I y II (familia WAD, PD/EQF/EQS) |
| `hospa.py` | fila HOSPA de las Tablas I y II |
| `dump_deadlines.py` | RTA-CD de las Tablas I y II, y volcados de deadlines |
| `families.py` | Fig. 1 |
| `sensitivity.py` | Fig. 5 |
| `seeds.py` | Tabla III |
| `stats_full.py` | Tabla IV |
| `ablation.py` | §VII-A (orden vs magnitud, rejilla γ×λ) |
| `counterfactual.py` | tabla del contrafactual |
| `deadline_shape.py` | tabla de forma (γ, λ) a partir de los volcados |
| `flatten_check.py` | control plano de la ablación de forma |
| `probe_rta_cd.py` | sondeo que justifica las 16 pasadas de RTA-CD |
| `sweep.py` | convención de barrido; lo usan todos los anteriores |

## Datos

- `edf-wad-15/`, `edf-wad-25/`: runner principal. Los `.xlsx` son los que lee
  `make_figures.py`.
- `rta_cd-15/`, `rta_cd-25/`: conteos, tiempos, deadlines y diagnóstico de RTA-CD.
- `hospa-15/`, `hospa-25/`: HOSPA + PD. Los `.xlsx` los lee `make_figures.py`.
- `wad-15/25/`, `wad_adapt-15/25/`, `wad_budget-15/25/`: volcados para la tabla
  de forma y el control plano.
- `*.csv`: salidas de los estudios. No editar a mano.
- `_archive/`: legado que ya no usa el paper (GDPA, ablación γ=0, copias de
  `main.tex`, briefs e informes antiguos). Se conserva por trazabilidad, no hace
  falta para compilar.

## Convención

Todos los scripts usan `sweep.levels`: **una** población reutilizada y
`set_utilization` en el sitio en cada nivel. No hagas `deepcopy` del base y
escales una sola vez: los WCET difieren ~5·10⁻¹⁶ y el análisis es discontinuo,
así que cambian 4 de 1000 veredictos (PD 706 frente a 710). Ver `HANDOFF.md` §2.
