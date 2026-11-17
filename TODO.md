# TODO (correcciones del revisor, rama `grade`)

## Hecho (sin commit)
- Referencia fuera del bucle: campeón y parada temprana por `validation_score` (residuo + frontera en malla de validación + chequeo del ansatz en el borde); `ref_err` solo se registra.
- Problemas bien planteados: Lane-Emden como IVP (ansatz u0 + u0' x + x² N), Thomas-Fermi como BVP en [0,1], Schrödinger k²=4 (no resonante), Airy verdadero (Ai), GL tanh(x/√2), NS con residuo real de orden 4 y ψ, ∂ψ/∂n en el borde.
- Aritmética compleja: `project_real` en campos reales; gradiente por diferencias centradas (`constant_gradient`) en vez de paso complejo.
- Priors: sesgan la inicialización y ya no excluyen (se quitaron anti-plano, max_grad, varianza, espejo y separabilidad como restricciones).
- Eliminación de duplicados en `nsga2_select_next`. Sin anti-plano, el clon N=0 llenaba el frente 1 y congelaba la búsqueda (Poisson se estancaba en ref_err 1e-3; ahora 1e-7).
- Tests: `oracle_test` (21/21) y `gradient_test`; `run_tests.sh`. `main` corre el oráculo antes de cada problema y lo omite si reprueba (`--skip-oracle` para forzar).
- Solver IVP: el arranque de Taylor usa secante (el punto fijo divergía). Verificado contra la serie de Lane-Emden (err 1e-13).
- `run_once` usa `benchmark_suite()`; `thesis_eq.cpp` usa `eval_solution`; `generate_formulas_table.py` tiene las exactas nuevas.
- Documentos temporales borrados.
- Representacion hibrida (N con ansatz y U directo compiten; `--no-hybrid` para ablacion). Oraculo chequea el arbol exacto como U directo.
- Prior `modos_exp` (operador lineal homogeneo y autonomo -> semillas e^{lambda r}, lambda complejo), sondeado con jets genericos.
- U definida exactamente en la frontera (0*N = 0 en el ansatz; U directo debe ser finita en el borde).
- Bugs complejos: simplify() trataba i como 0 (exp(i a x) -> 1), division por imaginario puro daba NaN, el LaTeX omitia la parte imaginaria de las constantes.

- Frontera de U directo exigida como limite lateral (extrapolacion cubica a 1e-3 y 1e-5): cierra la trampa del pico exp(-(x/eps)^2) que dio Thomas-Fermi 1D con error 0.98.
- `modos_exp` despeja la ultima componente de lambda de P(lambda)=0 (simbolo del operador): ~60% de las semillas son solucion exacta de la EDP (Schrodinger-Unsteady 2D quedaba atascado en u real).

- Polinomios ortogonales (P/H/T/L) evaluados en Re(arg): L_2(x/i) daba L_2(0)=1 en el valor y otra cosa en las derivadas. Ahora la recurrencia es compleja (Lane-Emden 1D re-corrido con ese bug: invalido).
- main.cpp: comparison_summary.csv y all_runs_summary.csv se FUSIONAN por EDP (antes --only borraba el resto) y se guardan en notacion cientifica (antes 1e-30 -> 0.0000000000).
- Reporte: report/benchmarks.py es el catalogo unico (orden, familias, ecuaciones, referencias); todos los scripts lo usan. Lineas base externas anteriores a 2026-10-20 se excluyen solas (resolvian otro problema). generate_report.sh crea .venv si falta.
- results/comparison_summary.csv actual fue RECONSTRUIDO desde los logs (sin MSE de frontera ni hipervolumen); RK4/FDM recalculado. Troesch = corrida --no-hybrid.

- Simplificacion simbolica (report/simplify_expressions.py, SymPy): arma U final (ansatz incluido, Coons en 2D), Re() en campos reales, varias estrategias, constantes limpias; cada forma se VERIFICA contra la grilla del campeon (dif. rel. <= 1e-3). El binario ahora exporta expr_*.sym (arbol exacto).
- Printer LaTeX: A - (B + C) se imprimia sin parentesis (formulas publicadas incorrectas); constantes fisicas se imprimian vacias (tambien en print(), que usa simplify() para comparar subarboles); producto de dos constantes sin \cdot.

- Extrapolacion fuera de [0,1] (report/extrapolation.py): campeon tal como se encontro vs polinomio de Chebyshev con la MISMA precision dentro del dominio. Seccion propia en el reporte.
- Navier-Stokes 2D recalibrado: Kovasznay estandar (amplitud 1/2pi, Re=20). Antes psi ~ y daba MSE 1e-7 (casi trivial). Oraculo: chequeo de DF por convergencia O(h^2) en vez de umbral absoluto.
- --no-hybrid escribe en results/ablation_no_hybrid/; el reporte muestra la tabla hibrido vs solo ansatz. Resultados viejos de NS 2D apartados en results/stale_navier_stokes_re100/.

- Fidelidad fisica: nombres honestos (Duffing-type BVP, Fisher-KPP steady, Thomas-Fermi-type 2D), notas bajo la suite (equivalencia de Painleve verificada: u=6^{3/5}w, x=6^{1/5}z; NS con no linealidad degenerada).
- Consistencia fisica (report/physics_checks.py): cantidades fisicas extraidas de cada formula vs valor conocido (energia, pendientes, k, dispersion, decaimiento, radio de Lane-Emden).

## Pendiente
- [ ] (Trabajo futuro) Navier-Stokes con no linealidad real (solucion manufacturada).
- [ ] Re-correr con el binario nuevo: Navier-Stokes_2D (nueva definicion) y Troesch_1D (hibrido; sus archivos eran de --no-hybrid y se movieron a la ablacion).
- [ ] Re-correr con el binario nuevo: Thomas-Fermi_1D, Lane-Emden_1D (misma trampa del pico, entro al hall of fame en gen 126), Schrodinger-Unsteady_2D, Troesch_1D (regresion: 2e-2 -> 0.36, campeon U directo que no cumple u(1)=1; probar tambien con --no-hybrid).
- [ ] Troesch: hibrido 1.7e-2 vs --no-hybrid 3.3e-5. Decidir: arreglar la competencia raw/ansatz o desactivar raw cuando hay ansatz exacto.
- [ ] Re-corridas aun malas con binario de 18:44: Thomas-Fermi_1D (2.4e-2), Schrodinger-Unsteady_2D (1.0, el prior de dispersion no basto).
- [ ] Lineas base RK4/FDM: Navier-Stokes_2D da NaN a 50 nodos.
- [ ] Regenerar DeepXDE/PySR/PySINDy con los problemas actuales.
- [ ] Navier-Stokes-Unsteady 2D sigue mal (err 0.27; antes 1.0): revisar.
- [ ] Re-correr TODOS (los resultados del 20-oct tambien quedan obsoletos: bug de i en simplify) los resultados en `results/` (los actuales son anteriores a las correcciones) y luego `generate_report.sh`.
- [ ] Ablaciones del revisor (20 semillas, Wilcoxon + Holm, métricas SymPy).
- [ ] (Opcional) agregar al oráculo la comparación de Lane-Emden con su serie de potencias.
- Respaldo pre-refactor: /tmp/claude-1000/-home-uwo-Projects-PISR-NSGA-II/e03e5f97-85c1-4683-9143-0add412f0abb/scratchpad/backup_pre_refactor_20261020_173033
