#!/bin/bash
# Genera report/results.pdf a partir de results/. No corre el algoritmo.
# La suite, el orden y la agrupacion salen de report/benchmarks.py.
set -e
cd "$(dirname "$0")"

PY=./.venv/bin/python3
if [ ! -x "$PY" ]; then
    echo ">>> Creando .venv (pandas, numpy, matplotlib, scipy, sympy)"
    python3 -m venv .venv
    ./.venv/bin/pip install -q pandas numpy matplotlib scipy sympy
fi

echo ">>> 1/4 Consolidando resultados"
$PY scripts/consolidate_runs.py

echo ">>> 2/4 Figuras"
$PY scripts/plot_solutions.py
$PY scripts/plot_pareto.py
$PY scripts/plot_extra_report_figures.py

echo ">>> 3/4 Simplificacion simbolica, tablas y documento"
$PY report/simplify_expressions.py
$PY report/extrapolation.py
$PY report/physics_checks.py
$PY report/generate_report.py
$PY report/generate_formulas_table.py

echo ">>> 4/4 Compilando LaTeX"
cd report
for i in 1 2 3; do
    pdflatex -interaction=nonstopmode -halt-on-error results.tex > results.build.log 2>&1 || {
        echo "[ERROR] pdflatex fallo; ver report/results.build.log"; tail -30 results.build.log; exit 1; }
done
echo ">>> Listo: report/results.pdf"
