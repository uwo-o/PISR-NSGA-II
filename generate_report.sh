#!/bin/bash
# Genera report/results.pdf a partir de results/. No corre el algoritmo.
# La suite, el orden y la agrupacion salen de report/benchmarks.py.
set -e
cd "$(dirname "$0")"

PY=./.venv/bin/python3
if ! "$PY" -c "import pandas, numpy, matplotlib, scipy, sympy" >/dev/null 2>&1; then
    echo ">>> Creando .venv (pandas, numpy, matplotlib, scipy, sympy)"
    rm -rf .venv
    if ! python3 -m venv .venv >/dev/null 2>&1; then
        # Sin python3-venv (falta ensurepip) y sin sudo: venv sin pip + get-pip.py
        rm -rf .venv
        python3 -m venv --without-pip .venv
        GETPIP=$(mktemp)
        curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$GETPIP"
        "$PY" "$GETPIP" -q
        rm -f "$GETPIP"
    fi
    "$PY" -m pip install -q pandas numpy matplotlib scipy sympy
fi

echo ">>> 1/4 Consolidando resultados"
$PY scripts/consolidate_runs.py

echo ">>> 2/4 Figuras"
$PY scripts/plot_solutions.py
$PY scripts/plot_pareto.py
$PY scripts/plot_extra_report_figures.py

echo ">>> 3/4 Simplificacion simbolica, tablas y documento"
$PY report/simplify_expressions.py
# Red del PINN evaluada fuera de [0,1] (necesita PyTorch: .venv-competitors de run_competitors.sh)
if ./.venv-competitors/bin/python3 -c "import torch" >/dev/null 2>&1; then
    ./.venv-competitors/bin/python3 competitors/pinn_extrapolate.py
fi
$PY report/extrapolation.py
$PY report/physics_checks.py
$PY report/interpretability.py
$PY report/pysr_front_analysis.py
$PY report/generate_report.py
$PY report/generate_formulas_table.py

if ! command -v pdflatex >/dev/null 2>&1; then
    echo ">>> 4/4 pdflatex no esta instalado: se omite la compilacion."
    echo "    Tablas, figuras y report/results.tex quedaron listos; compilar en otra maquina con:"
    echo "    cd report && pdflatex results.tex (3 veces)"
    exit 0
fi
echo ">>> 4/4 Compilando LaTeX"
cd report
for i in 1 2 3; do
    pdflatex -interaction=nonstopmode -halt-on-error results.tex > results.build.log 2>&1 || {
        echo "[ERROR] pdflatex fallo; ver report/results.build.log"; tail -30 results.build.log; exit 1; }
done
echo ">>> Listo: report/results.pdf"
