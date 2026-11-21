#!/usr/bin/env bash
# Competidores de PISR-NSGA-II sobre la misma suite y la misma metrica:
#   1. PINN (PyTorch)        : precision, sin formula.
#   2. PINN -> PySR          : formula interpretable sin datos (estilo SymTorch).
#
# Uso:  ./run_competitors.sh [--runs 30] [--jobs N] [--only EDP] [--test]
#   --runs N   semillas 0..N-1, escritas en results/run_<k>/ (como --runs del binario).
#              Con --runs 1 escribe en results/ directamente.
#   --jobs N   trabajos en paralelo (por defecto nproc/2). Cada uno usa 1 hilo.
#   --test     pocas iteraciones: solo prueba la tuberia (no usar para el reporte).
# Es reanudable: lo ya calculado se omite, asi que relanzarlo continua donde quedo.
set -euo pipefail
cd "$(dirname "$0")"

RUNS=1; JOBS=$(( $(nproc) / 2 )); ONLY=""; TEST=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift ;;
        --jobs) JOBS="$2"; shift ;;
        --only) ONLY="$2"; shift ;;
        --test) TEST="--test" ;;
        *) echo "opcion desconocida: $1"; exit 1 ;;
    esac
    shift
done
[[ "$JOBS" -lt 1 ]] && JOBS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHON_JULIACALL_THREADS=1 PYTHON_JULIACALL_HANDLE_SIGNALS=yes

echo ">>> 0/3 Entorno (.venv-competitors: torch CPU, PySR + Julia)"
PY=./.venv-competitors/bin/python3
if [[ ! -x "$PY" ]]; then
    python3 -m venv .venv-competitors
    ./.venv-competitors/bin/pip install -q --upgrade pip
    ./.venv-competitors/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch
    ./.venv-competitors/bin/pip install -q numpy pandas sympy pysr
fi
# Primera vez: PySR descarga Julia y precompila SymbolicRegression (varios minutos).
"$PY" -c "import juliacall, pysr" >/dev/null 2>&1 || { echo "[ERROR] no se pudo instalar PySR/Julia"; exit 1; }

echo ">>> 1/3 Referencia exportada por el binario (misma grilla y referencia que PISR)"
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build_release -j"$(nproc)" --target PISR-NSGA-II >/dev/null
./build_release/PISR-NSGA-II --export-reference results/reference ${ONLY:+--only "$ONLY"}

if [[ -n "$ONLY" ]]; then PROBS="$ONLY"; else PROBS=$(ls results/reference/*.json | xargs -n1 basename | sed 's/\.json$//'); fi
if [[ "$RUNS" -gt 1 ]]; then SEEDS=$(seq 0 $((RUNS - 1))); else SEEDS="-1"; fi
mkdir -p results/competitors_logs

echo ">>> 2/3 PINN: $(echo $PROBS | wc -w) problemas x $(echo $SEEDS | wc -w) semillas, $JOBS en paralelo"
for s in $SEEDS; do for p in $PROBS; do echo "$s $p"; done; done |
    xargs -P "$JOBS" -L 1 bash -c '"$0" competitors/pinn.py --run "$1" --problem "$2" '"$TEST"' \
        > "results/competitors_logs/pinn_$2_run$1.log" 2>&1 \
        && tail -1 "results/competitors_logs/pinn_$2_run$1.log" \
        || echo "[PINN] $2 run $1: FALLO (ver results/competitors_logs/pinn_$2_run$1.log)"' "$PY"

echo ">>> 3/3 PINN -> PySR, $JOBS procesos"
RUNARG=""; [[ "$RUNS" -gt 1 ]] && RUNARG="--runs $RUNS"
for i in $(seq 0 $((JOBS - 1))); do
    "$PY" competitors/pinn_pysr.py --shard "$i" --nshards "$JOBS" $RUNARG ${ONLY:+--only "$ONLY"} $TEST \
        > "results/competitors_logs/pysr_shard$i.log" 2>&1 &
done
wait
cat results/competitors_logs/pysr_shard*.log | grep "^\[PINN+PySR\] .* run " || true
rm -rf results/pysr_tmp
echo "COMPETIDORES OK"
