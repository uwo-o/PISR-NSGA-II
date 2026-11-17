#!/usr/bin/env bash
# Correr ANTES de cada experimento: compila y ejecuta oraculo + gradiente.
set -euo pipefail
cd "$(dirname "$0")"
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build_release -j"$(nproc)"
./build_release/gradient_test
./build_release/oracle_test
echo "TESTS OK"
