#!/usr/bin/env python3
"""
consolidate_runs.py — Arma results/all_runs_summary.csv para el reporte.

Fuentes posibles (se usa la MAS RECIENTE):
  * Corrida multi-semilla (--runs N > 1): results/run_<k>/comparison_summary.csv
  * Corrida unica (--runs 1, con o sin --only): results/comparison_summary.csv
    (el binario fusiona por EDP, asi que re-correr un problema con --only
    reemplaza solo sus filas).

Antes se consolidaba siempre desde run_*, aunque esos directorios fueran de
una corrida vieja; el reporte salia con resultados obsoletos sin avisar.
"""
import glob
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_CSV = os.path.join(RESULTS_DIR, "all_runs_summary.csv")

COLUMNS = [
    "run", "method", "pde", "pareto_size",
    "best_mse_domain", "best_mse_boundary",
    "mean_mse_domain", "mean_mse_boundary",
    "hypervolume", "runtime_s", "solution_mse",
]


def multi_run_sources():
    dirs = glob.glob(os.path.join(RESULTS_DIR, "run_*"))
    dirs = [d for d in dirs if os.path.basename(d).replace("run_", "").isdigit()]
    out = []
    for d in sorted(dirs, key=lambda p: int(os.path.basename(p).replace("run_", ""))):
        csv = os.path.join(d, "comparison_summary.csv")
        if os.path.exists(csv):
            out.append((int(os.path.basename(d).replace("run_", "")), csv))
    return out


def read_summary(path, run_id):
    df = pd.read_csv(path)
    df.insert(0, "run", run_id)
    return df


def consolidate():
    multi = multi_run_sources()
    single = os.path.join(RESULTS_DIR, "comparison_summary.csv")

    t_multi = max((os.path.getmtime(p) for _, p in multi), default=-1)
    t_single = os.path.getmtime(single) if os.path.exists(single) else -1

    if t_multi < 0 and t_single < 0:
        print("[WARN] No hay resumenes en results/: nada que consolidar.")
        return

    if t_single >= t_multi:
        print(f"[INFO] Fuente: corrida unica ({os.path.relpath(single, BASE_DIR)})")
        if multi:
            print(f"       Se ignoran {len(multi)} directorios run_* mas antiguos.")
        df = read_summary(single, 0)
    else:
        print(f"[INFO] Fuente: {len(multi)} corridas independientes (results/run_*)")
        df = pd.concat([read_summary(p, r) for r, p in multi], ignore_index=True)

    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLUMNS]
    for c in COLUMNS[3:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # solution_mse = -1 es el centinela de "sin campeon".
    df.loc[df["solution_mse"] < 0, "solution_mse"] = np.nan

    df.to_csv(OUT_CSV, index=False)
    n_runs = df["run"].nunique()
    n_pdes = df[df["method"] == "PISR-NSGA-II"]["pde"].nunique()
    print(f"[OK] {len(df)} filas | {n_runs} corrida(s) | {n_pdes} EDPs -> {os.path.relpath(OUT_CSV, BASE_DIR)}")


if __name__ == "__main__":
    consolidate()
