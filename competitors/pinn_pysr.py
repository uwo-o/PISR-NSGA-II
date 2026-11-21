#!/usr/bin/env python3
"""PINN -> PySR (destilacion simbolica, estilo SymTorch): el competidor
interpretable sin datos.

Toma la prediccion del PINN en los puntos de validacion (<EDP>_pinn_pred.csv,
nunca la referencia) y le ajusta una formula con PySR. La formula se mide
contra la referencia en los mismos puntos que solution_mse de PISR-NSGA-II.
Campos complejos: una formula para la parte real y otra para la imaginaria.

Procesa todos los pares (semilla, EDP) que tengan prediccion del PINN y aun no
tengan resultado; --shard i --nshards K reparte esa lista entre K procesos.
Cada proceso arranca Julia una sola vez.

Salidas, junto a la prediccion del PINN:
    <EDP>_pinn_pysr.csv   metricas (method=PINN+PySR), complejidad y formula
"""
# juliacall debe importarse antes que torch (aqui no se usa torch).
import juliacall  # noqa: F401
import argparse, glob, os, re, sys, time

import numpy as np
import pandas as pd
import sympy as sp
from pysr import PySRRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def jobs(results, runs, only):
    out = []
    for f in sorted(glob.glob(os.path.join(results, "**", "*_pinn_pred.csv"), recursive=True)):
        d = os.path.dirname(f)
        lbl = os.path.basename(f)[:-len("_pinn_pred.csv")]
        if only and lbl != only:
            continue
        m = re.search(r"run_(\d+)$", d)
        run = int(m.group(1)) if m else -1
        if runs is not None and run not in runs:
            continue
        if os.path.exists(os.path.join(d, f"{lbl}_pinn_pysr.csv")):
            continue
        out.append((run, lbl, d))
    return out


def fit(X, y, names, seed, a, tmp):
    model = PySRRegressor(
        niterations=a.niterations,
        populations=a.populations,
        maxsize=a.maxsize,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "tanh"],
        model_selection="best",
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        timeout_in_seconds=a.timeout,
        progress=False,
        verbosity=0,
        output_directory=tmp,
    )
    model.fit(X, y, variable_names=names)
    best = model.get_best()
    front = model.equations_[["complexity", "loss", "equation"]].copy()
    return model, sp.sympify(model.sympy()), int(best["complexity"]), front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--ref-dir", default=os.path.join(ROOT, "results", "reference"))
    ap.add_argument("--runs", type=int, default=None, help="solo semillas 0..N-1 (por defecto todas las halladas)")
    ap.add_argument("--only", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--niterations", type=int, default=100)
    ap.add_argument("--populations", type=int, default=15)
    ap.add_argument("--maxsize", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=900, help="segundos por ajuste")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        a.niterations, a.timeout = 5, 60

    runs = set(range(a.runs)) if a.runs is not None else None
    todo = jobs(a.results, runs, a.only)[a.shard::a.nshards]
    print(f"[PINN+PySR] shard {a.shard}/{a.nshards}: {len(todo)} ajustes", flush=True)
    tmp = os.path.join(a.results, "pysr_tmp", f"shard{a.shard}")
    os.makedirs(tmp, exist_ok=True)

    for run, lbl, d in todo:
        pred = pd.read_csv(os.path.join(d, f"{lbl}_pinn_pred.csv"))
        ref = pd.read_csv(os.path.join(a.ref_dir, f"{lbl}.csv"))
        ref = ref[ref.kind == "dom"].reset_index(drop=True)
        names = [c for c in ["x", "y", "t"] if c in pred.columns]
        X = pred[names].to_numpy()
        cplx = bool(np.any(pred["u_im"].to_numpy() != 0.0)) or bool(np.any(ref["u_im"].to_numpy() != 0.0))
        seed = 1000 + max(run, 0)
        t0 = time.time()
        try:
            m_re, e_re, c_re, f_re = fit(X, pred["u_re"].to_numpy(), names, seed, a, tmp)
            u = m_re.predict(X).astype(complex)
            expr, comp = e_re, c_re
            f_re.insert(0, "part", "re")
            fronts = [f_re]
            if cplx:
                m_im, e_im, c_im, f_im = fit(X, pred["u_im"].to_numpy(), names, seed + 7, a, tmp)
                u = u + 1j * m_im.predict(X)
                expr, comp = e_re + sp.I * e_im, c_re + c_im
                f_im.insert(0, "part", "im")
                fronts.append(f_im)
            # Frente completo de PySR (complejidad, perdida vs PINN, formula): para
            # analizar si la forma simple aparece aunque "best" elija otra.
            pd.concat(fronts).to_csv(os.path.join(d, f"{lbl}_pinn_pysr_front.csv"), index=False)
            ok = True
        except Exception as ex:  # un ajuste fallido no detiene el resto
            print(f"[PINN+PySR] {lbl} run {run}: fallo ({ex})", flush=True)
            u, expr, comp, ok = np.full(len(X), np.nan, dtype=complex), sp.nan, -1, False
        rt_pysr = time.time() - t0
        # Tiempo total del competidor = entrenamiento del PINN + destilacion.
        pinn_csv = os.path.join(d, f"{lbl}_pinn.csv")
        rt = rt_pysr + (float(pd.read_csv(pinn_csv)["runtime_s"].iloc[0]) if os.path.exists(pinn_csv) else 0.0)
        r = ref["u_re"].to_numpy() + 1j * ref["u_im"].to_numpy()
        p = pred["u_re"].to_numpy() + 1j * pred["u_im"].to_numpy()
        sol = float(np.mean(np.abs(u - r) ** 2))
        fit_mse = float(np.mean(np.abs(u - p) ** 2))
        pd.DataFrame([{"method": "PINN+PySR", "pde": lbl, "seed": seed, "solution_mse": sol,
                       "fit_mse_vs_pinn": fit_mse, "complexity": comp, "runtime_s": rt,
                       "pysr_runtime_s": rt_pysr, "ok": ok,
                       "expr": str(expr), "expr_srepr": sp.srepr(expr)}]).to_csv(
            os.path.join(d, f"{lbl}_pinn_pysr.csv"), index=False)
        print(f"[PINN+PySR] {lbl} run {run}: solution_mse={sol:.3e} complejidad={comp} "
              f"t={rt_pysr:.0f}s  u = {expr}", flush=True)


if __name__ == "__main__":
    main()
