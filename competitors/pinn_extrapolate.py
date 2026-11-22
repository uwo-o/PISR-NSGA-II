#!/usr/bin/env python3
"""Evalua cada PINN entrenado fuera de su dominio de entrenamiento, para la
seccion de extrapolacion del reporte.

Problemas 1D estacionarios (entrenados en [0,1]): la red se evalua en la misma
grilla que usa report/extrapolation.py, x en [0, 1.5] con 601 puntos.
Escribe results/run_<k>/<EDP>_pinn_extrap.csv (x, u_re, u_im).
No reentrena nada: solo carga los pesos guardados (<EDP>_pinn.pt).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pinn import MLP  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X_MAX, N = 1.5, 601


def main():
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    xs = torch.linspace(0.0, X_MAX, N, dtype=torch.float64).reshape(-1, 1)
    n = 0
    for pt in sorted(glob.glob(os.path.join(ROOT, "results", "run_*", "*_pinn.pt"))):
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        if ck["cols"] != ["x"]:  # solo 1D estacionarios
            continue
        model = MLP(1, ck["n_out"], ck["width"], ck["depth"]).double()
        model.load_state_dict(ck["state"])
        model.eval()
        with torch.no_grad():
            u = model(xs).numpy()
        out = pt.replace("_pinn.pt", "_pinn_extrap.csv")
        pd.DataFrame({"x": xs.numpy().ravel(), "u_re": u[:, 0],
                      "u_im": u[:, 1] if ck["n_out"] > 1 else 0.0}).to_csv(out, index=False)
        n += 1
    print(f"[PINN extrap] {n} redes evaluadas en [0, {X_MAX}]")


if __name__ == "__main__":
    main()
