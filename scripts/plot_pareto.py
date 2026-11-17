#!/usr/bin/env python3
"""
plot_pareto.py — Frentes de Pareto por pares (dominio, frontera, complejidad)
para cada problema de la suite (report/benchmarks.py).

Escribe directamente report/figures/pareto_pairs_<Problema>_<d>D.pdf.
Gris: soluciones dominadas; azul: frente no dominado (rank 1). Si la frontera
es identicamente 0 (ansatz exacto en casi toda la poblacion), los ejes de
frontera pasan a escala lineal con una nota: log(0) no se puede graficar.
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "report"))
from benchmarks import BENCHMARKS, latest_result  # noqa: E402

FIGS_DIR = os.path.join(BASE_DIR, "report", "figures")
os.makedirs(FIGS_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
    "legend.fontsize": 9,
})
C_FRONT = "#2E86C1"
C_DOM = "#BDC3C7"


def plot(b, df):
    for c in ["mse_domain", "mse_boundary", "tree_size", "rank"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["mse_domain", "mse_boundary", "tree_size"])
    df = df[(df["mse_domain"] < 1e5) & (df["mse_boundary"] < 1e5)]
    front, dom = df[df["rank"] == 1], df[df["rank"] != 1]
    if front.empty:
        return None
    bnd_flat = (df["mse_boundary"].abs() < 1e-12).mean() > 0.9
    floor = 1e-35

    pairs = [
        ("mse_domain", "mse_boundary", "Domain residual MSE", "Boundary MSE", "Physics vs. boundary"),
        ("mse_domain", "tree_size", "Domain residual MSE", "Complexity (nodes)", "Physics vs. complexity"),
        ("mse_boundary", "tree_size", "Boundary MSE", "Complexity (nodes)", "Boundary vs. complexity"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for ax, (xc, yc, xl, yl, title) in zip(axes, pairs):
        xlog = xc != "tree_size" and not (xc == "mse_boundary" and bnd_flat)
        ylog = yc != "tree_size" and not (yc == "mse_boundary" and bnd_flat)
        fx = (lambda s: s.clip(lower=floor)) if xlog else (lambda s: s)
        fy = (lambda s: s.clip(lower=floor)) if ylog else (lambda s: s)
        if not dom.empty:
            ax.scatter(fx(dom[xc]), fy(dom[yc]), s=14, color=C_DOM, alpha=0.6, label="Dominated", zorder=1)
        ax.scatter(fx(front[xc]), fy(front[yc]), s=30, color=C_FRONT, edgecolor="white", linewidth=0.5,
                   label="Pareto front (rank 1)", zorder=2)
        if xlog: ax.set_xscale("log")
        if ylog: ax.set_yscale("log")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
        if bnd_flat and "mse_boundary" in (xc, yc):
            ax.text(0.5, 0.92, "Exact boundary ansatz: boundary MSE ≡ 0", transform=ax.transAxes,
                    ha="center", va="top", fontsize=8, color="#666666", style="italic")
    axes[0].legend(loc="best")
    name = f"{b.display} ({b.dim}D)".replace("\\'e", "é").replace('\\"o', "ö").replace("--", "–").replace("$", "")
    fig.suptitle(f"Pareto trade-offs — {name}", fontsize=12, fontweight="bold")
    return fig


def main():
    made = 0
    for b in BENCHMARKS:
        path = latest_result(f"{b.label}_pi_gn_pareto.csv")
        if not path:
            print(f"  [--] {b.label}: sin frente de Pareto")
            continue
        fig = plot(b, pd.read_csv(path))
        if fig is None:
            continue
        out = os.path.join(FIGS_DIR, f"pareto_pairs_{b.label}.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        made += 1
    print(f"{made} figuras de Pareto en {os.path.relpath(FIGS_DIR, BASE_DIR)}")


if __name__ == "__main__":
    main()
