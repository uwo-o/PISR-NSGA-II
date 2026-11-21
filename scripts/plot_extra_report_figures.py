#!/usr/bin/env python3
"""
plot_extra_report_figures.py — Figuras globales del reporte, un panel por
problema de la suite (report/benchmarks.py), en el orden y agrupacion del
reporte:

  convergence_history.pdf : val_score del campeon (criterio de seleccion, solo
                            informacion de la EDP) y error contra la referencia
                            (solo registro, nunca usado en la busqueda);
                            mediana y banda intercuartil sobre las semillas.
  Las demas figuras usan la semilla mediana de cada problema
  (benchmarks.representative_run).
  parsimony_pressure.pdf  : mejor MSE total del frente por tamano de arbol.
  pareto_fronts_3d.pdf    : frente de Pareto en (dominio, frontera, nodos).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "report"))
from benchmarks import BENCHMARKS, FAMILY_TITLE, is_fresh, latest_result, seed_dirs  # noqa: E402

FIGS_DIR = os.path.join(BASE_DIR, "report", "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})
C_VAL = "#2E86C1"
C_REF = "#E67E22"
FAMILY_COLOR = {"linear": "#2E86C1", "nonlin": "#C0392B", "complex": "#8E44AD",
                "fluids": "#16A085", "singular": "#D35400"}
NCOLS = 4


def nice(b):
    return f"{b.display} ({b.dim}D)".replace("\\'e", "é").replace('\\"o', "ö").replace("--", "–").replace("\\lambda", "λ").replace("$", "")


def grid_axes(projection=None, panel_h=3.3):
    n = len(BENCHMARKS)
    nrows = (n + NCOLS - 1) // NCOLS
    fig = plt.figure(figsize=(4.2 * NCOLS, panel_h * nrows), layout="constrained")
    axes = [fig.add_subplot(nrows, NCOLS, i + 1, projection=projection) for i in range(n)]
    return fig, axes


def title(ax, b):
    ax.set_title(nice(b), color=FAMILY_COLOR.get(b.family, "black"))


def family_legend(fig):
    handles = [plt.Line2D([], [], color=c, lw=6) for c in FAMILY_COLOR.values()]
    labels = [FAMILY_TITLE[k].replace("\\'e", "é") for k in FAMILY_COLOR]
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9, title="Panel title colour = problem family")


def seed_curves(label, col):
    """Matriz (semillas x generaciones) de `col`. Una semilla que paro antes
    (parada temprana) conserva su ultimo valor: el campeon ya no cambia."""
    curves = []
    for d in seed_dirs():
        p = os.path.join(d, f"{label}_pi_gn_convergence.csv")
        if not os.path.exists(p) or not is_fresh(p):
            continue
        df = pd.read_csv(p)
        if df.empty or col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce").where(lambda v: v < 1e17)
        curves.append(pd.Series(y.to_numpy(), index=df["gen"].to_numpy()))
    if not curves:
        return None
    n = int(max(c.index.max() for c in curves)) + 1
    M = np.vstack([c.reindex(range(n)).ffill().to_numpy() for c in curves])
    return M


def plot_convergence():
    fig, axes = grid_axes()
    n_seeds = sum(any(is_fresh(os.path.join(d, f)) for f in os.listdir(d)) for d in seed_dirs())
    for ax, b in zip(axes, BENCHMARKS):
        title(ax, b)
        M0 = seed_curves(b.label, "best_val_score") if n_seeds else None
        if n_seeds and (M0 is None or M0.shape[1] <= 2):
            # Parada temprana en la generacion 0-1: el registro corta antes de
            # anotar la generacion en que se resolvio (solo queda la gen. 0).
            ax.text(0.5, 0.5, "solved within the first generation\nin every seed (early stop)",
                    ha="center", va="center", transform=ax.transAxes, color="#777", fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        drew = False
        for col, color, label, ls in [("best_val_score", C_VAL, "Validation score (selection)", "-"),
                                      ("champion_ref_err", C_REF, "Error vs. reference (log only)", "--")]:
            M = seed_curves(b.label, col) if n_seeds else None
            if M is None:
                p = latest_result(f"{b.label}_pi_gn_convergence.csv")  # corrida unica
                if not p or pd.read_csv(p).empty:
                    continue
                df = pd.read_csv(p)
                M = pd.to_numeric(df[col], errors="coerce").to_numpy()[None, :]
            M = np.clip(M, 1e-30, None)
            g = np.arange(M.shape[1])
            with np.errstate(all="ignore"):
                med = np.nanmedian(M, axis=0)
                q25, q75 = np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0)
            ax.plot(g, med, color=color, lw=1.6, ls=ls, label=label)
            if M.shape[0] > 1:
                ax.fill_between(g, q25, q75, color=color, alpha=0.18, lw=0)
            drew = True
        if not drew:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="#999", fontsize=8.5)
            continue
        ax.set_yscale("log")
        ax.set_xlabel("generation")
    # Leyenda en el primer panel con curvas (Laplace se resuelve en la gen. 0).
    next((ax for ax in axes if ax.get_lines()), axes[0]).legend(loc="upper right")
    family_legend(fig)
    sub = f"median and interquartile band over {n_seeds} seeds" if n_seeds > 1 else "single run"
    fig.suptitle(f"Convergence of the hall-of-fame champion ({sub})")
    save(fig, "convergence_history.pdf")


def plot_parsimony():
    fig, axes = grid_axes()
    for ax, b in zip(axes, BENCHMARKS):
        title(ax, b)
        p = latest_result(f"{b.label}_pi_gn_pareto.csv")
        if not p:
            continue
        df = pd.read_csv(p)
        df = df[(df["mse_domain"] < 1e5) & (df["mse_boundary"] < 1e5)].copy()
        if df.empty:
            continue
        df["total"] = (df["mse_domain"] + df["mse_boundary"]).clip(lower=1e-35)
        best = df.groupby("tree_size")["total"].min().reset_index().sort_values("tree_size")
        best["envelope"] = best["total"].cummin()
        ax.scatter(df["tree_size"], df["total"], s=6, color="#BDC3C7", alpha=0.6)
        ax.step(best["tree_size"], best["envelope"], where="post", color=FAMILY_COLOR.get(b.family, C_VAL), lw=2)
        ax.set_yscale("log")
        ax.set_xlabel("tree size (nodes)")
        ax.set_ylabel("domain + boundary MSE")
    family_legend(fig)
    fig.suptitle("Parsimony: best attainable loss for a given expression size")
    save(fig, "parsimony_pressure.pdf")


def plot_pareto_3d():
    fig, axes = grid_axes(projection="3d", panel_h=3.6)
    sc = None
    for ax, b in zip(axes, BENCHMARKS):
        ax.set_title(nice(b), color=FAMILY_COLOR.get(b.family, "black"), fontsize=10, fontweight="bold")
        p = latest_result(f"{b.label}_pi_gn_pareto.csv")
        if not p:
            continue
        df = pd.read_csv(p)
        df = df[(df["mse_domain"] < 1e5) & (df["mse_boundary"] < 1e5)]
        if df.empty:
            continue
        x = np.log10(df["mse_domain"].clip(lower=1e-35))
        y = np.log10(df["mse_boundary"].clip(lower=1e-35))
        z = df["tree_size"]
        front = df["rank"] == 1
        ax.scatter(x[~front], y[~front], z[~front], c="#D5D8DC", s=6, alpha=0.3)
        sc = ax.scatter(x[front], y[front], z[front], c=z[front], cmap="viridis", s=18, edgecolors="white", linewidth=0.2)
        ax.set_xlabel("log dom", fontsize=8, labelpad=-2)
        ax.set_ylabel("log bnd", fontsize=8, labelpad=-2)
        ax.set_zlabel("nodes", fontsize=8, labelpad=-2)
        ax.tick_params(labelsize=7, pad=0)
        ax.view_init(elev=20, azim=-120)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, shrink=0.35, aspect=30, location="right")
        cb.set_label("complexity (nodes)")
    fig.suptitle("Pareto fronts: domain residual vs. boundary error vs. complexity")
    save(fig, "pareto_fronts_3d.pdf")


def save(fig, name):
    out = os.path.join(FIGS_DIR, name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {os.path.relpath(out, BASE_DIR)}")


if __name__ == "__main__":
    plot_convergence()
    plot_parsimony()
    plot_pareto_3d()
