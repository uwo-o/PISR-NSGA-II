#!/usr/bin/env python3
"""
plot_solutions.py — Solucion simbolica (PISR-NSGA-II) vs referencia, una figura
por problema de la suite (report/benchmarks.py).

  1D real     : u(x) arriba, |error| en escala log abajo.
  1D complejo : Re u e Im u arriba, |error| complejo abajo.
  2D          : referencia | PISR-NSGA-II | mapa de |error| (Re u si es complejo).
Problemas no estacionarios: corte en t = 0.5 (el que exporta save_best_grid).

Las grillas de lineas base externas (DeepXDE, PySR, PySINDy) se superponen solo
si son posteriores a la redefinicion de los problemas (benchmarks.is_fresh).
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "report"))
from benchmarks import BENCHMARKS, is_fresh, latest_result  # noqa: E402

RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGS_DIR = os.path.join(BASE_DIR, "report", "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})

C_REF = "#1B1B1B"
C_PISR = "#2E86C1"
EXTERNAL = [  # (metodo, sufijos de archivo, color, estilo)
    ("DeepXDE", ["DeepXDE", "PINN"], "#27AE60", "-."),
    ("PySR", ["PySR"], "#16A085", ":"),
    ("PySINDy", ["PySINDy"], "#8E44AD", "--"),
]
CMAP = "viridis"
CMAP_ERR = "magma"


def grid_path(label, suffix):
    # Misma busqueda que el resto del reporte: results/ o results/run_*/, solo
    # archivos posteriores a la redefinicion de los problemas (no ablaciones
    # ni resultados apartados en otras subcarpetas).
    return latest_result(f"grid_{label}_{suffix}.csv")


def load_field(df, which):
    """Columna compleja u_<which> (+ u_<which>_im si existe)."""
    re = df[f"u_{which}"].to_numpy(dtype=float)
    im_col = f"u_{which}_im"
    im = df[im_col].to_numpy(dtype=float) if im_col in df.columns else np.zeros_like(re)
    return re + 1j * im


def external_grids(b):
    out = []
    for name, suffixes, color, ls in EXTERNAL:
        for s in suffixes:
            p = grid_path(b.label, s)
            if p and is_fresh(p):
                out.append((name, pd.read_csv(p), color, ls))
                break
    return out


def title_for(b):
    t = f"{b.display} ({b.dim}D)"
    if b.unsteady:
        t += "  —  t = 0.5"
    return t.replace("\\'e", "é").replace('\\"o', "ö").replace("--", "–").replace("$", "")


def mse_text(ref, approx):
    err = np.abs(ref - approx)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return "grid MSE: n/a"
    return f"grid MSE = {np.mean(err ** 2):.2e}   max|err| = {err.max():.2e}"


def plot_1d(b, df, ext):
    x = df["x"].to_numpy(dtype=float)
    ref, pisr = load_field(df, "exact"), load_field(df, "approx")
    parts = [("Re u", np.real)] + ([("Im u", np.imag)] if b.complex_field else [])

    fig = plt.figure(figsize=(11 if b.complex_field else 8, 6.2), layout="constrained")
    gs = fig.add_gridspec(2, len(parts), height_ratios=[2.2, 1])
    ref_label = "Numerical reference" if b.numerical else "Exact solution"

    for k, (name, f) in enumerate(parts):
        ax = fig.add_subplot(gs[0, k])
        ax.plot(x, f(ref), color=C_REF, lw=3.2, alpha=0.35, label=ref_label)
        ax.plot(x, f(pisr), color=C_PISR, lw=1.8, ls="--", label="PISR-NSGA-II")
        for ename, edf, color, ls in ext:
            ex = edf["x"] if "x" in edf.columns else edf["t"]
            ax.plot(ex, f(load_field(edf, "approx")), color=color, lw=1.4, ls=ls, label=ename)
        ax.set_xlabel("x")
        ax.set_ylabel(name if b.complex_field else "u(x)")
        if b.complex_field:
            ax.set_title(name, fontsize=11)
        ax.set_xlim(x.min(), x.max())
        if k == 0:
            ax.legend(loc="best")

    ax_e = fig.add_subplot(gs[1, :])
    err = np.clip(np.abs(ref - pisr), 1e-17, None)
    ax_e.semilogy(x, err, color=C_PISR, lw=1.5, label="PISR-NSGA-II")
    for ename, edf, color, ls in ext:
        ex = edf["x"] if "x" in edf.columns else edf["t"]
        e2 = np.clip(np.abs(load_field(edf, "exact") - load_field(edf, "approx")), 1e-17, None)
        ax_e.semilogy(ex, e2, color=color, lw=1.2, ls=ls, label=ename)
    ax_e.set_xlabel("x")
    ax_e.set_ylabel("|error|")
    ax_e.set_xlim(x.min(), x.max())
    ax_e.text(0.99, 0.95, mse_text(ref, pisr), transform=ax_e.transAxes, ha="right", va="top", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D5D8DC"))
    if ext:
        ax_e.legend(loc="best")

    fig.suptitle(title_for(b))
    return fig


def plot_2d(b, df, ext):
    n = int(round(np.sqrt(len(df))))
    if n * n != len(df):
        print(f"  [WARN] {b.label}: grilla no cuadrada ({len(df)} puntos), se omite")
        return None
    X = df["x"].to_numpy(dtype=float).reshape(n, n)
    Y = df["y"].to_numpy(dtype=float).reshape(n, n)
    ref = load_field(df, "exact").reshape(n, n)
    pisr = load_field(df, "approx").reshape(n, n)
    part = "Re u" if b.complex_field else "u"

    panels = [("Reference" if not b.numerical else "Numerical reference", np.real(ref)),
              ("PISR-NSGA-II", np.real(pisr))]
    for ename, edf, _, _ in ext:
        if len(edf) == n * n:
            panels.append((ename, np.real(load_field(edf, "approx")).reshape(n, n)))

    ncol = len(panels) + 1
    fig = plt.figure(figsize=(4.0 * ncol, 7.4), layout="constrained")
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.15, 1])
    vmin = min(np.nanmin(p[1]) for p in panels)
    vmax = max(np.nanmax(p[1]) for p in panels)

    for k, (name, Z) in enumerate(panels):
        ax3 = fig.add_subplot(gs[0, k], projection="3d")
        ax3.plot_surface(X, Y, Z, cmap=CMAP, vmin=vmin, vmax=vmax, edgecolor="none", antialiased=True)
        ax3.set_title(name)
        ax3.set_xlabel("x"); ax3.set_ylabel("y")
        ax3.set_xticks([0, 0.5, 1]); ax3.set_yticks([0, 0.5, 1])
        ax3.view_init(elev=25, azim=-60)
        ax = fig.add_subplot(gs[1, k])
        im = ax.imshow(Z, origin="lower", extent=[0, 1, 0, 1], cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(f"{part} (top view)", fontsize=10, fontweight="normal")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)

    E = np.abs(ref - pisr)
    ax = fig.add_subplot(gs[:, -1])
    im = ax.imshow(E, origin="lower", extent=[0, 1, 0, 1], cmap=CMAP_ERR)
    ax.set_title("|error| PISR-NSGA-II")
    ax.set_xlabel(mse_text(ref, pisr).replace("   ", "\n"), fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle(title_for(b))
    return fig


def main():
    made = 0
    for b in BENCHMARKS:
        p = grid_path(b.label, "PISR-EMOAD")
        if not p:
            print(f"  [--] {b.label}: sin grilla, se omite")
            continue
        df = pd.read_csv(p)
        ext = external_grids(b)
        fig = plot_1d(b, df, ext) if b.dim == 1 else plot_2d(b, df, ext)
        if fig is None:
            continue
        out = os.path.join(FIGS_DIR, f"solution_{b.label}.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        made += 1
        print(f"  [OK] {os.path.relpath(out, BASE_DIR)}")
    print(f"{made} figuras de solucion en {os.path.relpath(FIGS_DIR, BASE_DIR)}")


if __name__ == "__main__":
    main()
