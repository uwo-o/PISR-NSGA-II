#!/usr/bin/env python3
"""
plot_solutions.py — Solucion simbolica (PISR-NSGA-II) vs referencia, una figura
por problema de la suite (report/benchmarks.py).

  1D real     : u(x) arriba, |error| en escala log abajo.
  1D complejo : Re u e Im u arriba, |error| complejo abajo.
  2D          : referencia | PISR-NSGA-II | mapa de |error| (Re u si es complejo).
Problemas no estacionarios: corte en t = 0.5 (el que exporta save_best_grid).

Se superponen los competidores (PINN y PINN+PySR, de run_competitors.sh), cada
uno en su semilla mediana, igual que PISR-NSGA-II (benchmarks.latest_result).
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
# Competidores (run_competitors.sh), cada uno en su propia semilla mediana.
C_PINN, C_PYSR = "#27AE60", "#E67E22"
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


def _median_run(files):
    """Archivo de la semilla mediana (por solution_mse) entre los CSV de un competidor."""
    rows = []
    for f in files:
        if not is_fresh(f):
            continue
        try:
            v = float(pd.read_csv(f)["solution_mse"].iloc[0])
        except Exception:
            continue
        if np.isfinite(v):
            rows.append((v, f))
    if not rows:
        return None
    rows.sort()
    return rows[(len(rows) - 1) // 2][1]


def _reference_points(b):
    p = os.path.join(RESULTS_DIR, "reference", f"{b.label}.csv")
    if not os.path.exists(p):
        return None
    r = pd.read_csv(p)
    return r[r.kind == "dom"].reset_index(drop=True)


def competitor_grids(b, df):
    """[(nombre, DataFrame con x[,y], u_exact[_im], u_approx[_im], color, estilo)] en el corte que
    muestra la figura (t = 0.5 si no es estacionario).
    PINN: su prediccion en los puntos de validacion (en 2D, interpolada a la grilla de la figura).
    PINN+PySR: su formula evaluada en la grilla de la figura."""
    import sympy as sp
    ref = _reference_points(b)
    if ref is None:
        return []
    out = []
    t_cut = 0.5 if b.unsteady else 0.0
    sel = np.isclose(ref["t"].to_numpy(), t_cut) if b.unsteady else np.ones(len(ref), bool)

    f = _median_run(glob.glob(os.path.join(RESULTS_DIR, "run_*", f"{b.label}_pinn.csv")))
    if f:
        pred = pd.read_csv(f.replace("_pinn.csv", "_pinn_pred.csv"))
        r, q = ref[sel].reset_index(drop=True), pred[sel].reset_index(drop=True)
        if b.dim == 1:
            e = pd.DataFrame({"x": r["x"], "u_exact": r["u_re"], "u_exact_im": r["u_im"],
                              "u_approx": q["u_re"], "u_approx_im": q["u_im"]})
        else:
            from scipy.interpolate import RegularGridInterpolator
            xs, ys = np.unique(r["x"]), np.unique(r["y"])
            pts = np.column_stack([df["x"], df["y"]])
            cols = {}
            for c in ("u_re", "u_im"):
                Z = q[c].to_numpy().reshape(len(xs), len(ys))
                cols[c] = RegularGridInterpolator((xs, ys), Z, method="cubic", bounds_error=False,
                                                  fill_value=None)(pts)
            e = pd.DataFrame({"x": df["x"], "y": df["y"], "u_exact": df["u_exact"],
                              "u_exact_im": df.get("u_exact_im", 0.0),
                              "u_approx": cols["u_re"], "u_approx_im": cols["u_im"]})
        out.append(("PINN", e, C_PINN, "-."))

    f = _median_run(glob.glob(os.path.join(RESULTS_DIR, "run_*", f"{b.label}_pinn_pysr.csv")))
    if f:
        row = pd.read_csv(f).iloc[0]
        try:
            expr = sp.sympify(eval(row["expr_srepr"], sp.__dict__.copy()))
            names = {s.name: s for s in expr.free_symbols}
            X, Y, T = (names.get(k, sp.Symbol(k)) for k in "xyt")
            fn = sp.lambdify((X, Y, T), expr, modules=["numpy"])
            if b.dim == 1:
                r = ref[sel].reset_index(drop=True)
                xv, yv = r["x"].to_numpy(float), np.zeros(len(r))
                ex_re, ex_im = r["u_re"], r["u_im"]
            else:
                xv, yv = df["x"].to_numpy(float), df["y"].to_numpy(float)
                ex_re, ex_im = df["u_exact"], df.get("u_exact_im", 0.0)
            v = np.asarray(fn(xv.astype(complex), yv.astype(complex), np.full(len(xv), t_cut, complex)),
                           dtype=complex)
            v = np.full(len(xv), complex(v)) if v.shape == () else v
            e = pd.DataFrame({"x": xv, "u_exact": ex_re, "u_exact_im": ex_im,
                              "u_approx": v.real, "u_approx_im": v.imag if b.complex_field else 0.0})
            if b.dim == 2:
                e.insert(1, "y", yv)
            out.append(("PINN+PySR", e, C_PYSR, ":"))
        except Exception as ex:
            print(f"  [WARN] {b.label}: no se pudo evaluar la formula de PINN+PySR ({ex})")
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
        return "MSE en grilla: n/d"
    return f"MSE en grilla = {np.mean(err ** 2):.2e}   max|err| = {err.max():.2e}"


def plot_1d(b, df, ext):
    x = df["x"].to_numpy(dtype=float)
    ref, pisr = load_field(df, "exact"), load_field(df, "approx")
    parts = [("Re u", np.real)] + ([("Im u", np.imag)] if b.complex_field else [])

    fig = plt.figure(figsize=(11 if b.complex_field else 8, 6.2), layout="constrained")
    gs = fig.add_gridspec(2, len(parts), height_ratios=[2.2, 1])
    ref_label = "Referencia numérica" if b.numerical else "Solución exacta"

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

    methods = [("PISR-NSGA-II", pisr)]
    for ename, edf, _, _ in ext:
        if len(edf) == n * n:
            methods.append((ename, load_field(edf, "approx").reshape(n, n)))
    panels = [("Referencia" if not b.numerical else "Referencia numérica", ref)] + methods

    ncol = len(panels)
    fig = plt.figure(figsize=(4.0 * ncol, 10.5), layout="constrained")
    gs = fig.add_gridspec(3, ncol, height_ratios=[1.15, 1, 1])
    vmin = min(np.nanmin(np.real(p[1])) for p in panels)
    vmax = max(np.nanmax(np.real(p[1])) for p in panels)

    for k, (name, Zc) in enumerate(panels):
        Z = np.real(Zc)
        ax3 = fig.add_subplot(gs[0, k], projection="3d")
        ax3.plot_surface(X, Y, Z, cmap=CMAP, vmin=vmin, vmax=vmax, edgecolor="none", antialiased=True)
        ax3.set_title(name)
        ax3.set_xlabel("x"); ax3.set_ylabel("y")
        ax3.set_xticks([0, 0.5, 1]); ax3.set_yticks([0, 0.5, 1])
        ax3.view_init(elev=25, azim=-60)
        ax = fig.add_subplot(gs[1, k])
        im = ax.imshow(Z.T, origin="lower", extent=[0, 1, 0, 1], cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(f"{part} (vista superior)", fontsize=10, fontweight="normal")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
        if k == 0:
            ax = fig.add_subplot(gs[2, 0]); ax.axis("off")
            ax.text(0.5, 0.5, "Fila inferior:\n|error| de cada método\ncontra la referencia",
                    ha="center", va="center", fontsize=10, color="#555")
            continue
        E = np.abs(ref - Zc)
        ax = fig.add_subplot(gs[2, k])
        im = ax.imshow(E.T, origin="lower", extent=[0, 1, 0, 1], cmap=CMAP_ERR)
        ax.set_title(f"|error| {name}", fontsize=10)
        ax.set_xlabel(mse_text(ref, Zc).replace("   ", "\n"), fontsize=8.5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)

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
        ext = competitor_grids(b, df)
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
