#!/usr/bin/env python3
"""
plot_solutions.py — Visualización de soluciones simbólicas vs verdad numérica (RK4/FD).

Para cada ecuación:
  - 1D: curva simbólica vs verdad (exacta o RK4) + panel de error absoluto.
  - 2D: superficie simbólica + verdad + heatmap de error.

La columna u_exact en los CSV puede ser:
  - La solución analítica exacta (Laplace, Poisson, …)
  - La solución numérica RK4/Relajación Jacobi (Airy, HO, Liouville, Sine-Gordon)
"""
import os, glob, numpy as np, pandas as pd, matplotlib, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   88,
    "axes.titlesize": 96,
    "axes.labelsize": 92,
    "xtick.labelsize": 84,
    "ytick.labelsize": 84,
    "legend.fontsize": 80,
    "figure.titlesize": 120,
    "figure.dpi":  200,
})

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGS_DIR    = os.path.join(BASE_DIR, "report", "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

# Todas las ecuaciones que el benchmark genera
PDE_ORDER = [
    "Laplace", "Poisson", "Helmholtz", "Schrodinger",
    "Airy", "HarmonicOscillator",
    "Fisher", "Duffing", "ThomasFermi",
    "NonlinearPoisson", "Liouville", "Sine-Gordon", "Navier-Stokes",
]

# Ecuaciones cuya "u_exact" en el CSV proviene de RK4 (no hay fórmula cerrada)
NUMERICAL_TRUTH = {"Airy", "HarmonicOscillator", "Liouville", "Sine-Gordon", "NonlinearPoisson", "Fisher", "Duffing", "ThomasFermi"}

# Etiqueta del eje / título para cada ecuación
PDE_LABELS = {
    "Laplace":            r"Laplace: $\nabla^2 u = 0$",
    "Poisson":            r"Poisson: $\nabla^2 u = f$",
    "Helmholtz":          r"Helmholtz: $\nabla^2 u + k^2 u = f$",
    "Schrodinger":        r"Schrödinger: $-u'' + V u = E u$",
    "Airy":               r"Airy: $u'' = x\,u$  [RK4 truth]",
    "HarmonicOscillator": r"Harmonic Oscillator: $u'' = (x^2-1)u$  [RK4 truth]",
    "Fisher":             r"Fisher: $\nabla^2 u + u(1-u) = 0$  [RK4/FD truth]",
    "Duffing":            r"Duffing: $\nabla^2 u + u + u^3 = 0$  [RK4/FD truth]",
    "ThomasFermi":        r"Thomas-Fermi: $\nabla^2 u = u^2 / (x+y+0.5)$  [RK4/FD truth]",
    "NonlinearPoisson":   r"Nonlinear Poisson: $\nabla^2 u + u^2 = f$  [FD truth]",
    "Liouville":          r"Liouville: $\nabla^2 u = e^u$  [FD truth]",
    "Sine-Gordon":        r"Sine-Gordon: $\nabla^2 u = \sin(u)$  [FD truth]",
    "Navier-Stokes":      r"Navier-Stokes: $\psi_y (\nabla^2 \psi)_x - \psi_x (\nabla^2 \psi)_y = \nu \nabla^4 \psi$",
}

CMAP_SOLUTION = "viridis"
CMAP_ERROR    = "inferno"


def find_file(pattern):
    hits = glob.glob(os.path.join(RESULTS_DIR, "**", pattern), recursive=True)
    return hits[0] if hits else None


def truth_label(pde):
    return "RK4 / FD truth" if pde in NUMERICAL_TRUTH else "Analytical"


# ─── 1D ────────────────────────────────────────────────────────────────────────
def plot_1d(pde, df_pi, df_pn=None):
    fig = plt.figure(figsize=(40, 28), layout="constrained")
    fig.suptitle(PDE_LABELS.get(pde, pde) + "  —  1D", fontweight="bold", fontsize=160)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1])

    ax_sol = fig.add_subplot(gs[0])
    ax_err = fig.add_subplot(gs[1])

    x = df_pi["x"].values
    u_exact  = df_pi["u_exact"].values

    # Panel superior: soluciones
    ax_sol.plot(x, u_exact,  "k-",  lw=4.0, label=truth_label(pde), alpha=0.75, zorder=3)
    
    if df_pi is not None:
        u_pi = df_pi["u_approx"].values
        ax_sol.plot(x, u_pi, "b--", lw=3.5, label="PISR-NSGA-II", zorder=4)
        err_pi = np.abs(u_exact - u_pi)
        ax_err.plot(x, err_pi, "b-", lw=3.0, label="Error PISR-NSGA-II")

    if df_pn is not None:
        ax_sol.plot(df_pn["x"], df_pn["u_approx"], "g-.", lw=3.0, label="PINN", zorder=6)
        err_pn = np.abs(df_pn["u_exact"].values - df_pn["u_approx"].values)
        ax_err.plot(df_pn["x"], err_pn, "g-", lw=2.5, label="Error PINN")

    ax_sol.set_ylabel("u(x)")
    ax_sol.legend(framealpha=0.85)
    ax_sol.grid(True, alpha=0.3)

    # Panel inferior: error absoluto (log)
    ax_err.set_yscale("log")
    ax_err.set_xlabel("x")
    ax_err.set_ylabel(r"$|u - u^*|$")
    ax_err.legend(framealpha=0.85)
    ax_err.grid(True, which="both", alpha=0.3)

    out = os.path.join(FIGS_DIR, f"solution_1d_{pde}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1D] {pde}: {out}")


# ─── 2D ────────────────────────────────────────────────────────────────────────
def plot_2d(pde, df_pi, df_pn=None):
    N = int(np.sqrt(len(df_pi)))
    if N * N != len(df_pi):
        print(f"  [WARN] {pde} 2D: len={len(df_pi)} no es cuadrado perfecto, saltando.")
        return

    X = df_pi["x"].values.reshape(N, N)
    Y = df_pi["y"].values.reshape(N, N)
    Z_ex  = df_pi["u_exact"].values.reshape(N, N)

    has_pi = df_pi is not None
    has_pinn = df_pn is not None
    
    cols = []
    cols.append(("Exacta", Z_ex, None, CMAP_SOLUTION))
    if has_pi:
        Z_pi = df_pi["u_approx"].values.reshape(N, N)
        cols.append(("PISR-NSGA-II", Z_pi, np.abs(Z_ex - Z_pi), CMAP_SOLUTION))
    if has_pinn:
        Z_pn = df_pn["u_approx"].values.reshape(N, N)
        cols.append(("PINN", Z_pn, np.abs(Z_ex - Z_pn), CMAP_SOLUTION))

    n_cols = len(cols)
    fig = plt.figure(figsize=(28 * n_cols, 44), layout="constrained")
    fig.suptitle(PDE_LABELS.get(pde, pde) + "  —  2D", fontweight="bold", fontsize=160)

    def surface(pos, Z, title, cmap=CMAP_SOLUTION):
        ax = fig.add_subplot(2, n_cols, pos, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85, antialiased=True)
        ax.set_title(title + f"\nmax={Z.max():.2e}  min={Z.min():.2e}", fontsize=100, pad=60)
        ax.set_xlabel("x", labelpad=30); ax.set_ylabel("y", labelpad=30); ax.view_init(elev=28, azim=-50)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08)
        cbar.ax.tick_params(labelsize=76)
        return ax

    def heatmap(pos, Z, title, cmap=CMAP_SOLUTION):
        ax = fig.add_subplot(2, n_cols, pos)
        im = ax.imshow(Z, origin="lower", extent=[0,1,0,1], cmap=cmap, aspect="auto")
        ax.set_title(title + f"\nmax={Z.max():.2e}", fontsize=100, pad=60)
        ax.set_xlabel("x", labelpad=30); ax.set_ylabel("y", labelpad=30)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=76)
        return ax

    # Fila 1: superficies 3D
    for idx, (name, Z, _, cmap) in enumerate(cols):
        surface(idx + 1, Z, name, cmap)

    # Fila 2: heatmaps
    for idx, (name, Z, E, cmap) in enumerate(cols):
        if E is None:
            heatmap(n_cols + idx + 1, Z, name + " (top)", cmap)
        else:
            heatmap(n_cols + idx + 1, E, f"Error {name}", CMAP_ERROR)

    out = os.path.join(FIGS_DIR, f"solution_2d_{pde}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2D] {pde}: {out}")


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def plot_equation(pde, dim):
    suffix = f"_{dim}D"
    path_pi = find_file(f"grid_{pde}{suffix}_PI-NSGA-II.csv")
    path_pn = find_file(f"grid_{pde}{suffix}_PINN.csv")

    if not path_pi:
        print(f"  [SKIP] Sin datos para {pde} {dim}D.")
        return

    try:
        df_pi = pd.read_csv(path_pi)
        df_pn = pd.read_csv(path_pn) if path_pn else None

        u_ex = df_pi["u_exact"].values
        if np.all(u_ex == 0) or np.all(~np.isfinite(u_ex)):
            print(f"  [WARN] {pde} {dim}D: u_exact todo cero/NaN, graficando solo aprox.")
            df_pi["u_exact"] = df_pi["u_approx"]

        if dim == 1:
            plot_1d(pde, df_pi, df_pn)
        else:
            plot_2d(pde, df_pi, df_pn)

    except Exception as e:
        print(f"  [ERROR] {pde} {dim}D: {e}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    skips_1d = {"NonlinearPoisson", "Liouville", "Sine-Gordon", "Navier-Stokes"}   # solo 2D
    skips_2d = set()

    for pde in PDE_ORDER:
        if pde not in skips_1d:
            plot_equation(pde, 1)
        if pde not in skips_2d:
            plot_equation(pde, 2)

    print("\nPlots updated in", FIGS_DIR)


if __name__ == "__main__":
    main()
