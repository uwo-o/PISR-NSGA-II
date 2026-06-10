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
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 55,
    "axes.titlesize": 60,
    "axes.labelsize": 55,
    "xtick.labelsize": 48,
    "ytick.labelsize": 48,
    "legend.fontsize": 45,
    "figure.titlesize": 88,
})

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGS_DIR    = os.path.join(BASE_DIR, "report", "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

# Todas las ecuaciones que el benchmark genera
PDE_ORDER = [
    "Airy", "Fisher", "Duffing", "ThomasFermi",
    "Navier-Stokes", "Navier-Stokes-Unsteady", 
    "Lane-Emden", "Troesch", "Ginzburg-Landau", "Painleve-I"
]

# Ecuaciones cuya "u_exact" en el CSV proviene de RK4 (no hay fórmula cerrada)
NUMERICAL_TRUTH = {
    "Airy", "HarmonicOscillator", "Liouville", "Sine-Gordon", 
    "NonlinearPoisson", "Fisher", "Duffing", "ThomasFermi",
    "Bratu", "Allen-Cahn", "Lane-Emden",
    "Troesch", "Ginzburg-Landau", "Painleve-I"
}

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
    "Sine-Gordon":        r"Sine-Gordon: $\nabla^2 u = \sin(u)$",
    "Navier-Stokes":      r"Navier-Stokes: $\psi_y (\nabla^2 \psi)_x - \psi_x (\nabla^2 \psi)_y = \nu \nabla^4 \psi$",
    "Navier-Stokes-Unsteady": r"Navier-Stokes (Unsteady): $u_t + u u_x = \nu \nabla^2 u$",
    "Bratu":              r"Bratu: $\nabla^2 u + \lambda e^u = 0$",
    "Allen-Cahn":         r"Allen-Cahn: $u_t = \epsilon^2 \Delta u - (u^3 - u)$",
    "Lane-Emden":         r"Lane-Emden: $u'' + \frac{2}{x} u' + u^3 = 0$",
    "Troesch":            r"Troesch: $u'' = \mu \sinh(\mu u)$ [RK4 truth]",
    "Ginzburg-Landau":    r"Ginzburg-Landau: $u'' + u - u^3 = 0$ [RK4 truth]",
    "Painleve-I":         r"Painleve-I: $u'' = u^2 + x$ [RK4 truth]"
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
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(PDE_LABELS.get(pde, pde) + "  —  1D", fontweight="bold", y=0.96)
    gs = GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[3, 1],
        hspace=0.30,
        top=0.88,
        bottom=0.15,
        left=0.20,
        right=0.95
    )

    ax_sol = fig.add_subplot(gs[0])
    ax_err = fig.add_subplot(gs[1])

    # Remove top and right spines
    ax_sol.spines['top'].set_visible(False)
    ax_sol.spines['right'].set_visible(False)
    ax_err.spines['top'].set_visible(False)
    ax_err.spines['right'].set_visible(False)

    x = df_pi["x"].values
    u_exact  = df_pi["u_exact"].values

    # Panel superior: soluciones
    ax_sol.plot(x, u_exact,  "k-",  lw=1.8, label=truth_label(pde), alpha=0.75, zorder=3)
    
    if df_pi is not None:
        u_pi = df_pi["u_approx"].values
        ax_sol.plot(x, u_pi, "b--", lw=1.5, label="PISR-NSGA-II", zorder=4)
        err_pi = np.abs(u_exact - u_pi)
        # Clip para evitar errores de escala logarítmica con ceros exactos
        err_pi = np.clip(err_pi, 1e-32, None)
        ax_err.plot(x, err_pi, "b-", lw=1.3, label="Error PISR-NSGA-II")

    if df_pn is not None:
        ax_sol.plot(df_pn["x"], df_pn["u_approx"], "g-.", lw=1.3, label="DeepXDE", zorder=6)
        err_pn = np.abs(df_pn["u_exact"].values - df_pn["u_approx"].values)
        # Clip para evitar errores de escala logarítmica con ceros exactos
        err_pn = np.clip(err_pn, 1e-32, None)
        ax_err.plot(df_pn["x"], err_pn, "g-", lw=1.2, label="Error DeepXDE")

    ax_sol.set_ylabel("u(x)", labelpad=60)
    ax_sol.set_xticks([0, 0.5, 1])
    ax_sol.set_xticklabels(["0", "0.5", "1"])
    ax_sol.legend(framealpha=0.85)
    ax_sol.grid(True, alpha=0.3)

    # Panel inferior: error absoluto (log)
    ax_err.set_yscale("log")
    ax_err.set_xlabel("x", labelpad=50)
    ax_err.set_ylabel(r"$|u - u^*|$", labelpad=60)
    ax_err.set_xticks([0, 0.5, 1])
    ax_err.set_xticklabels(["0", "0.5", "1"])
    ax_err.legend(framealpha=0.85)
    ax_err.grid(True, which="both", alpha=0.3)

    out = os.path.join(FIGS_DIR, f"solution_1d_{pde}.pdf")
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  [1D] {pde}: {out}")


# ─── 2D ────────────────────────────────────────────────────────────────────────
def plot_2d(pde, df_pi, df_pn=None):
    # Si hay columna 't', filtramos por t=0 (Condición Inicial / Slice)
    if "t" in df_pi.columns:
        t_unique = np.unique(df_pi["t"].values)
        t_slice = t_unique[0] # Usar el primer instante (t=0 usualmente)
        df_pi = df_pi[df_pi["t"] == t_slice]
        if df_pn is not None and "t" in df_pn.columns:
            df_pn = df_pn[df_pn["t"] == t_slice]
        print(f"  [2D-Slice] {pde} at t={t_slice:.2f}")

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
    cols.append(("Exact", Z_ex, None, CMAP_SOLUTION))
    if has_pi:
        Z_pi = df_pi["u_approx"].values.reshape(N, N)
        cols.append(("PISR-NSGA-II", Z_pi, np.abs(Z_ex - Z_pi), CMAP_SOLUTION))
    if has_pinn:
        N_pn = int(np.sqrt(len(df_pn)))
        if N_pn * N_pn == len(df_pn):
            Z_pn = df_pn["u_approx"].values.reshape(N_pn, N_pn)
            Z_ex_pn = df_pn["u_exact"].values.reshape(N_pn, N_pn)
            cols.append(("DeepXDE", Z_pn, np.abs(Z_ex_pn - Z_pn), CMAP_SOLUTION))
        else:
            print(f"  [WARN] {pde} 2D: len(df_pn)={len(df_pn)} no es cuadrado perfecto, saltando DeepXDE.")

    n_cols = len(cols)
    fig = plt.figure(figsize=(15 * n_cols, 22))
    fig.suptitle(PDE_LABELS.get(pde, pde) + "  —  2D", fontweight="bold", y=0.95)
    gs = matplotlib.gridspec.GridSpec(
        2, n_cols,
        figure=fig,
        height_ratios=[2.2, 1],   # superficie más alta que heatmap
        hspace=0.25,              # separación vertical optimizada
        wspace=0.48,              # separación horizontal optimizada
        top=0.88,                 # dejar espacio para el título grande
        bottom=0.10,              # margen inferior amplio
        left=0.12,                # margen izquierdo amplio para y-labels
        right=0.88                # margen derecho amplio para colorbars
    )

    def surface(row, col, Z, title, cmap=CMAP_SOLUTION):
        ax = fig.add_subplot(gs[row, col], projection="3d")
        N_Z = Z.shape[0]
        xs = np.linspace(0, 1, N_Z)
        X_Z, Y_Z = np.meshgrid(xs, xs)
        surf = ax.plot_surface(X_Z, Y_Z, Z, cmap=cmap, alpha=0.92,
                               rcount=30, ccount=30,
                               linewidth=0, antialiased=False)
        ax.set_title(title, pad=35, fontweight="bold")
        ax.set_xlabel("\n\nx", labelpad=10)
        ax.set_ylabel("\n\ny", labelpad=10)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "0.5", "1"])
        ax.view_init(elev=28, azim=-50)
        # Separar los números (ticks) de las líneas de los ejes
        ax.tick_params(axis='x', pad=20)
        ax.tick_params(axis='y', pad=20)
        ax.tick_params(axis='z', pad=25)
        # Ocultar paneles grises del fondo
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("lightgray")
        ax.yaxis.pane.set_edgecolor("lightgray")
        ax.zaxis.pane.set_edgecolor("lightgray")
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.20)
        return ax

    def heatmap(row, col, Z, title, cmap=CMAP_SOLUTION, is_error=False):
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(Z, origin="lower", extent=[0, 1, 0, 1],
                       cmap=cmap, aspect="auto")
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title(title, pad=25)
        ax.set_xlabel("x", labelpad=40)
        ax.set_ylabel("y", labelpad=45)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "0.5", "1"])
        
        if is_error:
            def format_sci(val, pos):
                if abs(val) < 1e-15:
                    return "$0$"
                s = f"{val:.1e}"
                base, exp = s.split('e')
                exp = int(exp)
                if base == "1.0":
                    return f"$10^{{{exp}}}$"
                return f"${base} \\times 10^{{{exp}}}$"
            
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, format=ticker.FuncFormatter(format_sci), pad=0.10)
        else:
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.10)
            
        return ax

    # Fila 0: superficies 3D
    for idx, (name, Z, _, cmap) in enumerate(cols):
        surface(0, idx, Z, name, cmap)

    # Fila 1: heatmaps
    for idx, (name, Z, E, cmap) in enumerate(cols):
        if E is None:
            heatmap(1, idx, Z, name + " (top view)", cmap, is_error=False)
        else:
            heatmap(1, idx, E, f"Error — {name}", CMAP_ERROR, is_error=True)

    out = os.path.join(FIGS_DIR, f"solution_2d_{pde}.pdf")
    fig.savefig(out, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  [2D] {pde}: {out}")


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def plot_equation(pde, dim):
    suffix = f"_{dim}D"
    path_pi = find_file(f"grid_{pde}{suffix}_PI-NSGA-II.csv")
    path_pn = find_file(f"grid_{pde}{suffix}_DeepXDE.csv") or find_file(f"grid_{pde}{suffix}_PINN.csv")

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

    skips_1d = {"NonlinearPoisson", "Liouville", "Sine-Gordon", "Navier-Stokes", "Navier-Stokes-Unsteady", "Bratu", "Allen-Cahn"}   # solo 2D/3D
    skips_2d = {"Lane-Emden"} # Lane-Emden es 1D en este benchmark

    for pde in PDE_ORDER:
        if pde not in skips_1d:
            plot_equation(pde, 1)
        if pde not in skips_2d:
            # Para Navier-Stokes-Unsteady, graficamos el slice temporal t=0 si es 2D
            plot_equation(pde, 2)

    print("\nPlots updated in", FIGS_DIR)


if __name__ == "__main__":
    main()
