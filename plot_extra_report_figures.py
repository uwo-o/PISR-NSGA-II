#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGS_DIR = os.path.join(BASE_DIR, "report", "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

# Estilo premium
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 40,
    "axes.labelsize": 48,
    "axes.titlesize": 52,
    "xtick.labelsize": 40,
    "ytick.labelsize": 40,
    "figure.titlesize": 64,
    "figure.dpi": 200
})

PDES_TO_PLOT = ["Laplace_2D", "Helmholtz_2D", "Schrodinger_2D", "Fisher_2D", "Duffing_2D", "ThomasFermi_2D", "Liouville_2D", "Sine-Gordon_2D"]

def plot_convergence():
    # Calcular filas necesarias para 2 columnas
    n_pdes = len(PDES_TO_PLOT)
    n_cols = 3
    n_rows = (n_pdes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(44, 10 * n_rows), sharey=False, layout="constrained")
    axes = axes.flatten()
    
    for idx, pde_name in enumerate(PDES_TO_PLOT):
        ax = axes[idx]
        # Cargar PI-NSGA-II
        gn_file = os.path.join(RESULTS_DIR, f"{pde_name}_pi_gn_convergence.csv")
        if os.path.exists(gn_file):
            df_gn = pd.read_csv(gn_file)
            ax.plot(df_gn["gen"], df_gn["best_total_mse"], label="PISR-NSGA-II", color="#2E86C1", lw=4)
            
        ax.set_title(pde_name.replace("_", " "), pad=40, fontweight="bold")
        ax.set_xlabel("Generations")
        ax.set_ylabel("Best Total MSE")
        ax.set_yscale("log")
        ax.legend(frameon=True)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    # Ocultar ejes vacíos si hay un número impar de PDEs
    for i in range(n_pdes, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Convergence History: PISR-NSGA-II Constants Optimization", fontweight="bold")
    out_path = os.path.join(FIGS_DIR, "convergence_history.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Generated: {out_path}")
    plt.close(fig)

def plot_parsimony():
    n_pdes = len(PDES_TO_PLOT)
    n_cols = 3
    n_rows = (n_pdes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(44, 10 * n_rows), sharey=False, layout="constrained")
    axes = axes.flatten()
    
    for idx, pde_name in enumerate(PDES_TO_PLOT):
        ax = axes[idx]
        
        # Cargar PI-NSGA-II pareto
        gn_file = os.path.join(RESULTS_DIR, f"{pde_name}_pi_gn_pareto.csv")
        if os.path.exists(gn_file):
            df = pd.read_csv(gn_file)
            df = df[df["mse_domain"] < 1e5] # Filtrar outliers iniciales
            df["total_mse"] = df["mse_domain"] + df["mse_boundary"]
            # Agrupar por tree_size y tomar el mejor total_mse
            best_by_size = df.groupby("tree_size")["total_mse"].min().reset_index()
            best_by_size = best_by_size.sort_values("tree_size")
            ax.step(best_by_size["tree_size"], best_by_size["total_mse"], where="post",
                    label="PISR-NSGA-II", color="#2E86C1", marker="o", lw=3)
            
        ax.set_title(pde_name.replace("_", " "), pad=40, fontweight="bold")
        ax.set_xlabel("Complexity (Tree Nodes)")
        ax.set_ylabel("Total MSE")
        ax.set_yscale("log")
        ax.legend(frameon=True)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    for i in range(n_pdes, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Parsimony Pressure: Accuracy vs. Expression Complexity", fontweight="bold")
    out_path = os.path.join(FIGS_DIR, "parsimony_pressure.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Generated: {out_path}")
    plt.close(fig)

def plot_pareto_fronts():
    n_pdes = len(PDES_TO_PLOT)
    n_cols = 3
    n_rows = (n_pdes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(44, 10 * n_rows), sharey=False, layout="constrained")
    axes = axes.flatten()
    
    for idx, pde_name in enumerate(PDES_TO_PLOT):
        ax = axes[idx]
        
        # PI-NSGA-II Pareto
        gn_file = os.path.join(RESULTS_DIR, f"{pde_name}_pi_gn_pareto.csv")
        if os.path.exists(gn_file):
            df = pd.read_csv(gn_file)
            df = df[(df["mse_domain"] < 1e5) & (df["mse_boundary"] < 1e5)]
            if len(df) == 0:
                continue
                
            x = df["mse_domain"].clip(lower=1e-12)
            y = df["mse_boundary"].clip(lower=1e-12)
            z = df["tree_size"]
            
            # Plot 2D scatter with color mapped to 3rd dimension
            scatter = ax.scatter(x, y, c=z, cmap='viridis', s=100, alpha=0.9, edgecolors="white", linewidth=0.8, zorder=3)
            
            # Add colorbar to current axis
            cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label("Complexity (Tree Nodes)", rotation=270, labelpad=25, fontweight="bold")
            
        ax.set_title(pde_name.replace("_", " "), pad=40, fontweight="bold")
        ax.set_xlabel("Domain MSE (Ω)", labelpad=20)
        ax.set_ylabel("Boundary MSE (∂Ω)", labelpad=20)
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.4, zorder=0)

    # Ocultar ejes vacíos
    for i in range(n_pdes, len(axes)):
        axes[i].set_visible(False)
        
    plt.suptitle("Pareto Fronts: Domain vs. Boundary (Color = Complexity)", fontweight="bold")
    out_path = os.path.join(FIGS_DIR, "pareto_fronts.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Generated: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    plot_convergence()
    plot_parsimony()
    plot_pareto_fronts()
