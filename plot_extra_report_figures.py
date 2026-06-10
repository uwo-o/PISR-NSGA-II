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
import matplotlib
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "figure.titlesize": 22,
})

PDES_TO_PLOT = [
    "Airy_2D", "Fisher_2D", "Duffing_2D", "ThomasFermi_2D", 
    "Navier-Stokes_2D", "Navier-Stokes-Unsteady_2D",
    "Lane-Emden_1D", "Troesch_1D", "Ginzburg-Landau_1D", "Painleve-I_1D"
]

def plot_convergence():
    # Calcular filas necesarias para 3 columnas
    n_pdes = len(PDES_TO_PLOT)
    n_cols = 3
    n_rows = (n_pdes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16.5, 4.5 * n_rows), sharey=False, layout="constrained")
    axes = axes.flatten()
    
    for idx, pde_name in enumerate(PDES_TO_PLOT):
        ax = axes[idx]
        # Cargar PI-NSGA-II
        gn_file = os.path.join(RESULTS_DIR, f"{pde_name}_pi_gn_convergence.csv")
        if os.path.exists(gn_file):
            df_gn = pd.read_csv(gn_file)
            ax.plot(df_gn["gen"], df_gn["best_total_mse"], label="PISR-NSGA-II", color="#2E86C1", lw=3)
            
        ax.set_title(pde_name.replace("_", " "), pad=15, fontweight="bold")
        ax.set_xlabel("Generations", labelpad=8)
        ax.set_ylabel("Best Total MSE", labelpad=8)
        ax.set_yscale("log")
        ax.legend(frameon=True)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    # Ocultar ejes vacíos si hay un número impar de PDEs
    for i in range(n_pdes, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Convergence History: PISR-NSGA-II Constants Optimization", fontweight="bold")
    out_path = os.path.join(FIGS_DIR, "convergence_history.pdf")
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    print(f"Generated: {out_path}")
    plt.close(fig)

def plot_parsimony():
    n_pdes = len(PDES_TO_PLOT)
    n_cols = 3
    n_rows = (n_pdes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16.5, 4.5 * n_rows), sharey=False, layout="constrained")
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
                    label="PISR-NSGA-II", color="#2E86C1", marker="o", lw=2.5)
            
        ax.set_title(pde_name.replace("_", " "), pad=15, fontweight="bold")
        ax.set_xlabel("Complexity (Tree Nodes)", labelpad=8)
        ax.set_ylabel("Total MSE", labelpad=8)
        ax.set_yscale("log")
        ax.legend(frameon=True)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    for i in range(n_pdes, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Parsimony Pressure: Accuracy vs. Expression Complexity", fontweight="bold")
    out_path = os.path.join(FIGS_DIR, "parsimony_pressure.pdf")
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    print(f"Generated: {out_path}")
    plt.close(fig)

def plot_pareto_fronts():
    n_pdes = len(PDES_TO_PLOT)
    n_cols = 3
    n_rows = (n_pdes + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    for idx, pde_name in enumerate(PDES_TO_PLOT):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        
        # PI-NSGA-II Pareto
        gn_file = os.path.join(RESULTS_DIR, f"{pde_name}_pi_gn_pareto.csv")
        if os.path.exists(gn_file):
            df = pd.read_csv(gn_file)
            df = df[(df["mse_domain"] < 1e5) & (df["mse_boundary"] < 1e5)]
            if len(df) == 0:
                continue
                
            x = df["mse_domain"].clip(lower=1e-20)
            y = df["mse_boundary"].clip(lower=1e-20)
            z = df["tree_size"]
            r = df["rank"]
            
            # Identify fronts
            mask_rank1 = (r == 1)
            mask_dominated = (r > 1)
            
            # Plot dominated (transparent)
            if mask_dominated.any():
                ax.scatter(np.log10(x[mask_dominated]), 
                           np.log10(y[mask_dominated]), 
                           z[mask_dominated], 
                           c='gray', s=30, alpha=0.4, label='Dominated')
            
            # Plot rank 1 (opaque)
            if mask_rank1.any():
                scatter = ax.scatter(np.log10(x[mask_rank1]), 
                                     np.log10(y[mask_rank1]), 
                                     z[mask_rank1], 
                                     c=z[mask_rank1], cmap='viridis', s=80, alpha=1.0, 
                                     edgecolors="white", linewidth=0.5, label='Pareto Front')
                # Colorbar only for the main front
                cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
                cbar.set_label("Complexity", rotation=270, labelpad=15)
            
        ax.set_title(pde_name.replace("_", " "), pad=15, fontweight="bold")
        ax.set_xlabel("log10(Dom MSE)", labelpad=10)
        ax.set_ylabel("log10(Bnd MSE)", labelpad=10)
        ax.set_zlabel("Complexity", labelpad=10)
        
        # Orientación para ver bien las 3 dimensiones
        ax.view_init(elev=25, azim=-135)

    plt.suptitle("3D Pareto Analysis: Domain MSE vs. Boundary MSE vs. Complexity", fontweight="bold", y=0.95)
    out_path = os.path.join(FIGS_DIR, "pareto_fronts.pdf")
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    print(f"Generated: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    plot_convergence()
    plot_parsimony()
    plot_pareto_fronts()
