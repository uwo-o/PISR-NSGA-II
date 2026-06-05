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
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 200
})

PDES_TO_PLOT = ["Laplace_2D", "Helmholtz_2D", "Schrodinger_2D", "Fisher_2D", "Duffing_2D", "ThomasFermi_2D", "Liouville_2D", "Sine-Gordon_2D"]

def plot_convergence():
    fig, axes = plt.subplots(1, len(PDES_TO_PLOT), figsize=(24, 4), sharey=False)
    for idx, pde_name in enumerate(PDES_TO_PLOT):
        ax = axes[idx]
        # Cargar PI-NSGA-II
        gn_file = os.path.join(RESULTS_DIR, f"{pde_name}_pi_gn_convergence.csv")
        if os.path.exists(gn_file):
            df_gn = pd.read_csv(gn_file)
            ax.plot(df_gn["gen"], df_gn["best_total_mse"], label="PI-NSGA-II", color="#2E86C1", lw=2)
            
        ax.set_title(pde_name.replace("_", " "))
        ax.set_xlabel("Generations")
        if idx == 0:
            ax.set_ylabel("Best Total MSE")
        ax.set_yscale("log")
        ax.legend(frameon=True)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.suptitle("Convergence History: PI-NSGA-II Constants Optimization", y=1.02, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(FIGS_DIR, "convergence_history.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Generated: {out_path}")
    plt.close(fig)

def plot_parsimony():
    fig, axes = plt.subplots(1, len(PDES_TO_PLOT), figsize=(24, 4), sharey=False)
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
                    label="PI-NSGA-II", color="#2E86C1", marker="o", lw=1.5)
            
        ax.set_title(pde_name.replace("_", " "))
        ax.set_xlabel("Complexity (Tree Nodes)")
        if idx == 0:
            ax.set_ylabel("Total MSE")
        ax.set_yscale("log")
        ax.legend(frameon=True)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.suptitle("Parsimony Pressure: Accuracy vs. Expression Complexity", y=1.02, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(FIGS_DIR, "parsimony_pressure.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Generated: {out_path}")
    plt.close(fig)

def plot_pareto_fronts():
    fig, axes = plt.subplots(1, len(PDES_TO_PLOT), figsize=(24, 4), sharey=False)
    for idx, pde_name in enumerate(PDES_TO_PLOT):
        ax = axes[idx]
        
        # PI-NSGA-II Pareto
        gn_file = os.path.join(RESULTS_DIR, f"{pde_name}_pi_gn_pareto.csv")
        if os.path.exists(gn_file):
            df = pd.read_csv(gn_file)
            df = df[(df["mse_domain"] < 1e5) & (df["mse_boundary"] < 1e5)]
            ax.scatter(df["mse_domain"], df["mse_boundary"], label="PI-NSGA-II", color="#2E86C1", alpha=0.7, edgecolors="none")
            
        ax.set_title(pde_name.replace("_", " "))
        ax.set_xlabel("Domain MSE (Ω)")
        if idx == 0:
            ax.set_ylabel("Boundary MSE (∂Ω)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(frameon=True)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.suptitle("Pareto Fronts: Domain Objective vs. Boundary Objective", y=1.02, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(FIGS_DIR, "pareto_fronts.png")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Generated: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    plot_convergence()
    plot_parsimony()
    plot_pareto_fronts()
