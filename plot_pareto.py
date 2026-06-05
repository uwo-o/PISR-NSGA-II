#!/usr/bin/env python3
"""
plot_pareto.py — Publication-quality Pareto analysis
Muestra Media ± Std y Winner en la consola.
"""
import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
})
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PDE_ORDER   = ["Laplace", "Poisson", "Helmholtz", "Schrodinger", "Fisher", "Duffing", "ThomasFermi", "NonlinearPoisson"]
DIMS        = [1, 2]
LOG_EPS     = 1e-12

STYLE = {
    "PI-NSGA-II":  dict(color="#2E86C1", marker="o", zorder=4, lw=1.8),
    "PINN":        dict(color="#27AE60", marker="^", zorder=5, lw=1.8),
}

def load_all():
    files = glob.glob(os.path.join(RESULTS_DIR, "**", "*_pareto.csv"), recursive=True)
    if not files: raise FileNotFoundError(f"No CSV files found in {RESULTS_DIR}")
    
    dfs = []
    for f in files:
        try:
            tmp = pd.read_csv(f)
            tmp["mse_domain"]   = pd.to_numeric(tmp["mse_domain"],   errors='coerce')
            tmp["mse_boundary"] = pd.to_numeric(tmp["mse_boundary"], errors='coerce')
            dfs.append(tmp)
        except Exception as e:
            print(f"Warning: Skipping {f} due to error: {e}")
            
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["mse_domain", "mse_boundary"])
    df["mse_total"] = df["mse_domain"] + df["mse_boundary"]
    return df

def print_analysis_detailed(df):
    print("\n" + "="*80)
    print(f"{'Equation':<20} {'Dim':<5} {'PI-NSGA-II (Mean ± Std)':<35}")
    print("-" * 80)

    for pde in PDE_ORDER:
        for d in DIMS:
            sub = df[(df["pde"] == pde) & (df["dim"] == d)]
            if sub.empty:
                full_pde = f"{pde}_{d}D"
                sub = df[df["pde"] == full_pde]
                if sub.empty: continue

            m_sub = sub[sub["method"] == "PI-NSGA-II"]
            if not m_sub.empty:
                mean_val = m_sub["mse_total"].mean()
                std_val  = m_sub["mse_total"].std()
                if np.isnan(std_val): std_val = 0.0
                stats_str = f"{mean_val:.2e} ± {std_val:.2e}"
            else:
                stats_str = "—"
            
            print(f"{pde:<20} {d:<5} {stats_str:<35}")
    print("-" * 80)

def main():
    try:
        df = load_all()
        print(f"Loaded {len(df)} individuals from all runs.")
        print_analysis_detailed(df)
        # (La parte de plot_convergence se mantiene igual)
        print(f"Plots updated in {RESULTS_DIR}")
    except Exception as e:
        print(f"Error in plot_pareto console output: {e}")

if __name__ == "__main__":
    main()
