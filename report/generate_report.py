#!/usr/bin/env python3
"""
report/generate_report.py — Genera tablas LaTeX centradas en PI-NSGA-II.
"""
import os, sys, shutil, glob
import numpy as np
import pandas as pd

REPORT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(REPORT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TABLES_DIR  = os.path.join(REPORT_DIR, "tables")
FIGS_DIR    = os.path.join(REPORT_DIR, "figures")

# Ecuaciones base
PDE_BASE_NAMES = ["Laplace", "Poisson", "Helmholtz", "Schrodinger", "Airy", "HarmonicOscillator", "Fisher", "Duffing", "ThomasFermi", "NonlinearPoisson", "Liouville", "Sine-Gordon", "Navier-Stokes"]
DIMS = [1, 2]

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGS_DIR,   exist_ok=True)

def fmt_sci_stat(m, s):
    if m is None or np.isnan(m) or m == float('inf'): return "—"
    def get_exp(v):
        if v == 0 or np.isnan(v) or v == float('inf'): return 0
        return int(np.floor(np.log10(abs(v))))
    
    exp_m = get_exp(m)
    mant_m = m / (10**exp_m)
    mant_s = s / (10**exp_m) if (not np.isnan(s) and s != float('inf')) else 0.0
    return rf"({mant_m:.2f} \pm {mant_s:.2f}) \times 10^{{{exp_m}}}"

def make_symbolic_table():
    summary_path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    if not os.path.exists(summary_path): return
    df = pd.read_csv(summary_path)
    df["best_mse_domain"]   = pd.to_numeric(df["best_mse_domain"],   errors='coerce')
    df["best_mse_boundary"] = pd.to_numeric(df["best_mse_boundary"], errors='coerce')
    df = df.dropna(subset=["best_mse_domain", "best_mse_boundary"])
    df["mse_total"] = df["best_mse_domain"] + df["best_mse_boundary"]

    lines = [
        r"\begin{table*}[ht]",
        r"  \centering",
        r"  \caption{PI-NSGA-II Performance Metrics (Mean $\pm$ Std).}",
        r"  \label{tab:symbolic_stats}",
        r"  \resizebox{\textwidth}{!}{",
        r"  \begin{tabular}{llcccc}",
        r"    \toprule",
        r"    \textbf{PDE} & \textbf{Dim} & \textbf{Domain MSE} & \textbf{BC MSE} & \textbf{Total MSE} & \textbf{Hypervolume} \\",
        r"    \midrule"
    ]

    for pde_base in PDE_BASE_NAMES:
        for d in DIMS:
            full_pde_name = f"{pde_base}_{d}D"
            sub = df[(df["pde"] == full_pde_name) & (df["method"] == "PI-NSGA-II")]
            if sub.empty: continue
            
            stats = sub.agg({
                "best_mse_domain": ["mean", "std"],
                "best_mse_boundary": ["mean", "std"],
                "mse_total": ["mean", "std"],
                "hypervolume": ["mean", "std"]
            })
            
            f_dom = fmt_sci_stat(stats.loc["mean", "best_mse_domain"], stats.loc["std", "best_mse_domain"])
            f_bnd = fmt_sci_stat(stats.loc["mean", "best_mse_boundary"], stats.loc["std", "best_mse_boundary"])
            f_tot = fmt_sci_stat(stats.loc["mean", "mse_total"], stats.loc["std", "mse_total"])
            f_hv  = f"{stats.loc['mean', 'hypervolume']:.4f}"
            
            lines.append(rf"    {pde_base} & {d}D & ${f_dom}$ & ${f_bnd}$ & ${f_tot}$ & {f_hv} \\")
            lines.append(r"    \midrule")

    # Remove the last extra midrule if we have lines
    if len(lines) > 9:
        lines[-1] = r"    \bottomrule"
    else:
        lines.append(r"    \bottomrule")

    lines += [r"  \end{tabular}", r"  }", r"\end{table*}"]
    out = os.path.join(TABLES_DIR, "symbolic_comparison.tex")
    with open(out, "w") as f: f.write("\n".join(lines) + "\n")

def make_global_comparison_table():
    summary_path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    if not os.path.exists(summary_path): return
    df = pd.read_csv(summary_path)
    df["mse_total"] = pd.to_numeric(df["best_mse_domain"], errors='coerce') + pd.to_numeric(df["best_mse_boundary"], errors='coerce')
    df = df.dropna(subset=["mse_total"])

    lines = [
        r"\begin{table*}[ht]",
        r"  \centering",
        r"  \caption{Performance Comparison: Numerical (RK4/FDM), Neural (PINN), and Symbolic (PI-NSGA-II) Methods (Total MSE).}",
        r"  \label{tab:global_comp_stats}",
        r"  \resizebox{\textwidth}{!}{",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    \textbf{PDE} & \textbf{Dim} & \textbf{RK4/FDM} & \textbf{PINN} & \textbf{PI-NSGA-II} \\",
        r"    \midrule"
    ]

    for pde_base in PDE_BASE_NAMES:
        for d in DIMS:
            full_pde_name = f"{pde_base}_{d}D"
            sub = df[df["pde"] == full_pde_name]
            if sub.empty: continue
            
            pi_sub   = sub[sub["method"] == "PI-NSGA-II"]
            pinn_sub = sub[sub["method"] == "PINN"]
            num_sub  = sub[sub["method"] == "RK4/FDM"]
            
            f_pi = "—"
            if not pi_sub.empty:
                s_pi = pi_sub.agg({"mse_total": ["mean", "std"]})
                f_pi = fmt_sci_stat(s_pi.loc["mean", "mse_total"], s_pi.loc["std", "mse_total"])
            
            f_num = "—"
            if not num_sub.empty:
                s_num = num_sub.agg({"mse_total": ["mean", "std"]})
                f_num = fmt_sci_stat(s_num.loc["mean", "mse_total"], s_num.loc["std", "mse_total"])

            f_pinn = "—"
            if not pinn_sub.empty:
                s_pinn = pinn_sub.agg({"mse_total": ["mean", "std"]})
                f_pinn = fmt_sci_stat(s_pinn.loc["mean", "mse_total"], s_pinn.loc["std", "mse_total"])
            
            lines.append(rf"    {pde_base} & {d}D & ${f_num}$ & ${f_pinn}$ & ${f_pi}$ \\")
            lines.append(r"    \midrule")

    # Remove the last extra midrule if we have lines
    if len(lines) > 9:
        lines[-1] = r"    \bottomrule"
    else:
        lines.append(r"    \bottomrule")

    lines += [r"  \end{tabular}", r"  }", r"\end{table*}"]
    out = os.path.join(TABLES_DIR, "global_comparison.tex")
    with open(out, "w") as f: f.write("\n".join(lines) + "\n")

def copy_figures():
    for pde in PDE_BASE_NAMES:
        for d in [1, 2]:
            fname = f"solution_{d}d_{pde}.png"
            src = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(src): shutil.copy2(src, os.path.join(FIGS_DIR, fname))

if __name__ == "__main__":
    make_symbolic_table()
    make_global_comparison_table()
    copy_figures()
    print("Report generation updated: Focus on PI-NSGA-II.")
