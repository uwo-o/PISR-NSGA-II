#!/usr/bin/env python3
"""
report/generate_report.py — Genera tablas LaTeX centradas en PISR-NSGA-II.

Fuentes de datos:
  - PI-NSGA-II / RK4/FDM : results/all_runs_summary.csv
  - PINN                  : results/**/*_pinn_pareto.csv  (formato: pde, dim, mse_domain, mse_boundary)
"""
import os, sys, shutil, glob
import numpy as np
import pandas as pd

REPORT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(REPORT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TABLES_DIR  = os.path.join(REPORT_DIR, "tables")
FIGS_DIR    = os.path.join(REPORT_DIR, "figures")

# Ecuaciones base (en orden de la tabla)
PDE_BASE_NAMES = [
    "Laplace", "Poisson", "Helmholtz", "Schrodinger",
    "Airy", "HarmonicOscillator",
    "Fisher", "Duffing", "ThomasFermi",
    "NonlinearPoisson", "Liouville", "Sine-Gordon", "Navier-Stokes",
]
DIMS = [1, 2]

# PDEs que solo existen en 2D
ONLY_2D = {"NonlinearPoisson", "Liouville", "Sine-Gordon", "Navier-Stokes"}

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGS_DIR,   exist_ok=True)


# ─── Formato numérico ─────────────────────────────────────────────────────────
def fmt_sci_stat(m, s):
    """Devuelve cadena LaTeX $(a \\pm b) \\times 10^{e}$ o «—» si inválido."""
    if m is None or np.isnan(m) or not np.isfinite(m):
        return "—"
    if m == 0.0:
        return r"(0.00 \pm 0.00) \times 10^{0}"

    def get_exp(v):
        if v == 0 or not np.isfinite(v): return 0
        return int(np.floor(np.log10(abs(v))))

    exp_m  = get_exp(m)
    mant_m = m / (10 ** exp_m)
    s_safe = s if (s is not None and np.isfinite(s)) else 0.0
    mant_s = s_safe / (10 ** exp_m) if exp_m != 0 else s_safe
    return rf"({mant_m:.2f} \pm {mant_s:.2f}) \times 10^{{{exp_m}}}"


# ─── Carga de datos ───────────────────────────────────────────────────────────
def load_pi_rk4() -> pd.DataFrame:
    """Carga all_runs_summary.csv (PI-NSGA-II + RK4/FDM)."""
    path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["best_mse_domain"]   = pd.to_numeric(df["best_mse_domain"],   errors="coerce")
    df["best_mse_boundary"] = pd.to_numeric(df["best_mse_boundary"], errors="coerce")
    df["mse_total"] = df["best_mse_domain"] + df["best_mse_boundary"]
    # pde ya viene como "Laplace_1D", "Poisson_2D", etc.
    return df


def load_pinn() -> pd.DataFrame:
    """
    Carga todos los *_pinn_pareto.csv y los normaliza al mismo esquema que
    all_runs_summary.csv: columnas method, pde (con sufijo _ND), mse_total.
    """
    pattern = os.path.join(RESULTS_DIR, "**", "*pinn_pareto.csv")
    files   = glob.glob(pattern, recursive=True)
    # También buscar en el nivel raíz de results/
    files  += glob.glob(os.path.join(RESULTS_DIR, "*pinn_pareto.csv"))
    files   = list(set(files))   # deduplicar

    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            tmp = pd.read_csv(f)
            tmp["mse_domain"]   = pd.to_numeric(tmp["mse_domain"],   errors="coerce")
            tmp["mse_boundary"] = pd.to_numeric(tmp["mse_boundary"], errors="coerce")
            # Normalizar nombre de PDE: añadir sufijo _ND si no lo tiene
            if "dim" in tmp.columns:
                tmp["pde"] = tmp["pde"].astype(str) + "_" + tmp["dim"].astype(int).astype(str) + "D"
            # Renombrar columnas para que coincidan con el schema principal
            tmp = tmp.rename(columns={
                "mse_domain":   "best_mse_domain",
                "mse_boundary": "best_mse_boundary",
            })
            tmp["method"]    = "PINN"
            tmp["mse_total"] = tmp["best_mse_domain"] + tmp["best_mse_boundary"]
            dfs.append(tmp[["method", "pde", "best_mse_domain", "best_mse_boundary", "mse_total"]])
        except Exception as e:
            print(f"  [WARN] Skipping {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    # Si hay varias corridas del PINN para la misma PDE, promediar
    df = df.groupby(["method", "pde"], as_index=False).agg(
        best_mse_domain  = ("best_mse_domain",  "mean"),
        best_mse_boundary= ("best_mse_boundary","mean"),
        mse_total        = ("mse_total",         "mean"),
        mse_total_std    = ("mse_total",         "std"),
    )
    return df


def load_all() -> pd.DataFrame:
    """Une PI-NSGA-II/RK4 con PINN en un único DataFrame."""
    df_main = load_pi_rk4()
    df_pinn = load_pinn()

    # Añadir columna std al df_main si no existe
    if not df_main.empty:
        df_main["mse_total_std"] = 0.0

    if df_pinn.empty and df_main.empty:
        return pd.DataFrame()
    if df_pinn.empty:
        return df_main
    if df_main.empty:
        return df_pinn

    # Alinear columnas
    keep = ["method", "pde", "best_mse_domain", "best_mse_boundary",
            "mse_total", "mse_total_std"]
    for c in keep:
        if c not in df_main.columns: df_main[c] = np.nan
        if c not in df_pinn.columns: df_pinn[c] = np.nan

    return pd.concat([df_main[keep], df_pinn[keep]], ignore_index=True)


# ─── Tabla 1: Solo PISR-NSGA-II (métricas detalladas) ────────────────────────
def make_symbolic_table():
    summary_path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    if not os.path.exists(summary_path): return
    df = pd.read_csv(summary_path)
    df["best_mse_domain"]   = pd.to_numeric(df["best_mse_domain"],   errors="coerce")
    df["best_mse_boundary"] = pd.to_numeric(df["best_mse_boundary"], errors="coerce")
    df = df.dropna(subset=["best_mse_domain", "best_mse_boundary"])
    df["mse_total"] = df["best_mse_domain"] + df["best_mse_boundary"]

    lines = [
        r"\begin{table*}[ht]",
        r"  \centering",
        r"  \caption{PISR-NSGA-II Performance Metrics (Mean $\pm$ Std).}",
        r"  \label{tab:symbolic_stats}",
        r"  \resizebox{\textwidth}{!}{",
        r"  \begin{tabular}{llcccc}",
        r"    \toprule",
        r"    \textbf{PDE} & \textbf{Dim} & \textbf{Domain MSE} & \textbf{BC MSE} & \textbf{Total MSE} & \textbf{Hypervolume} \\",
        r"    \midrule"
    ]

    for pde_base in PDE_BASE_NAMES:
        for d in DIMS:
            if pde_base in ONLY_2D and d == 1:
                continue
            full_pde_name = f"{pde_base}_{d}D"
            sub = df[(df["pde"] == full_pde_name) & (df["method"] == "PI-NSGA-II")]
            if sub.empty: continue

            stats = sub.agg({
                "best_mse_domain":  ["mean", "std"],
                "best_mse_boundary":["mean", "std"],
                "mse_total":        ["mean", "std"],
                "hypervolume":      ["mean", "std"],
            })

            f_dom = fmt_sci_stat(stats.loc["mean","best_mse_domain"],  stats.loc["std","best_mse_domain"])
            f_bnd = fmt_sci_stat(stats.loc["mean","best_mse_boundary"],stats.loc["std","best_mse_boundary"])
            f_tot = fmt_sci_stat(stats.loc["mean","mse_total"],        stats.loc["std","mse_total"])
            f_hv  = f"{stats.loc['mean','hypervolume']:.4f}"

            lines.append(rf"    {pde_base} & {d}D & ${f_dom}$ & ${f_bnd}$ & ${f_tot}$ & {f_hv} \\")
            lines.append(r"    \midrule")

    if len(lines) > 9:
        lines[-1] = r"    \bottomrule"
    else:
        lines.append(r"    \bottomrule")

    lines += [r"  \end{tabular}", r"  }", r"\end{table*}"]
    out = os.path.join(TABLES_DIR, "symbolic_comparison.tex")
    with open(out, "w") as f: f.write("\n".join(lines) + "\n")
    print(f"  [OK] {out}")


# ─── Tabla 2: Comparación global RK4/FDM vs PINN vs PISR-NSGA-II ─────────────
def make_global_comparison_table():
    df = load_all()
    if df.empty:
        print("  [WARN] No data found for global comparison table.")
        return

    lines = [
        r"\begin{table*}[ht]",
        r"  \centering",
        r"  \caption{Performance Comparison: Numerical (RK4/FDM), Neural (PINN), and Symbolic (PISR-NSGA-II) Methods (Total MSE).}",
        r"  \label{tab:global_comp_stats}",
        r"  \resizebox{\textwidth}{!}{",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    \textbf{PDE} & \textbf{Dim} & \textbf{RK4/FDM} & \textbf{PINN} & \textbf{PISR-NSGA-II} \\",
        r"    \midrule"
    ]

    for pde_base in PDE_BASE_NAMES:
        for d in DIMS:
            if pde_base in ONLY_2D and d == 1:
                continue

            full_name = f"{pde_base}_{d}D"
            sub = df[df["pde"] == full_name]

            # ── PISR-NSGA-II ──
            pi_sub = sub[sub["method"] == "PI-NSGA-II"]
            if not pi_sub.empty:
                s = pi_sub.agg({"mse_total": ["mean","std"]})
                f_pi = fmt_sci_stat(s.loc["mean","mse_total"], s.loc["std","mse_total"])
            else:
                f_pi = "—"

            # ── RK4/FDM ──
            num_sub = sub[sub["method"] == "RK4/FDM"]
            if not num_sub.empty:
                s = num_sub.agg({"mse_total": ["mean","std"]})
                f_num = fmt_sci_stat(s.loc["mean","mse_total"], s.loc["std","mse_total"])
            else:
                f_num = "—"

            # ── PINN ──
            pinn_sub = sub[sub["method"] == "PINN"]
            if not pinn_sub.empty:
                # PINN puede tener mse_total_std ya calculado al promediar corridas
                m = pinn_sub["mse_total"].mean()
                s_col = "mse_total_std" if "mse_total_std" in pinn_sub.columns else "mse_total"
                s_val = pinn_sub[s_col].mean() if s_col == "mse_total_std" else pinn_sub["mse_total"].std()
                s_val = 0.0 if np.isnan(s_val) else s_val
                f_pinn = fmt_sci_stat(m, s_val)
            else:
                f_pinn = "—"

            # Formatear celdas con $…$ solo cuando no son «—»
            def wrap(v):
                return f"${v}$" if v != "—" else "---"

            lines.append(
                rf"    {pde_base} & {d}D & {wrap(f_num)} & {wrap(f_pinn)} & {wrap(f_pi)} \\"
            )
            lines.append(r"    \midrule")

    if len(lines) > 9:
        lines[-1] = r"    \bottomrule"
    else:
        lines.append(r"    \bottomrule")

    lines += [r"  \end{tabular}", r"  }", r"\end{table*}"]
    out = os.path.join(TABLES_DIR, "global_comparison.tex")
    with open(out, "w") as f: f.write("\n".join(lines) + "\n")
    print(f"  [OK] {out}")


# ─── Copiar figuras ───────────────────────────────────────────────────────────
def copy_figures():
    for pde in PDE_BASE_NAMES:
        for d in [1, 2]:
            for ext in ["pdf", "png"]:
                fname = f"solution_{d}d_{pde}.{ext}"
                src = os.path.join(RESULTS_DIR, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(FIGS_DIR, fname))


if __name__ == "__main__":
    print("Generando tablas LaTeX…")
    make_symbolic_table()
    make_global_comparison_table()
    copy_figures()
    print("\nReport generation complete. Tables in:", TABLES_DIR)
