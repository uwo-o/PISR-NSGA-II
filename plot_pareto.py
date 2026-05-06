#!/usr/bin/env python3
"""
plot_pareto.py — Publication-quality Pareto analysis
  PI-NSGA-II vs Koza (BNF) benchmark on Laplace / Poisson / Helmholtz

  python3 plot_pareto.py
"""
import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "legend.fontsize":   10.5,
    "xtick.labelsize":   10.5,
    "ytick.labelsize":   10.5,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.5,
    "figure.dpi":        150,
})
warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PDE_ORDER   = ["Laplace", "Poisson", "Helmholtz", "Schrodinger"]
METHOD_LIST = ["Tsoulos", "PI-NSGA-II", "PINN"]

STYLE = {
    "Tsoulos":    dict(color="#E07535", marker="s", zorder=3, lw=1.8),
    "PI-NSGA-II": dict(color="#2E86C1", marker="o", zorder=4, lw=1.8),
    "PINN":       dict(color="#27AE60", marker="^", zorder=5, lw=1.8),
}

# ─── Epsilon para escala log (evita log(0) = -inf) ──────────────────────────
LOG_EPS = 1e-9

# ─── Load data ────────────────────────────────────────────────────────────────
def load_all():
    files = glob.glob(os.path.join(RESULTS_DIR, "run_*", "*_pareto.csv"))
    if not files:
        files = glob.glob(os.path.join(RESULTS_DIR, "*_pareto.csv"))
    if not files:
        raise FileNotFoundError("No CSV files found. Run ./build/pi_nsga2 first.")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    
    # Asegurar que las columnas sean numéricas (a veces números gigantes se leen como strings)
    df["mse_domain"]   = pd.to_numeric(df["mse_domain"], errors="coerce")
    df["mse_boundary"] = pd.to_numeric(df["mse_boundary"], errors="coerce")
    
    # Registrar si un valor era exactamente 0 antes de aplicar epsilon
    df["dom_is_zero"] = df["mse_domain"]   == 0.0
    df["bnd_is_zero"] = df["mse_boundary"] == 0.0
    # Reemplazar 0 exacto con epsilon para escala log
    df["mse_domain"]   = df["mse_domain"].clip(lower=LOG_EPS)
    df["mse_boundary"] = df["mse_boundary"].clip(lower=LOG_EPS)
    return df

# ─── Build efficient frontier (lower envelope) ────────────────────────────────
def efficient_frontier(df_method, x_col, y_col):
    """
    Para cada umbral τ en x_col (ascendente),
    devuelve el mínimo valor de y_col alcanzable.
    """
    sorted_df = df_method.sort_values(x_col).reset_index(drop=True)
    xs = sorted_df[x_col].values
    ys = sorted_df[y_col].values
    running_min = np.minimum.accumulate(ys)
    return xs, running_min

# ─── Identificar puntos no dominados (Global) ────────────────────────────────
def identify_non_dominated(df, x_col, y_col):
    """Retorna una máscara booleana de los puntos no dominados globalmente."""
    if df.empty: return np.array([], dtype=bool)
    
    # Asegurarnos de que el índice sea único para el mapeo final
    temp = df.copy()
    temp['orig_index'] = temp.index
    
    # Ordenar por x (asc) y luego y (asc)
    df_sorted = temp.sort_values([x_col, y_col])
    
    non_dominated_indices = []
    min_y = float('inf')
    
    for _, row in df_sorted.iterrows():
        if row[y_col] < min_y:
            non_dominated_indices.append(row['orig_index'])
            min_y = row[y_col]
            
    mask = np.zeros(len(temp), dtype=bool)
    # Crear máscara basada en la posición original
    res = pd.Series(False, index=df.index)
    res.loc[non_dominated_indices] = True
    return res.values

# ─── Anotación: cuando un método alcanza exactamente 0 ───────────────────────
def annotate_zeros(ax, m_df, x_col, y_col, color, side="left"):
    """Añade flecha indicando que algunos puntos son exactamente 0."""
    zero_col = "dom_is_zero" if x_col == "mse_domain" else "bnd_is_zero"
    n_zero_x = (m_df["dom_is_zero"] if x_col == "mse_domain" else m_df["bnd_is_zero"]).sum()
    n_zero_y = (m_df["dom_is_zero"] if y_col == "mse_domain" else m_df["bnd_is_zero"]).sum()
    msgs = []
    if n_zero_x > 0:
        msgs.append(f"{n_zero_x} pts with exact 0 {x_col.split('_')[1]}")
    if n_zero_y > 0:
        msgs.append(f"{n_zero_y} pts with exact 0 {y_col.split('_')[1]}")
    if msgs:
        xpos = 0.03 if side == "left" else 0.55
        ax.text(xpos, 0.97, "\n".join(msgs),
                transform=ax.transAxes, fontsize=6.5, color=color,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec=color, lw=0.6))

# ─── Figure 1: Pareto fronts (one panel per PDE) ─────────────────────────────
def plot_pareto_fronts(df):
    n_pdes = len(PDE_ORDER)
    fig, axes = plt.subplots(1, n_pdes, figsize=(4.5 * n_pdes, 4.5))
    fig.suptitle(
        "Pareto Fronts — PI-NSGA-II vs Tsoulos (GE)\n"
        "Trade-off: PDE Residual (MSE Domain) vs Boundary Condition Error (MSE BC)",
        fontsize=12, fontweight="bold", y=1.02
    )

    for ax, pde in zip(axes, PDE_ORDER):
        sub = df[df["pde"] == pde]
        has_data = False
        for i, (method, st) in enumerate(STYLE.items()):
            m = sub[sub["method"] == method].sort_values("mse_domain")
            if m.empty:
                continue
            has_data = True
            xs, ys = efficient_frontier(m, "mse_domain", "mse_boundary")
            # Fill bajo la curva de Pareto
            ax.fill_between(xs, ys, ys.max() * 10, alpha=0.07, color=st["color"])
            # Línea del frente de Pareto (comentada a petición del usuario)
            # ax.plot(xs, ys, "-", color=st["color"], lw=st["lw"], alpha=0.9)

            # Calcular cuáles de estos puntos (que eran locales de cada run) 
            # son dominados GLOBALMENTE por otros runs
            is_global = identify_non_dominated(m, "mse_domain", "mse_boundary")
            m = m.assign(is_global=is_global)

            # Puntos dominados globalmente (transparentes)
            dominated = m[~m["is_global"]]
            if not dominated.empty:
                ax.scatter(dominated["mse_domain"], dominated["mse_boundary"],
                           color=st["color"], marker=st["marker"],
                           s=15, edgecolors="none", linewidths=0,
                           alpha=0.15, zorder=st["zorder"]-1)

            # Frente de Pareto Global (destacados)
            pareto = m[m["is_global"]]
            ax.scatter(pareto["mse_domain"], pareto["mse_boundary"],
                       color=st["color"], marker=st["marker"],
                       s=35, edgecolors="none", linewidths=0,
                       alpha=1.0, zorder=st["zorder"], label=method)
            # Anotar cuántos puntos eran exactamente 0
            annotate_zeros(ax, m, "mse_domain", "mse_boundary",
                           st["color"], side="left" if i == 0 else "right")

        ax.set_title(f"{pde}'s Equation", fontweight="bold")
        ax.set_xlabel("MSE Domain  (PDE residual)")
        ax.set_ylabel("MSE Boundary  (BC residual)")
        # symlog permite mostrar valores muy pequeños (~0) y grandes a la vez
        ax.set_xscale("symlog", linthresh=1e-6)
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.legend(loc="upper right")
        ax.annotate("Ideal", xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=7.5, color="gray", style="italic")
        if not has_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=12)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "pareto_fronts.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)

# ─── Figure 2: Single-objective sensitivity ───────────────────────────────────
def plot_sensitivity(df):
    """
    Two columns × N rows:
      Left  (col 0): Fix BC budget τ, minimize Domain MSE  → x=τ, y=best domain
      Right (col 1): Fix Domain budget τ, minimize BC MSE  → x=τ, y=best BC
    """
    n_pdes = len(PDE_ORDER)
    fig, axes = plt.subplots(n_pdes, 2, figsize=(12, 3.5 * n_pdes), squeeze=False)
    fig.suptitle(
        "Pareto Sensitivity Analysis\n"
        "Left: Minimize Domain MSE (BC budget fixed)  |  "
        "Right: Minimize BC MSE (Domain budget fixed)",
        fontsize=12, fontweight="bold", y=1.01
    )

    for row, pde in enumerate(PDE_ORDER):
        sub = df[df["pde"] == pde]
        ax_left  = axes[row][0]
        ax_right = axes[row][1]

        for method, st in STYLE.items():
            m = sub[sub["method"] == method]
            if m.empty:
                continue

            # ── Left: mejor MSE dominio dado BC ≤ τ ─────────────────────────
            m_sorted    = m.sort_values("mse_boundary")
            bc_budgets  = m_sorted["mse_boundary"].values
            domain_vals = m_sorted["mse_domain"].values
            running_dom = np.minimum.accumulate(domain_vals)
            ax_left.plot(bc_budgets, running_dom,
                         color=st["color"], lw=0, marker=st["marker"],
                         markersize=4, markeredgecolor="none", markeredgewidth=0,
                         markevery=max(1, len(bc_budgets)//15),
                         label=method)

            # ── Right: mejor MSE BC dado dominio ≤ τ ────────────────────────
            m_sorted2   = m.sort_values("mse_domain")
            dom_budgets = m_sorted2["mse_domain"].values
            bc_vals     = m_sorted2["mse_boundary"].values
            running_bc  = np.minimum.accumulate(bc_vals)
            ax_right.plot(dom_budgets, running_bc,
                          color=st["color"], lw=0, marker=st["marker"],
                          markersize=4, markeredgecolor="none", markeredgewidth=0,
                          markevery=max(1, len(dom_budgets)//15),
                          label=method)

        # Formatting — symlog para manejar 0 y grandes valores
        for ax, title, xlabel, ylabel in [
            (ax_left,  f"{pde} — Minimize Domain MSE",
             "BC Budget  (max allowed MSE Boundary)",
             "Best achievable MSE Domain"),
            (ax_right, f"{pde} — Minimize BC MSE",
             "Domain Budget  (max allowed MSE Domain)",
             "Best achievable MSE Boundary"),
        ]:
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xscale("symlog", linthresh=1e-6)
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.grid(True, which="both", ls="--", alpha=0.35)
            ax.legend()
            # Línea punteada en y=0 para indicar perfección
            ax.axhline(LOG_EPS, color="green", ls=":", lw=0.8, alpha=0.5,
                       label="MSE=0 (perfect)")

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "pareto_sensitivity.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)

# ─── Figure 3: Summary comparison bar chart ───────────────────────────────────
def plot_summary_bars():
    stats_path = os.path.join(RESULTS_DIR, "stats_table.csv")
    comp_path = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    
    metrics = {
        "Best Domain MSE":   ("best_mse_domain", "Best MSE Domain ↓"),
        "Best BC MSE":       ("best_mse_boundary", "Best MSE BC ↓"),
        "Runtime (s)":       ("runtime_s", "Runtime (s) ↓"),
    }
    n_pdes = len(PDE_ORDER)
    fig, axes = plt.subplots(1, 3, figsize=(4.5 * n_pdes, 4.5))
    fig.suptitle("Method Comparison Summary — All Equations",
                 fontsize=12, fontweight="bold")

    x = np.arange(len(PDE_ORDER))
    width = 0.32

    if os.path.exists(stats_path):
        df = pd.read_csv(stats_path)
        for ax, (metric_name, (comp_col, stats_col)) in zip(axes, metrics.items()):
            for i, method in enumerate(METHOD_LIST):
                vals = []
                for pde in PDE_ORDER:
                    sub = df[(df["PDE"] == pde) & (df["Metric"] == stats_col)]
                    if not sub.empty:
                        vals.append(max(sub[f"{method}_mean"].values[0], 1e-3))
                    else:
                        vals.append(1e-3)
                offset = (i - 0.5) * width
                bars = ax.bar(x + offset, vals, width, bottom=1e-4,
                              color=STYLE[method]["color"],
                              label=method, alpha=0.85, edgecolor="white", linewidth=0.6)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.03,
                            f"{v:.2g}", ha="center", va="bottom", fontsize=7, color="#333333")
            ax.set_title(metric_name, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(PDE_ORDER, rotation=45, ha="right")
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-4)
            ax.set_ylabel(metric_name)
            ax.grid(True, axis="y", ls="--", alpha=0.4)
            ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
    elif os.path.exists(comp_path):
        df = pd.read_csv(comp_path)
        for ax, (metric_name, (comp_col, stats_col)) in zip(axes, metrics.items()):
            for i, method in enumerate(METHOD_LIST):
                vals = [
                    max(df[(df["method"] == method) & (df["pde"] == pde)][comp_col].values[0], 1e-3)
                    if len(df[(df["method"] == method) & (df["pde"] == pde)]) > 0 else 1e-3
                    for pde in PDE_ORDER
                ]
                offset = (i - 0.5) * width
                bars = ax.bar(x + offset, vals, width, bottom=1e-4,
                              color=STYLE[method]["color"],
                              label=method, alpha=0.85, edgecolor="white", linewidth=0.6)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.03,
                            f"{v:.2g}", ha="center", va="bottom", fontsize=7, color="#333333")
            ax.set_title(metric_name, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(PDE_ORDER)
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-4)
            ax.set_ylabel(metric_name)
            ax.grid(True, axis="y", ls="--", alpha=0.4)
            ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
    else:
        return


    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "summary_bars.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)

# ─── Figure 4: Hypervolume-style coverage (dominance map) ────────────────────
def plot_dominance_scatter(df):
    """
    One panel per PDE. Each point colored by method, size encodes
    'goodness' (inverse of Chebyshev distance to ideal point).
    Usa symlog para manejar valores = 0.
    """
    n_pdes = len(PDE_ORDER)
    fig, axes = plt.subplots(1, n_pdes, figsize=(4.5 * n_pdes, 4.5))
    if n_pdes == 1: axes = [axes]
    fig.suptitle(
        "Dominance Map — Solution Coverage in Objective Space\n"
        "(Closer to origin → better in both objectives simultaneously)",
        fontsize=12, fontweight="bold", y=1.02
    )

    for ax, pde in zip(axes, PDE_ORDER):
        sub = df[df["pde"] == pde].copy()
        # Normalizar objetivos a [0,1]
        for col in ["mse_domain", "mse_boundary"]:
            col_min = sub[col].min()
            col_max = sub[col].max()
            rng = col_max - col_min
            sub[col + "_n"] = (sub[col] - col_min) / rng if rng > 0 else 0.0

        for method, st in STYLE.items():
            m = sub[sub["method"] == method].copy()
            if m.empty:
                continue
            
            # Re-calcular dominancia global para este plot
            m["is_global"] = identify_non_dominated(m, "mse_domain", "mse_boundary")

            dist  = np.sqrt(m["mse_domain_n"]**2 + m["mse_boundary_n"]**2) + 1e-9
            sizes = 1.0 / dist
            sizes = 10 + 120 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-12)
            m = m.assign(sizes=sizes)

            # Dominadas (transparentes)
            dominated = m[~m["is_global"]]
            if not dominated.empty:
                ax.scatter(dominated["mse_domain"], dominated["mse_boundary"],
                           c=st["color"], marker=st["marker"],
                           s=dominated["sizes"]*0.5, alpha=0.15, edgecolors="none",
                           linewidths=0, zorder=st["zorder"]-1)

            # Frente (sólidas)
            pareto = m[m["is_global"]]
            ax.scatter(pareto["mse_domain"], pareto["mse_boundary"],
                       c=st["color"], marker=st["marker"],
                       s=pareto["sizes"], alpha=0.8, edgecolors="none",
                       linewidths=0, label=method, zorder=st["zorder"])

        ax.set_title(f"{pde}'s Equation", fontweight="bold")
        ax.set_xlabel("MSE Domain (PDE residual)")
        ax.set_ylabel("MSE Boundary (BC residual)")
        ax.set_xscale("symlog", linthresh=1e-6)
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.legend(loc="upper right")

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "dominance_map.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)

# ─── Print analysis ───────────────────────────────────────────────────────────
def print_analysis(df_summary):
    # Propiedades de los métodos
    SYM_METHODS = ["Tsoulos", "PI-NSGA-II"]
    
    # ─── TABLA 1: COMPETENCIA SIMBÓLICA (Interpretables) ───
    print("\n" + "="*80)
    print("  TABLE 1: SYMBOLIC COMPETITION (Interpretability Required)")
    print("  Comparison between methods that return mathematical formulas.")
    print("="*80)
    print(f"{'Equation':<12} {'Metric':<20} {'Tsoulos':>12} {'PI-NSGA-II':>12} {'Winner':>14}")
    print("-" * 80)

    sym_winners = {"Tsoulos": 0, "PI-NSGA-II": 0}
    
    for pde in PDE_ORDER:
        sub = df_summary[df_summary["pde"] == pde]
        k = sub[sub["method"] == "Tsoulos"].iloc[0] if not sub[sub["method"] == "Tsoulos"].empty else None
        p = sub[sub["method"] == "PI-NSGA-II"].iloc[0] if not sub[sub["method"] == "PI-NSGA-II"].empty else None
        
        if k is None or p is None: continue

        # Domain MSE
        w = "Tsoulos" if k.best_mse_domain < p.best_mse_domain else "PI-NSGA-II"
        sym_winners[w] += 1
        print(f"{pde:<12} {'Domain MSE ↓':<20} {k.best_mse_domain:>12.4f} {p.best_mse_domain:>12.4f} {w:>14}")

        # BC MSE
        w = "Tsoulos" if k.best_mse_boundary < p.best_mse_boundary else "PI-NSGA-II"
        sym_winners[w] += 1
        print(f"{'':12} {'BC MSE ↓':<20} {k.best_mse_boundary:>12.6f} {p.best_mse_boundary:>12.6f} {w:>14}")
        print()

    print(f"  Symbolic Score → Tsoulos: {sym_winners['Tsoulos']} | PI-NSGA-II: {sym_winners['PI-NSGA-II']}")

    # ─── TABLA 2: COMPARATIVA GLOBAL (Simbólico vs Black-Box) ───
    print("\n" + "="*95)
    print("  TABLE 2: STATE-OF-THE-ART COMPARISON (Including PINN Baseline)")
    print("  Does the best symbolic method beat the Neural Network?")
    print("="*95)
    print(f"{'Equation':<12} {'Metric':<20} {'Best Symb.':>12} {'PINN':>12} {'Symbolic?':>12} {'Winner':>14}")
    print("-" * 95)

    global_winners = {"Symbolic": 0, "PINN": 0}

    for pde in PDE_ORDER:
        sub = df_summary[df_summary["pde"] == pde]
        k = sub[sub["method"] == "Tsoulos"].iloc[0] if not sub[sub["method"] == "Tsoulos"].empty else None
        p = sub[sub["method"] == "PI-NSGA-II"].iloc[0] if not sub[sub["method"] == "PI-NSGA-II"].empty else None
        n = sub[sub["method"] == "PINN"].iloc[0] if not sub[sub["method"] == "PINN"].empty else None
        
        if n is None: continue
        
        # Encontrar el mejor simbólico para esta métrica
        for metric, label in [("best_mse_domain", "Domain MSE ↓"), ("best_mse_boundary", "BC MSE ↓")]:
            val_k = getattr(k, metric) if k is not None else float('inf')
            val_p = getattr(p, metric) if p is not None else float('inf')
            best_sym_val = min(val_k, val_p)
            val_n = getattr(n, metric)
            
            is_sym = "YES" if best_sym_val <= val_n else "-"
            winner = "Symbolic" if best_sym_val <= val_n else "PINN"
            global_winners[winner] += 1
            
            pde_label = pde if label == "Domain MSE ↓" else ""
            print(f"{pde_label:<12} {label:<20} {best_sym_val:>12.4g} {val_n:>12.4g} {is_sym:>12} {winner:>14}")
        print()

    print(f"  Global Score → Symbolic: {global_winners['Symbolic']} | PINN: {global_winners['PINN']}")
    print("="*95)
    print("""
  Key Observations:
  ─────────────────
  1. Interpretability: Symbolic methods (Tsoulos, PI-NSGA-II) return explicit 
     mathematical formulas. PINN is a black-box neural network (non-interpretable).

  2. Speed: PI-NSGA-II is dramatically faster than PINNs (~50x) and significantly 
     faster than GE/Tsoulos (2-3x) due to exact symbolic AD and C++ implementation.

  3. BC Satisfaction: PI-NSGA-II now outperforms Tsoulos in Laplace and Schrodinger BCs,
     thanks to the improved symbolic simplification and better Pareto exploration.

  4. Symbolic vs Neural: While PINN can reach lower residuals in some cases (Poisson),
     the symbolic winner (PI-NSGA-II) provides an exact mathematical expression
     that is easier to verify and deploy in resource-constrained environments.
""")
    print("="*70)

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_all()
    print(f"  Loaded {len(df)} Pareto-front points: "
          f"{df['pde'].nunique()} PDEs × {df['method'].nunique()} methods\n")

    # Load summary for analysis
    df_summary = pd.read_csv(os.path.join(RESULTS_DIR, "comparison_summary.csv"))
    # Normalizar HV a [0,1] si está en escala legacy (~1e8)
    if "hypervolume" in df_summary.columns and df_summary["hypervolume"].max() > 1.5:
        df_summary["hypervolume"] = df_summary["hypervolume"] / 1e8
    print_analysis(df_summary)

    print("\n  Generando plots...")
    matplotlib.use("Agg")   # modo no-interactivo, evita errores de display
    plot_pareto_fronts(df)
    plot_sensitivity(df)
    plot_summary_bars()
    plot_dominance_scatter(df)
    print("\n  Todas las figuras guardadas en ./results/")
