#!/usr/bin/env python3
"""
report/generate_report.py — Tablas LaTeX y documento report/results.tex.

Todo se ordena y agrupa segun report/benchmarks.py (misma suite y orden que
benchmark_suite() en src/pde_problems.cpp).

Fuentes:
  PISR-NSGA-II y RK4/FDM : results/all_runs_summary.csv (scripts/consolidate_runs.py)
  PINN, PINN+PySR        : results/**/<EDP>_pinn.csv, <EDP>_pinn_pysr.csv (run_competitors.sh)
Las lineas base externas solo se usan si son posteriores a la redefinicion de
los problemas (benchmarks.is_fresh): antes resolvian otro problema.
"""
import datetime
import glob
import math
import os
import shutil

import numpy as np
import pandas as pd

from benchmarks import BENCHMARKS, BY_LABEL, PHYSICS_NOTES, PROBLEMS_REDEFINED, by_family, is_fresh
from generate_formulas_table import pisr_formula

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(REPORT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TABLES_DIR = os.path.join(REPORT_DIR, "tables")
FIGS_DIR = os.path.join(REPORT_DIR, "figures")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

PISR = "PISR-NSGA-II"
NUM = "RK4/FDM"
GOOD, FAIR = 1e-6, 1e-3   # umbrales de MSE de solucion para el semaforo


# ─── Datos ────────────────────────────────────────────────────────────────────
def load_summary() -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in df.columns:
        if c not in ("method", "pde"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df["solution_mse"] < 0, "solution_mse"] = np.nan
    return df


def _fresh(files):
    files = sorted(set(files))
    return [f for f in files if is_fresh(f)], [f for f in files if not is_fresh(f)]


def load_external():
    """(DataFrame con method, pde, solution_mse, runtime_s, complexity; n archivos viejos descartados).

    Competidores (run_competitors.sh): PINN y PINN+PySR, con la misma referencia
    y los mismos puntos que solution_mse de PISR-NSGA-II. Los archivos de las
    lineas base antiguas (DeepXDE, PySR/PySINDy sobre la grilla de PISR) solo
    se cuentan, para avisar que se descartaron."""
    cols = ["method", "pde", "solution_mse", "runtime_s", "complexity"]
    frames = []
    for pat in ("*_pinn.csv", "*_pinn_pysr.csv"):
        fresh, _ = _fresh(glob.glob(os.path.join(RESULTS_DIR, "**", pat), recursive=True))
        for f in fresh:
            t = pd.read_csv(f)
            for c in cols[2:]:
                t[c] = pd.to_numeric(t[c], errors="coerce") if c in t.columns else np.nan
            frames.append(t[cols])
    stale = sum(len(_fresh(glob.glob(os.path.join(RESULTS_DIR, "**", pat), recursive=True))[1])
                for pat in ("*_pinn_pareto.csv", "*_metrics.csv"))
    ext = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    ext.loc[ext["solution_mse"] < 0, "solution_mse"] = np.nan
    return ext, stale


def stats(df, method, label, col):
    """(media, desviacion, n) de `col` sobre las corridas."""
    if df.empty or col not in df.columns:
        return math.nan, math.nan, 0
    v = df[(df["method"] == method) & (df["pde"] == label)][col].dropna()
    if v.empty:
        return math.nan, math.nan, 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else math.nan, len(v)


# ─── Formato ──────────────────────────────────────────────────────────────────
def sci(m, s=math.nan, bold=False):
    """$a\\times10^{e}$ o $(a\\pm b)\\times10^{e}$; '---' si no hay dato."""
    if m is None or not np.isfinite(m):
        return r"\textemdash"
    if m == 0:
        body = "0"
    else:
        e = int(math.floor(math.log10(abs(m))))
        a = m / 10 ** e
        if np.isfinite(s):
            body = rf"({a:.2f}\pm{s / 10 ** e:.2f})\times10^{{{e}}}"
        else:
            body = rf"{a:.2f}\times10^{{{e}}}"
    if bold:
        body = rf"\mathbf{{{body}}}"
    return f"${body}$"


def seconds(t):
    if t is None or not np.isfinite(t):
        return r"\textemdash"
    if t < 1e-2:
        return rf"${t * 1e3:.2f}$\,ms"
    if t < 120:
        return rf"${t:.2f}$\,s"
    return rf"${t / 60:.1f}$\,min"


def status(m):
    if m is None or not np.isfinite(m):
        return r"\stnone"
    return r"\stgood" if m < GOOD else (r"\stfair" if m < FAIR else r"\stpoor")


def family_row(title, ncols):
    return rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\familyrow{{{title}}}}} \\"


def write(name, lines):
    out = os.path.join(TABLES_DIR, name)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [OK] {os.path.relpath(out, ROOT_DIR)}")


# ─── Tablas ───────────────────────────────────────────────────────────────────
def make_suite_table():
    rows = []
    for _, title, items in by_family():
        rows.append(family_row(title, 4))
        for b in items:
            rows.append(rf"{b.label_tex} & ${b.equation}$ & {b.conditions} & ${b.reference}$ \\")
        rows.append(r"\addlinespace[2pt]")
    write("suite_table.tex", [
        r"\begin{longtable}{@{}L{0.21\linewidth}L{0.34\linewidth}L{0.21\linewidth}L{0.19\linewidth}@{}}",
        r"\caption{Benchmark suite. Every problem is posed on $[0,1]^d$ (and $t\in[0,1]$ when unsteady). "
        r"Problems marked \emph{numerical} have no closed form; their reference is a high-resolution "
        r"numerical solution of exactly the same boundary-value problem.}\label{tab:suite}\\",
        r"\toprule \textbf{Problem} & \textbf{Equation} & \textbf{Conditions} & \textbf{Reference} \\ \midrule",
        r"\endfirsthead",
        r"\toprule \textbf{Problem} & \textbf{Equation} & \textbf{Conditions} & \textbf{Reference} \\ \midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
        *rows,
        r"\end{longtable}",
    ])


def make_results_table(df):
    n_runs = int(df["run"].nunique()) if not df.empty else 0
    metrics = [("solution_mse", "Solution MSE", "sci"), ("best_mse_domain", "Domain residual", "sci"),
               ("best_mse_boundary", "Boundary MSE", "sci"), ("pareto_size", "Front size", "int"),
               ("runtime_s", "Time", "time")]
    # columnas sin ningun dato (p. ej. resumen reconstruido desde logs) se omiten
    sub = df[df["method"] == PISR] if not df.empty else df
    metrics = [m for m in metrics if not sub.empty and m[0] in sub.columns and sub[m[0]].notna().any()]
    rows = []
    for _, title, items in by_family():
        rows.append(family_row(title, 1 + len(metrics)))
        for b in items:
            cells = []
            for col, _, kind in metrics:
                m, sd, _ = stats(df, PISR, b.label, col)
                if kind == "sci":
                    cells.append(sci(m, sd))
                elif kind == "int":
                    cells.append(r"\textemdash" if not np.isfinite(m) else f"{m:.0f}")
                else:
                    cells.append(seconds(m))
            sol = stats(df, PISR, b.label, "solution_mse")[0]
            rows.append(rf"{status(sol)}\ {b.label_tex} & " + " & ".join(cells) + r" \\")
    runs_txt = "one run" if n_runs == 1 else f"mean $\\pm$ standard deviation over {n_runs} independent runs"
    spec = "l" + "c" * (len(metrics) - 1) + "r"
    head = " & ".join(rf"\textbf{{{h}}}" for _, h, _ in metrics)
    write("results_table.tex", [
        r"\begin{table}[H]",
        r"\centering\small",
        rf"\caption{{PISR-NSGA-II results ({runs_txt}). \emph{{Solution MSE}} is measured against the "
        r"reference after the search; the search itself only uses the PDE residual and the boundary data. "
        r"Status: \stgood\ $<10^{-6}$, \stfair\ $<10^{-3}$, \stpoor\ $\geq10^{-3}$.}",
        r"\label{tab:results}",
        r"\begin{adjustbox}{max width=\linewidth}",
        rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}",
        r"\toprule",
        rf"\textbf{{Problem}} & {head} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}",
    ])


def make_comparison_table(df, ext):
    methods = [NUM] + sorted(ext["method"].unique().tolist()) + [PISR]
    src = pd.concat([df[["method", "pde", "solution_mse", "runtime_s"]], ext], ignore_index=True) if not df.empty else ext
    k = len(methods)
    head = " & ".join(rf"\textbf{{{m}}}" for m in methods)
    rows = []
    for _, title, items in by_family():
        rows.append(family_row(title, 1 + 2 * k))
        for b in items:
            mses = [stats(src, m, b.label, "solution_mse") for m in methods]
            times = [stats(src, m, b.label, "runtime_s") for m in methods]
            finite = [m[0] for m in mses if np.isfinite(m[0])]
            best = min(finite) if finite else None
            c_mse = [sci(m[0], m[1], bold=(best is not None and m[0] == best and len(finite) > 1)) for m in mses]
            c_t = [seconds(t[0]) for t in times]
            rows.append(rf"{b.label_tex} & " + " & ".join(c_mse) + " & " + " & ".join(c_t) + r" \\")
    write("comparison_table.tex", [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{Solution MSE against the reference and wall-clock time. RK4/FDM is the numerical solver "
        r"of the same problem at low resolution (50 nodes per axis); it returns a grid, not a formula. "
        r"PINN is a physics-informed neural network trained on the same residual and boundary data "
        r"(no reference data); it returns network weights, not a formula. PINN+PySR distils the PINN "
        r"prediction into a formula with PySR, the data-free interpretable alternative (its time includes "
        r"training the PINN). All methods are "
        r"scored on the same points against the same reference; mean $\pm$ standard deviation over seeds. "
        r"Best MSE per row in bold.}",
        r"\label{tab:comparison}",
        r"\begin{adjustbox}{max width=\linewidth}",
        rf"\begin{{tabular}}{{@{{}}l{'c' * k}{'r' * k}@{{}}}}",
        r"\toprule",
        rf" & \multicolumn{{{k}}}{{c}}{{\textbf{{Solution MSE}}}} & \multicolumn{{{k}}}{{c}}{{\textbf{{Time}}}} \\",
        rf"\cmidrule(lr){{2-{1 + k}}}\cmidrule(l){{{2 + k}-{1 + 2 * k}}}",
        rf"\textbf{{Problem}} & {head} & {head} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}",
    ])


# ─── Documento ────────────────────────────────────────────────────────────────
PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage[margin=2.2cm,headheight=14pt]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,longtable,adjustbox,array,float}
\usepackage{graphicx}
\usepackage[table]{xcolor}
\usepackage[font=small,labelfont=bf,skip=6pt]{caption}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage[colorlinks=true,linkcolor=accent,urlcolor=accent,citecolor=accent]{hyperref}
\graphicspath{{figures/}}

\definecolor{accent}{HTML}{1F4E79}
\definecolor{muted}{HTML}{6B7280}
\definecolor{good}{HTML}{1E8449}
\definecolor{fair}{HTML}{D68910}
\definecolor{poor}{HTML}{C0392B}
\newcommand{\stgood}{\textcolor{good}{\textbullet}}
\newcommand{\stfair}{\textcolor{fair}{\textbullet}}
\newcommand{\stpoor}{\textcolor{poor}{\textbullet}}
\newcommand{\stnone}{\textcolor{muted}{\textbullet}}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcommand{\familyrow}[1]{\rule{0pt}{3ex}\textcolor{accent}{\textbf{\textsc{#1}}}}

\titleformat{\section}{\Large\bfseries\color{accent}}{\thesection}{0.8em}{}
\titleformat{\subsection}{\large\bfseries\color{accent}}{\thesubsection}{0.8em}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{\thesubsubsection}{0.8em}{}
\setlength{\LTcapwidth}{\linewidth}
\renewcommand{\arraystretch}{1.15}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textcolor{muted}{PISR-NSGA-II --- Benchmark report}}
\fancyhead[R]{\small\textcolor{muted}{\nouppercase{\leftmark}}}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.3pt}
"""


GLOSSARY = r"""\section{Notation and glossary}
\subsection*{Solution representation}
\begin{description}[leftmargin=1.2em,itemsep=3pt,font=\normalfont\bfseries]
\item[$\hat u$] The discovered closed-form solution: the hall-of-fame champion, after constant polishing and
  symbolic simplification. Real-valued problems report $\operatorname{Re}(\hat u)$, which is exactly the
  quantity optimised (complex constants are only a representation device).
\item[$N$] The expression tree evolved by the algorithm when the boundary ansatz is used; $\hat u$ is built from it.
\item[Boundary ansatz] Construction that satisfies the boundary data \emph{exactly} for any $N$:
  \begin{itemize}[leftmargin=1.2em,itemsep=1pt]
  \item 1D Dirichlet: $\hat u = g(0) + x\,[g(1)-g(0)] + x(x-1)\,N(x)$.
  \item 1D initial-value problem: $\hat u = u_0 + u_0'\,x + x^2 N(x)$.
  \item 2D Dirichlet: $\hat u = \mathcal{C}[g](x,y) + x(1-x)\,y(1-y)\,N(x,y)$.
  \end{itemize}
  The factor multiplying $N$ vanishes on the boundary, so $\hat u = g$ there by construction.
\item[$g$] Boundary data of the problem (the conditions listed in Table~\ref{tab:suite}).
\item[{$\mathcal{C}[g]$}] Transfinite (Coons) interpolation of the boundary data on $[0,1]^2$: the surface that
  matches $g$ on all four edges,
  \[
  \begin{aligned}
  \mathcal{C}[g](x,y) ={}& (1-x)\,g(0,y) + x\,g(1,y) + (1-y)\,g(x,0) + y\,g(x,1)\\
  &- \big[(1-x)(1-y)\,g(0,0) + x(1-y)\,g(1,0) + (1-x)y\,g(0,1) + xy\,g(1,1)\big].
  \end{aligned}
  \]
  It contains no learned information; all of the learning is in $N$.
\item[Direct representation] Individuals whose tree is $\hat u$ itself (no ansatz); the boundary error is then an
  objective. Both representations compete in the same population (hybrid search).
\item[$\mathrm{Ai}$, $L_n$, $P_n$, $H_n$, $T_n$] Airy function; Laguerre, Legendre, Hermite and Chebyshev polynomials of degree $n$.
\end{description}

\subsection*{Metrics}
\begin{description}[leftmargin=1.2em,itemsep=3pt,font=\normalfont\bfseries]
\item[Solution MSE] Mean squared difference between $\hat u$ and the reference solution on a validation grid.
  Computed \emph{after} the search, for evaluation only.
\item[Domain residual] Mean squared PDE residual of $\hat u$ at the collocation points (the physics objective).
\item[Boundary MSE] Mean squared mismatch with the boundary data; identically $0$ under the ansatz.
\item[Validation score] Domain residual plus boundary error on a fixed validation grid. The only criterion used
  to choose the champion and to stop early; it never uses the reference solution.
\item[Front size] Number of non-dominated individuals (rank~1) in the final population.
\item[Grid MSE, $\max|\text{error}|$] Error of the champion on the exported plotting grid (figure annotations).
\item[Status] \stgood\ solution MSE $<10^{-6}$, \stfair\ $<10^{-3}$, \stpoor\ $\geq 10^{-3}$, \stnone\ no result.
\end{description}

\subsection*{Methods}
\begin{description}[leftmargin=1.2em,itemsep=3pt,font=\normalfont\bfseries]
\item[PISR-NSGA-II] Physics-informed symbolic regression with NSGA-II over three objectives: domain residual,
  boundary error and expression size.
\item[RK4/FDM] Numerical solver of the same problem at low resolution (50 nodes per axis): Runge--Kutta shooting
  in 1D, finite differences in 2D. It returns values on a grid, not a formula.
\item[Reference] Closed-form solution when one exists; otherwise a high-resolution numerical solution of exactly
  the same problem, marked \emph{numerical}.
\item[Hall of fame] Best individual seen during the whole run according to the validation score.
\end{description}

\subsection*{Simplification of the reported expressions}
Each champion is simplified symbolically (SymPy). Constants are shown with 4 significant figures and are
replaced by exact values ($2$, $\tfrac12$, $\pi$, $\sqrt2$, \dots) only when the resulting expression stays within
$10^{-3}$ (relative) of the champion on its whole export grid; e.g.\ a constant reported as $\pi$ agrees with $\pi$
to that tolerance, it is not claimed to be exactly $\pi$.
"""


def figure(path, caption, width=r"\linewidth", place="htbp"):
    return [rf"\begin{{figure}}[{place}]", r"\centering", rf"\includegraphics[width={width},height=0.80\textheight,keepaspectratio]{{{path}}}",
            rf"\caption{{{caption}}}", r"\end{figure}"]


def overview(df, ext, stale):
    n_runs = int(df["run"].nunique()) if not df.empty else 0
    sols = {b.label: stats(df, PISR, b.label, "solution_mse")[0] for b in BENCHMARKS}
    have = [v for v in sols.values() if np.isfinite(v)]
    n_good = sum(v < GOOD for v in have)
    n_fair = sum(GOOD <= v < FAIR for v in have)
    n_poor = sum(v >= FAIR for v in have)
    total_t = df[df["method"] == PISR].groupby("run")["runtime_s"].sum().mean() if not df.empty else math.nan
    poor = [BY_LABEL[k].label_tex for k, v in sols.items() if np.isfinite(v) and v >= FAIR]
    missing = [b.label_tex for b in BENCHMARKS if not np.isfinite(sols[b.label])]

    L = [r"\section{Overview}",
         r"PISR-NSGA-II searches for closed-form solutions of differential equations using only the "
         r"equation residual and the boundary data; the reference solution is never seen during the search "
         r"and is used only to score the result afterwards.",
         r"\paragraph{Goal.} Help physicists understand how differential equations \emph{without an analytic "
         r"solution} behave. A numerical solver returns numbers on a grid; PISR-NSGA-II returns a short formula "
         r"that can be read, differentiated, analysed for limits and asymptotics, and evaluated anywhere. The "
         r"method deliberately trades some precision for interpretability: it is not meant to beat numerical "
         r"solvers on accuracy, but to explain the solution. When a closed form exists the algorithm should "
         r"recover it; when it does not, it should produce a compact expression that is accurate inside the "
         r"domain and retains the structure of the equation (Section~\ref{sec:extrapolation}).",
         r"\begin{itemize}[leftmargin=1.5em,itemsep=2pt]",
         rf"\item \textbf{{{len(BENCHMARKS)}}} benchmark problems in {len(by_family())} families "
         rf"(Table~\ref{{tab:suite}}); \textbf{{{n_runs}}} run{'s' if n_runs != 1 else ''} per problem.",
         *([r"\item Accuracy tables and convergence curves aggregate all seeds. Anything that needs a single "
            r"formula (discovered expressions, per-problem figures, extrapolation, physical consistency) uses the "
            r"\emph{median seed} of each problem: the seed whose solution error ranks in the middle, i.e.\ the "
            r"typical outcome, neither the best run nor an arbitrary one."] if n_runs > 1 else []),
         rf"\item Solution MSE: \stgood\ \textbf{{{n_good}}} below $10^{{-6}}$, "
         rf"\stfair\ \textbf{{{n_fair}}} below $10^{{-3}}$, \stpoor\ \textbf{{{n_poor}}} at or above $10^{{-3}}$.",
         ]
    if np.isfinite(total_t):
        L.append(rf"\item Total search time per run: {seconds(total_t)}.")
    if poor:
        L.append(r"\item Problems not yet solved to $10^{-3}$: " + ", ".join(poor) + ".")
    if missing:
        L.append(r"\item Problems without results: " + ", ".join(missing) + ".")
    L.append(r"\end{itemize}")
    if stale and ext.empty:
        L.append(r"\paragraph{External baselines.} "
                 rf"{stale} result files from the old DeepXDE, PySR and PySINDy baselines were found, but "
                 rf"they were produced before the problems were redefined ({PROBLEMS_REDEFINED:%Y-%m-%d}) and "
                 r"solve different equations; they are excluded. Only the numerical solver (RK4/FDM) is "
                 r"compared until \texttt{run\_competitors.sh} (PINN and PINN+PySR) has been run.")
    return L


def per_problem():
    L = [r"\clearpage", r"\section{Per-problem analysis}",
         r"For each problem: the champion against the reference (top) and the Pareto trade-offs of the "
         r"final population (bottom). Unsteady problems are shown at $t=0.5$; complex-valued ones show "
         r"real and imaginary parts."]
    for _, title, items in by_family():
        L.append(rf"\subsection{{{title}}}")
        for b in items:
            sol, pairs = f"solution_{b.label}.pdf", f"pareto_pairs_{b.label}.pdf"
            if not os.path.exists(os.path.join(FIGS_DIR, sol)):
                continue
            L += [r"\clearpage", rf"\subsubsection{{{b.label_tex}}}",
                  r"\begin{center}\small\begin{tabular}{@{}ll@{}}",
                  rf"\textcolor{{muted}}{{Equation}} & ${b.equation}$ \\",
                  rf"\textcolor{{muted}}{{Conditions}} & {b.conditions} \\",
                  rf"\textcolor{{muted}}{{Reference}} & ${b.reference}$ \\",
                  *([rf"\textcolor{{muted}}{{Discovered}} & \adjustbox{{max width=0.75\linewidth}}{{$\hat u = {f}$}} \\"]
                    if (f := pisr_formula(b)) else []),
                  r"\end{tabular}\end{center}"]
            L += figure(sol, rf"{b.label_tex}: PISR-NSGA-II champion vs.\ reference.", r"0.8\linewidth", "H")
            if os.path.exists(os.path.join(FIGS_DIR, pairs)):
                L += figure(pairs, rf"{b.label_tex}: pairwise Pareto trade-offs of the final population.", r"\linewidth", "H")
    return L


def ablation_section(df):
    """Representacion hibrida (por defecto) vs. solo ansatz (--no-hybrid)."""
    path = os.path.join(RESULTS_DIR, "ablation_no_hybrid", "comparison_summary.csv")
    L = [r"\subsection{Hybrid representation vs.\ ansatz only}",
         r"By default both representations compete in the same population: trees that are the solution itself "
         r"(\emph{direct}) and trees $N$ wrapped by the boundary ansatz. This widens the search (some solutions are "
         r"much shorter in one representation than in the other) but it can also mislead it: a direct tree can "
         r"satisfy the equation in the interior while missing the boundary, and compete with good ansatz trees. "
         r"The ansatz-only variant (\texttt{--no-hybrid}) is kept as an option so each problem can be checked "
         r"with both."]
    if not os.path.exists(path):
        L.append(r"No ansatz-only runs are available yet.")
        return L
    ab = pd.read_csv(path)
    ab = ab[ab["method"] == PISR]
    rows = []
    for b in BENCHMARKS:
        a = ab[ab["pde"] == b.label]
        if a.empty:
            continue
        h = stats(df, PISR, b.label, "solution_mse")[0]
        n = float(pd.to_numeric(a["solution_mse"], errors="coerce").mean())
        th = stats(df, PISR, b.label, "runtime_s")[0]
        tn = float(pd.to_numeric(a["runtime_s"], errors="coerce").mean())
        rows.append(rf"{b.label_tex} & {sci(h, bold=np.isfinite(h) and h <= n)} & {sci(n, bold=np.isfinite(n) and n < h)} & "
                    rf"{seconds(th)} & {seconds(tn)} \\")
    if rows:
        L += [r"\begin{table}[H]", r"\centering\small",
              r"\caption{Solution MSE with the hybrid representation (default) and with the ansatz only "
              r"(\texttt{--no-hybrid}). Better value in bold.}", r"\label{tab:ablation_hybrid}",
              r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
              r" & \multicolumn{2}{c}{\textbf{Solution MSE}} & \multicolumn{2}{c}{\textbf{Time}} \\",
              r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
              r"\textbf{Problem} & Hybrid & Ansatz only & Hybrid & Ansatz only \\", r"\midrule",
              *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return L


def extrapolation_section():
    if not os.path.exists(os.path.join(TABLES_DIR, "extrapolation_table.tex")):
        return []
    L = [r"\clearpage", r"\section{Extrapolation beyond the training domain}\label{sec:extrapolation}",
         r"A formula that encodes the structure of the solution should remain meaningful outside the region it "
         r"was found in; an interpolant of data does not. Every 1D steady problem is posed on $[0,1]$. Here the "
         r"champion \emph{as found} (before rounding constants to exact values) is evaluated on $[0,1.5]$ and "
         r"compared with the reference: the closed form when it exists, otherwise the numerical solution of the "
         r"same problem continued past $x=1$ by integrating the equation from $u(1),u'(1)$.",
         r"The comparison is a Chebyshev polynomial fitted to the reference on $[0,1]$ with the lowest degree "
         r"that is as accurate as PISR-NSGA-II inside the domain. It stands for any surrogate that only "
         r"interpolates values (a numerical grid plus an interpolant, a neural network, a polynomial "
         r"regression): equally good inside, with no knowledge of the equation outside. The RK4/FDM grid itself "
         r"cannot be evaluated outside $[0,1]$ at all.",
         r"\input{tables/extrapolation_table.tex}"]
    for fig, cap in [("extrapolation_solutions.pdf", r"Champion (dashed), matched-accuracy Chebyshev polynomial "
                      r"(dotted) and reference (grey) on $[0,1.5]$; shaded region: outside the training domain."),
                     ("extrapolation_errors.pdf", r"Pointwise error of the champion and of the matched-accuracy "
                      r"Chebyshev polynomial on $[0,1.5]$.")]:
        if os.path.exists(os.path.join(FIGS_DIR, fig)):
            L += figure(fig, cap, place="H")
    return L


def physics_section():
    if not os.path.exists(os.path.join(TABLES_DIR, "physics_checks.tex")):
        return []
    return [r"\clearpage", r"\section{Physical consistency of the formulas}\label{sec:physics}",
            r"A small MSE says the formula is close to the solution; it does not say the formula is physically "
            r"right. A physicist reads specific quantities off a solution: an energy, a slope at the origin, a "
            r"wavenumber and its dispersion relation, a decay rate, the radius of a star. Table~\ref{tab:physics} "
            r"extracts them from each discovered formula and compares them with their known values. This is the "
            r"measure that matters for the goal of this work (interpretability), and it can disagree with the "
            r"MSE: a formula can be accurate on $[0,1]$ and still give a wrong slope or a wrong far field.",
            r"\input{tables/physics_checks.tex}"]


def interpretability_section():
    if not os.path.exists(os.path.join(TABLES_DIR, "interpretability_table.tex")):
        return []
    L = [r"\clearpage", r"\section{Interpretability: PISR-NSGA-II versus PINN+PySR}\label{sec:interpretability}",
         r"The central claim of this work is that PISR-NSGA-II trades precision for interpretability. The fair "
         r"test of that claim is not the PINN (which is more precise but returns no formula) but the other route "
         r"to a formula without solution data: train a PINN on the same residual and boundary data, then distil "
         r"its prediction with symbolic regression (PySR). Table~\ref{tab:interpretability} compares both on "
         r"what a reader of the formula cares about: how small it is, whether it is \emph{the} analytic "
         r"solution when one exists, and whether it stays correct outside the domain where it was found.",
         r"The formula size counts the complete expression each method returns. For PISR-NSGA-II this includes "
         r"the boundary construction of the ansatz; in 2D that is the transfinite (Coons) interpolation of the "
         r"boundary data, which by itself accounts for most of the roughly 100 nodes of those formulas.",
         r"\input{tables/interpretability_table.tex}"]
    if os.path.exists(os.path.join(FIGS_DIR, "interpretability_tradeoff.pdf")):
        L += figure("interpretability_tradeoff.pdf", r"Median formula size against median solution error per "
                    r"problem. Each grey segment joins the two methods on the same problem.", place="H")
    return L


def make_latex_report(df, ext, stale):
    today = datetime.date.today().strftime("%B %d, %Y")
    L = [PREAMBLE, r"\begin{document}",
         r"\begin{center}",
         r"{\LARGE\bfseries\color{accent} PISR-NSGA-II}\\[4pt]",
         r"{\large Physics-informed multi-objective symbolic regression for differential equations}\\[8pt]",
         rf"{{\small\color{{muted}} Benchmark report --- generated {today}}}",
         r"\end{center}",
         r"\vspace{4pt}",
         r"{\small\tableofcontents}",
         r"\vspace{6pt}"]
    L += overview(df, ext, stale)
    L += [GLOSSARY]
    L += [r"\clearpage", r"\section{Benchmark suite}", r"\input{tables/suite_table.tex}",
          r"\subsection*{Physical fidelity of the suite}",
          r"\begin{description}[leftmargin=1.2em,itemsep=3pt,font=\normalfont\bfseries]",
          *[rf"\item[{t}] {txt}" for t, txt in PHYSICS_NOTES],
          r"\end{description}"]
    L += [r"\clearpage", r"\section{Results}",
          r"\subsection{Accuracy of PISR-NSGA-II}", r"\input{tables/results_table.tex}",
          r"\subsection{Comparison with baselines}", r"\input{tables/comparison_table.tex}"]
    if not ext.empty:
        for name, title in [("data_efficiency_comparison", "Data efficiency"), ("noise_robustness", "Robustness to measurement noise")]:
            if os.path.exists(os.path.join(TABLES_DIR, f"{name}.tex")):
                L += [rf"\subsection{{{title}}}", rf"\input{{tables/{name}.tex}}"]
    L += ablation_section(df)
    L += extrapolation_section()
    L += physics_section()
    L += interpretability_section()
    L += [r"\clearpage", r"\section{Search dynamics}"]
    for fig, cap in [
        ("convergence_history.pdf", r"Hall-of-fame champion per generation. Solid: validation score (PDE "
                                    r"residual and boundary error on a fixed grid), the only criterion used to "
                                    r"select the champion. Dashed: error against the reference, logged for "
                                    r"analysis and never used by the search."),
        ("parsimony_pressure.pdf", r"Best loss reached by expressions of each size in the final population "
                                   r"(envelope), with all individuals in grey."),
        ("pareto_fronts_3d.pdf", r"Final population in objective space; rank-1 individuals coloured by "
                                 r"complexity, dominated ones in grey."),
    ]:
        if os.path.exists(os.path.join(FIGS_DIR, fig)):
            L += figure(fig, cap, place="H") + [r"\clearpage"]
    L += per_problem()
    L += [r"\clearpage", r"\section{Discovered expressions}", r"\input{tables/formulas_table.tex}",
          r"\end{document}"]
    out = os.path.join(REPORT_DIR, "results.tex")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  [OK] {os.path.relpath(out, ROOT_DIR)}")


def clean_stale_outputs():
    """Borra tablas/figuras con nombres de versiones anteriores del reporte."""
    keep_tables = {"suite_table.tex", "results_table.tex", "comparison_table.tex", "formulas_table.tex",
                   "data_efficiency_comparison.tex", "noise_robustness.tex", "extrapolation_table.tex",
                   "physics_checks.tex", "interpretability_table.tex"}
    for f in glob.glob(os.path.join(TABLES_DIR, "*.tex")):
        if os.path.basename(f) not in keep_tables:
            os.remove(f)
    valid = {f"solution_{b.label}.pdf" for b in BENCHMARKS} | {f"pareto_pairs_{b.label}.pdf" for b in BENCHMARKS}
    valid |= {"convergence_history.pdf", "parsimony_pressure.pdf", "pareto_fronts_3d.pdf", "noise_robustness.pdf",
              "extrapolation_solutions.pdf", "extrapolation_errors.pdf", "interpretability_tradeoff.pdf"}
    for f in glob.glob(os.path.join(FIGS_DIR, "*")):
        if os.path.basename(f) not in valid:
            os.remove(f)


if __name__ == "__main__":
    print("Generando reporte...")
    clean_stale_outputs()
    df = load_summary()
    if df.empty:
        print("  [WARN] results/all_runs_summary.csv no existe: correr scripts/consolidate_runs.py")
    ext, stale = load_external()
    if stale:
        print(f"  [INFO] {stale} archivos de lineas base externas anteriores a "
              f"{PROBLEMS_REDEFINED:%Y-%m-%d}: excluidos (resolvian otro problema)")
    make_suite_table()
    make_results_table(df)
    make_comparison_table(df, ext)
    make_latex_report(df, ext, stale)
