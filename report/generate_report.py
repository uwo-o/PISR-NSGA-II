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
        r"\caption{Suite de problemas. Todos se plantean en $[0,1]^d$ (y $t\in[0,1]$ si no son estacionarios). "
        r"Los marcados como \emph{numérica} no tienen forma cerrada; su referencia es una solución numérica de "
        r"alta resolución del mismo problema de contorno.}\label{tab:suite}\\",
        r"\toprule \textbf{Problema} & \textbf{Ecuación} & \textbf{Condiciones} & \textbf{Referencia} \\ \midrule",
        r"\endfirsthead",
        r"\toprule \textbf{Problema} & \textbf{Ecuación} & \textbf{Condiciones} & \textbf{Referencia} \\ \midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
        *rows,
        r"\end{longtable}",
    ])


def make_results_table(df):
    n_runs = int(df["run"].nunique()) if not df.empty else 0
    metrics = [("solution_mse", "MSE de solución", "sci"), ("best_mse_domain", "Residuo de dominio", "sci"),
               ("best_mse_boundary", "MSE de frontera", "sci"), ("pareto_size", "Tamaño del frente", "int"),
               ("runtime_s", "Tiempo", "time")]
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
    runs_txt = "una corrida" if n_runs == 1 else f"media $\\pm$ desviación estándar sobre {n_runs} corridas independientes"
    spec = "l" + "c" * (len(metrics) - 1) + "r"
    head = " & ".join(rf"\textbf{{{h}}}" for _, h, _ in metrics)
    write("results_table.tex", [
        r"\begin{table}[H]",
        r"\centering\small",
        rf"\caption{{Resultados de PISR-NSGA-II ({runs_txt}). El \emph{{MSE de solución}} se mide contra la "
        r"referencia después de la búsqueda; la búsqueda solo usa el residuo de la EDP y los datos de borde. "
        r"Estado: \stgood\ $<10^{-6}$, \stfair\ $<10^{-3}$, \stpoor\ $\geq10^{-3}$.}",
        r"\label{tab:results}",
        r"\begin{adjustbox}{max width=\linewidth}",
        rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}",
        r"\toprule",
        rf"\textbf{{Problema}} & {head} \\",
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
        r"\caption{MSE de solución contra la referencia y tiempo de pared. RK4/FDM es el solver numérico del "
        r"mismo problema a baja resolución (50 nodos por eje); entrega una grilla, no una fórmula. "
        r"PINN es una red neuronal informada por la física entrenada con el mismo residuo y los mismos datos "
        r"de borde (sin datos de la referencia); entrega pesos de una red, no una fórmula. PINN+PySR destila "
        r"la predicción del PINN en una fórmula con PySR, la alternativa interpretable sin datos (su tiempo "
        r"incluye entrenar el PINN). Todos los métodos se miden en los mismos puntos contra la misma "
        r"referencia; media $\pm$ desviación estándar sobre las semillas. Mejor MSE de cada fila en negrita.}",
        r"\label{tab:comparison}",
        r"\begin{adjustbox}{max width=\linewidth}",
        rf"\begin{{tabular}}{{@{{}}l{'c' * k}{'r' * k}@{{}}}}",
        r"\toprule",
        rf" & \multicolumn{{{k}}}{{c}}{{\textbf{{MSE de solución}}}} & \multicolumn{{{k}}}{{c}}{{\textbf{{Tiempo}}}} \\",
        rf"\cmidrule(lr){{2-{1 + k}}}\cmidrule(l){{{2 + k}-{1 + 2 * k}}}",
        rf"\textbf{{Problema}} & {head} & {head} \\",
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
\usepackage[spanish,es-noshorthands,es-tabla,es-nodecimaldot]{babel}
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
\fancyhead[L]{\small\textcolor{muted}{PISR-NSGA-II --- Reporte de resultados}}
\fancyhead[R]{\small\textcolor{muted}{\nouppercase{\leftmark}}}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.3pt}
"""


GLOSSARY = r"""\section{Notación y glosario}
\subsection*{Representación de la solución}
\begin{description}[leftmargin=1.2em,itemsep=3pt,font=\normalfont\bfseries]
\item[$\hat u$] La solución en forma cerrada descubierta: el campeón del Hall of Fame, después del pulido de
  constantes y la simplificación simbólica. En problemas de campo real se reporta $\operatorname{Re}(\hat u)$,
  que es exactamente la cantidad optimizada (las constantes complejas son solo un recurso de representación).
\item[$N$] El árbol de expresión que evoluciona el algoritmo cuando se usa el ansatz de frontera; $\hat u$ se construye a partir de él.
\item[Ansatz de frontera] Construcción que satisface los datos de borde \emph{exactamente} para cualquier $N$:
  \begin{itemize}[leftmargin=1.2em,itemsep=1pt]
  \item 1D Dirichlet: $\hat u = g(0) + x\,[g(1)-g(0)] + x(x-1)\,N(x)$.
  \item 1D problema de valor inicial: $\hat u = u_0 + u_0'\,x + x^2 N(x)$.
  \item 2D Dirichlet: $\hat u = \mathcal{C}[g](x,y) + x(1-x)\,y(1-y)\,N(x,y)$.
  \end{itemize}
  El factor que multiplica a $N$ se anula en la frontera, así que ahí $\hat u = g$ por construcción.
\item[$g$] Datos de borde del problema (las condiciones de la Tabla~\ref{tab:suite}).
\item[{$\mathcal{C}[g]$}] Interpolación transfinita (de Coons) de los datos de borde en $[0,1]^2$: la superficie
  que coincide con $g$ en las cuatro aristas,
  \[
  \begin{aligned}
  \mathcal{C}[g](x,y) ={}& (1-x)\,g(0,y) + x\,g(1,y) + (1-y)\,g(x,0) + y\,g(x,1)\\
  &- \big[(1-x)(1-y)\,g(0,0) + x(1-y)\,g(1,0) + (1-x)y\,g(0,1) + xy\,g(1,1)\big].
  \end{aligned}
  \]
  No contiene información aprendida; todo el aprendizaje está en $N$.
\item[Representación directa] Individuos cuyo árbol es $\hat u$ mismo (sin ansatz); el error de frontera pasa a
  ser un objetivo. Ambas representaciones compiten en la misma población (búsqueda híbrida).
\item[$\mathrm{Ai}$, $L_n$, $P_n$, $H_n$, $T_n$] Función de Airy; polinomios de Laguerre, Legendre, Hermite y Chebyshev de grado $n$.
\end{description}

\subsection*{Métricas}
\begin{description}[leftmargin=1.2em,itemsep=3pt,font=\normalfont\bfseries]
\item[MSE de solución] Diferencia cuadrática media entre $\hat u$ y la solución de referencia en una grilla de
  validación. Se calcula \emph{después} de la búsqueda, solo para evaluar.
\item[Residuo de dominio] Residuo cuadrático medio de la EDP para $\hat u$ en los puntos de colocación (el objetivo físico).
\item[MSE de frontera] Desajuste cuadrático medio con los datos de borde; idénticamente $0$ con el ansatz.
\item[Puntaje de validación] Residuo de dominio más error de frontera en una grilla fija de validación. Es el
  único criterio para elegir el campeón y para la parada temprana; nunca usa la solución de referencia.
\item[Tamaño del frente] Número de individuos no dominados (rango~1) en la población final.
\item[MSE en grilla, $\max|\text{error}|$] Error del campeón en la grilla exportada para gráficos (anotaciones de las figuras).
\item[Estado] \stgood\ MSE de solución $<10^{-6}$, \stfair\ $<10^{-3}$, \stpoor\ $\geq 10^{-3}$, \stnone\ sin resultado.
\end{description}

\subsection*{Métodos}
\begin{description}[leftmargin=1.2em,itemsep=3pt,font=\normalfont\bfseries]
\item[PISR-NSGA-II] Regresión simbólica informada por la física con NSGA-II sobre tres objetivos: residuo de
  dominio, error de frontera y tamaño de la expresión.
\item[RK4/FDM] Solver numérico del mismo problema a baja resolución (50 nodos por eje): Runge--Kutta y
  diferencias finitas en 1D, diferencias finitas en 2D. Entrega valores en una grilla, no una fórmula.
\item[PINN] Red neuronal informada por la física, entrenada con el mismo residuo y los mismos datos de borde.
\item[PINN+PySR] La predicción del PINN destilada en una fórmula con regresión simbólica (PySR).
\item[Referencia] Solución en forma cerrada cuando existe; si no, una solución numérica de alta resolución del
  mismo problema, marcada como \emph{numérica}.
\item[Hall of Fame] Mejor individuo visto durante toda la corrida según el puntaje de validación.
\end{description}

\subsection*{Simplificación de las expresiones reportadas}
Cada campeón se simplifica simbólicamente (SymPy). Las constantes se muestran con 4 cifras significativas y se
reemplazan por valores exactos ($2$, $\tfrac12$, $\pi$, $\sqrt2$, \dots) solo si la expresión resultante queda a
$10^{-3}$ (relativo) del campeón en toda su grilla de exportación; por ejemplo, una constante reportada como $\pi$
coincide con $\pi$ a esa tolerancia, no se afirma que sea exactamente $\pi$.
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

    L = [r"\section{Resumen}",
         r"PISR-NSGA-II busca soluciones en forma cerrada de ecuaciones diferenciales usando solo el residuo de "
         r"la ecuación y los datos de borde; la solución de referencia nunca se ve durante la búsqueda y solo "
         r"se usa para evaluar el resultado al final.",
         r"\paragraph{Objetivo.} Ayudar a los físicos a entender cómo se comportan las ecuaciones diferenciales "
         r"\emph{sin solución analítica}. Un solver numérico entrega números en una grilla; PISR-NSGA-II entrega "
         r"una fórmula corta que se puede leer, derivar, analizar en sus límites y asintóticas, y evaluar en "
         r"cualquier punto. El método cambia deliberadamente algo de precisión por interpretabilidad: no busca "
         r"superar a los solvers numéricos en exactitud, sino explicar la solución. Si existe una forma cerrada, "
         r"el algoritmo debería recuperarla; si no, debería producir una expresión compacta, precisa dentro del "
         r"dominio y que conserve la estructura de la ecuación (Sección~\ref{sec:extrapolation}).",
         r"\begin{itemize}[leftmargin=1.5em,itemsep=2pt]",
         rf"\item \textbf{{{len(BENCHMARKS)}}} problemas en {len(by_family())} familias "
         rf"(Tabla~\ref{{tab:suite}}); \textbf{{{n_runs}}} corrida{'s' if n_runs != 1 else ''} por problema.",
         *([r"\item Las tablas de precisión y las curvas de convergencia agregan todas las semillas. Todo lo que "
            r"necesita una sola fórmula (expresiones descubiertas, figuras por problema, extrapolación, "
            r"consistencia física) usa la \emph{semilla mediana} de cada problema: la semilla cuyo error queda "
            r"al medio del orden, es decir, el resultado típico, ni la mejor corrida ni una arbitraria."]
           if n_runs > 1 else []),
         rf"\item MSE de solución: \stgood\ \textbf{{{n_good}}} bajo $10^{{-6}}$, "
         rf"\stfair\ \textbf{{{n_fair}}} bajo $10^{{-3}}$, \stpoor\ \textbf{{{n_poor}}} en o sobre $10^{{-3}}$.",
         ]
    if np.isfinite(total_t):
        L.append(rf"\item Tiempo total de búsqueda por corrida: {seconds(total_t)}.")
    if poor:
        L.append(r"\item Problemas aún no resueltos a $10^{-3}$: " + ", ".join(poor) + ".")
    if missing:
        L.append(r"\item Problemas sin resultados: " + ", ".join(missing) + ".")
    L.append(r"\end{itemize}")
    if stale and ext.empty:
        L.append(r"\paragraph{Líneas base externas.} "
                 rf"Se encontraron {stale} archivos de las líneas base antiguas (DeepXDE, PySR y PySINDy), pero "
                 rf"son anteriores a la redefinición de los problemas ({PROBLEMS_REDEFINED:%Y-%m-%d}) y resuelven "
                 r"otras ecuaciones; se excluyen. Solo se compara con el solver numérico (RK4/FDM) hasta correr "
                 r"\texttt{run\_competitors.sh} (PINN y PINN+PySR).")
    return L


def per_problem():
    L = [r"\clearpage", r"\section{Análisis por problema}",
         r"Para cada problema: el campeón contra la referencia (arriba) y los compromisos de Pareto de la "
         r"población final (abajo). Los problemas no estacionarios se muestran en $t=0.5$; los de campo "
         r"complejo muestran la parte real y la imaginaria."]
    for _, title, items in by_family():
        L.append(rf"\subsection{{{title}}}")
        for b in items:
            sol, pairs = f"solution_{b.label}.pdf", f"pareto_pairs_{b.label}.pdf"
            if not os.path.exists(os.path.join(FIGS_DIR, sol)):
                continue
            L += [r"\clearpage", rf"\subsubsection{{{b.label_tex}}}",
                  r"\begin{center}\small\begin{tabular}{@{}ll@{}}",
                  rf"\textcolor{{muted}}{{Ecuación}} & ${b.equation}$ \\",
                  rf"\textcolor{{muted}}{{Condiciones}} & {b.conditions} \\",
                  rf"\textcolor{{muted}}{{Referencia}} & ${b.reference}$ \\",
                  *([rf"\textcolor{{muted}}{{Descubierta}} & \adjustbox{{max width=0.75\linewidth}}{{$\hat u = {f}$}} \\"]
                    if (f := pisr_formula(b)) else []),
                  r"\end{tabular}\end{center}"]
            L += figure(sol, rf"{b.label_tex}: campeón de PISR-NSGA-II contra la referencia.", r"0.8\linewidth", "H")
            if os.path.exists(os.path.join(FIGS_DIR, pairs)):
                L += figure(pairs, rf"{b.label_tex}: compromisos de Pareto por pares en la población final.", r"\linewidth", "H")
    return L


def ablation_section(df):
    """Representacion hibrida (por defecto) vs. solo ansatz (--no-hybrid)."""
    path = os.path.join(RESULTS_DIR, "ablation_no_hybrid", "comparison_summary.csv")
    L = [r"\subsection{Representación híbrida frente a solo ansatz}",
         r"Por defecto ambas representaciones compiten en la misma población: árboles que son la solución misma "
         r"(\emph{directos}) y árboles $N$ envueltos por el ansatz de frontera. Esto amplía la búsqueda (algunas "
         r"soluciones son mucho más cortas en una representación que en la otra), pero también puede desviarla: "
         r"un árbol directo puede satisfacer la ecuación en el interior sin cumplir la frontera y competir con "
         r"buenos árboles de ansatz. La variante solo ansatz (\texttt{-{}-no-hybrid}) se mantiene como opción "
         r"para revisar cada problema con ambas."]
    if not os.path.exists(path):
        L.append(r"Todavía no hay corridas solo ansatz.")
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
              r"\caption{MSE de solución con la representación híbrida (por defecto) y solo con ansatz "
              r"(\texttt{-{}-no-hybrid}). Mejor valor en negrita.}", r"\label{tab:ablation_hybrid}",
              r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
              r" & \multicolumn{2}{c}{\textbf{MSE de solución}} & \multicolumn{2}{c}{\textbf{Tiempo}} \\",
              r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
              r"\textbf{Problema} & Híbrida & Solo ansatz & Híbrida & Solo ansatz \\", r"\midrule",
              *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return L


def extrapolation_section():
    if not os.path.exists(os.path.join(TABLES_DIR, "extrapolation_table.tex")):
        return []
    L = [r"\clearpage", r"\section{Extrapolación fuera del dominio de entrenamiento}\label{sec:extrapolation}",
         r"Una fórmula que codifica la estructura de la solución debería seguir teniendo sentido fuera de la "
         r"región donde se encontró; un interpolante de datos no. Todos los problemas 1D estacionarios se "
         r"plantean en $[0,1]$. Aquí el campeón \emph{tal como se encontró} (antes de redondear constantes a "
         r"valores exactos) se evalúa en $[0,1.5]$ y se compara con la referencia: la forma cerrada si existe; "
         r"si no, la solución numérica del mismo problema continuada más allá de $x=1$ integrando la ecuación "
         r"desde $u(1),u'(1)$.",
         r"La comparación es un polinomio de Chebyshev ajustado a la referencia en $[0,1]$, con el menor grado "
         r"que iguala la precisión de PISR-NSGA-II dentro del dominio. Representa a cualquier sustituto que solo "
         r"interpola valores (una grilla numérica con un interpolante, una red neuronal, una regresión "
         r"polinomial): igual de bueno adentro, sin conocimiento de la ecuación afuera. La grilla de RK4/FDM "
         r"ni siquiera se puede evaluar fuera de $[0,1]$.",
         r"\input{tables/extrapolation_table.tex}"]
    for fig, cap in [("extrapolation_solutions.pdf", r"Campeón (discontinua), polinomio de Chebyshev de igual "
                      r"precisión (punteada) y referencia (gris) en $[0,1.5]$; región sombreada: fuera del "
                      r"dominio de entrenamiento."),
                     ("extrapolation_errors.pdf", r"Error puntual del campeón y del polinomio de Chebyshev de "
                      r"igual precisión en $[0,1.5]$.")]:
        if os.path.exists(os.path.join(FIGS_DIR, fig)):
            L += figure(fig, cap, place="H")
    return L


def physics_section():
    if not os.path.exists(os.path.join(TABLES_DIR, "physics_checks.tex")):
        return []
    return [r"\clearpage", r"\section{Consistencia física de las fórmulas}\label{sec:physics}",
            r"Un MSE pequeño dice que la fórmula está cerca de la solución; no dice que sea físicamente correcta. "
            r"Un físico lee cantidades específicas de una solución: una energía, una pendiente en el origen, un "
            r"número de onda y su relación de dispersión, una tasa de decaimiento, el radio de una estrella. La "
            r"Tabla~\ref{tab:physics} las extrae de cada fórmula descubierta y las compara con sus valores "
            r"conocidos. Es la medida que importa para el objetivo de este trabajo (interpretabilidad), y puede "
            r"discrepar del MSE: una fórmula puede ser precisa en $[0,1]$ y aun así dar una pendiente o un campo "
            r"lejano equivocados.",
            r"\input{tables/physics_checks.tex}"]


def interpretability_section():
    if not os.path.exists(os.path.join(TABLES_DIR, "interpretability_table.tex")):
        return []
    L = [r"\clearpage", r"\section{Interpretabilidad: PISR-NSGA-II frente a PINN+PySR}\label{sec:interpretability}",
         r"La tesis central de este trabajo es que PISR-NSGA-II cambia precisión por interpretabilidad. La prueba "
         r"justa de esa tesis no es el PINN (más preciso, pero sin fórmula), sino la otra vía hacia una fórmula "
         r"sin datos de la solución: entrenar un PINN con el mismo residuo y los mismos datos de borde, y destilar "
         r"su predicción con regresión simbólica (PySR). La Tabla~\ref{tab:interpretability} compara ambos en lo "
         r"que le importa a quien lee la fórmula: qué tan corta es, si es \emph{la} solución analítica cuando "
         r"existe, y si sigue siendo correcta fuera del dominio donde se encontró.",
         r"El tamaño cuenta la expresión completa que entrega cada método. Para PISR-NSGA-II incluye la "
         r"construcción de borde del ansatz; en 2D es la interpolación transfinita (de Coons) de los datos de "
         r"borde, que por sí sola explica la mayor parte de los cerca de 100 nodos de esas fórmulas.",
         r"\input{tables/interpretability_table.tex}"]
    if os.path.exists(os.path.join(FIGS_DIR, "interpretability_tradeoff.pdf")):
        L += figure("interpretability_tradeoff.pdf", r"Tamaño mediano de la fórmula contra error mediano de "
                    r"solución por problema. Cada segmento gris une a los dos métodos en el mismo problema.", place="H")
    if os.path.exists(os.path.join(TABLES_DIR, "pysr_front_table.tex")):
        L += [r"\subsection*{Lo que PySR encuentra frente a lo que entrega}",
              r"El criterio por defecto de PySR (\emph{best}) es solo una manera de elegir una fórmula de su "
              r"frente de Pareto. Para ser justos con PINN+PySR, la Tabla~\ref{tab:pysrfront} reporta además una "
              r"selección parsimoniosa que usa solo los datos del PINN, y un oráculo que revisa si la solución "
              r"exacta aparece en alguna parte del frente. El oráculo es una cota superior de lo que PySR "
              r"encontró, no un método aplicable; PISR-NSGA-II se reporta por su único campeón, sin un oráculo "
              r"comparable sobre su propio frente.",
              r"\input{tables/pysr_front_table.tex}"]
    return L


MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
         "octubre", "noviembre", "diciembre"]


def make_latex_report(df, ext, stale):
    d = datetime.date.today()
    today = f"{d.day} de {MESES[d.month - 1]} de {d.year}"
    L = [PREAMBLE, r"\begin{document}",
         r"\begin{center}",
         r"{\LARGE\bfseries\color{accent} PISR-NSGA-II}\\[4pt]",
         r"{\large Regresión simbólica multiobjetivo informada por la física para ecuaciones diferenciales}\\[8pt]",
         rf"{{\small\color{{muted}} Reporte de resultados --- generado el {today}}}",
         r"\end{center}",
         r"\vspace{4pt}",
         r"{\small\tableofcontents}",
         r"\vspace{6pt}"]
    L += overview(df, ext, stale)
    L += [GLOSSARY]
    L += [r"\clearpage", r"\section{Suite de problemas}", r"\input{tables/suite_table.tex}",
          r"\subsection*{Fidelidad física de la suite}",
          r"\begin{description}[leftmargin=1.2em,itemsep=3pt,font=\normalfont\bfseries]",
          *[rf"\item[{t}] {txt}" for t, txt in PHYSICS_NOTES],
          r"\end{description}"]
    L += [r"\clearpage", r"\section{Resultados}",
          r"\subsection{Precisión de PISR-NSGA-II}", r"\input{tables/results_table.tex}",
          r"\subsection{Comparación con las líneas base}", r"\input{tables/comparison_table.tex}"]
    L += ablation_section(df)
    L += extrapolation_section()
    L += physics_section()
    L += interpretability_section()
    L += [r"\clearpage", r"\section{Dinámica de la búsqueda}"]
    for fig, cap in [
        ("convergence_history.pdf", r"Campeón del Hall of Fame por generación. Continua: puntaje de validación "
                                    r"(residuo de la EDP y error de frontera en una grilla fija), el único "
                                    r"criterio para elegir el campeón. Discontinua: error contra la referencia, "
                                    r"registrado para el análisis y nunca usado por la búsqueda."),
        ("parsimony_pressure.pdf", r"Mejor pérdida alcanzada por expresiones de cada tamaño en la población "
                                   r"final (envolvente), con todos los individuos en gris."),
        ("pareto_fronts_3d.pdf", r"Población final en el espacio de objetivos; individuos de rango 1 coloreados "
                                 r"por complejidad, los dominados en gris."),
    ]:
        if os.path.exists(os.path.join(FIGS_DIR, fig)):
            L += figure(fig, cap, place="H") + [r"\clearpage"]
    L += per_problem()
    L += [r"\clearpage", r"\section{Expresiones descubiertas}", r"\input{tables/formulas_table.tex}",
          r"\end{document}"]
    out = os.path.join(REPORT_DIR, "results.tex")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  [OK] {os.path.relpath(out, ROOT_DIR)}")


def clean_stale_outputs():
    """Borra tablas/figuras con nombres de versiones anteriores del reporte."""
    keep_tables = {"suite_table.tex", "results_table.tex", "comparison_table.tex", "formulas_table.tex", "extrapolation_table.tex",
                   "physics_checks.tex", "interpretability_table.tex", "pysr_front_table.tex"}
    for f in glob.glob(os.path.join(TABLES_DIR, "*.tex")):
        if os.path.basename(f) not in keep_tables:
            os.remove(f)
    valid = {f"solution_{b.label}.pdf" for b in BENCHMARKS} | {f"pareto_pairs_{b.label}.pdf" for b in BENCHMARKS}
    valid |= {"convergence_history.pdf", "parsimony_pressure.pdf", "pareto_fronts_3d.pdf",
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
