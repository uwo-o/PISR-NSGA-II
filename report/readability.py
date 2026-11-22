#!/usr/bin/env python3
"""Legibilidad de las formulas: medidas objetivas, ademas del numero de nodos.

Para cada problema y semilla, sobre la formula de PISR-NSGA-II, la de PINN+PySR
con su criterio por defecto ("best") y la seleccion parsimoniosa de su frente
(report/pysr_front_analysis.py):
  profundidad   maximo de funciones anidadas en un camino raiz-hoja (exp, log,
                trigonometricas, hiperbolicas, especiales y potencias no
                enteras); la aritmetica (+, *, /, potencias enteras) no cuenta;
  constantes    constantes numericas ajustadas por el metodo. En PISR con
                ansatz solo las del arbol N: las del ansatz vienen de los datos
                de borde y no se ajustan;
  singular      la formula es no finita o explota (|u| > 1000 max|u| en el
                dominio) en algun punto de [0, 1.5]^d: polos o crecimientos que
                no corresponden a la fisica del problema.

Salidas: results/readability.csv y report/tables/readability_table.tex.
"""
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import BENCHMARKS, BY_LABEL, RESULTS_DIR, by_family, is_fresh, seed_dirs  # noqa: E402
from simplify_expressions import SYMS  # noqa: E402
import interpretability as I  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "report", "tables")
METHODS = ["PISR-NSGA-II", "PySR best", "PySR pars."]
BLOWUP = 1e3


# ─── Medidas ──────────────────────────────────────────────────────────────────
def _is_function(e):
    if isinstance(e, sp.Function) or isinstance(e, sp.exp):
        return True
    if isinstance(e, sp.Pow):
        ex = e.exp
        return not (ex.is_Integer or (ex.is_Number and float(ex).is_integer()))
    return False


def depth(e):
    """Funciones anidadas en el peor camino raiz-hoja."""
    if not e.args:
        return 0
    inner = max(depth(a) for a in e.args)
    return inner + (1 if _is_function(e) else 0)


def n_constants(e):
    return len({float(c) for c in e.atoms(sp.Float)})


def singular(e, dim, unsteady):
    """No finita o explosiva en [0,1.5]^d, comparada con su tamano en [0,1]^d."""
    names = ["x"] + (["y"] if dim == 2 else []) + (["t"] if unsteady else [])
    n = {1: 3001, 2: 121, 3: 31}[len(names)]
    axes = np.meshgrid(*[np.linspace(0.0, 1.5, n)] * len(names), indexing="ij")
    vals = {k: a.ravel().astype(complex) for k, a in zip(names, axes)}
    args = [vals.get(k, np.zeros(1, complex)) for k in ("x", "y", "t")]
    try:
        f = sp.lambdify((I.x, I.y, I.t), e, modules=["numpy", "scipy"])
        with np.errstate(all="ignore"):
            u = np.asarray(f(*args), dtype=complex)
    except Exception:
        return np.nan
    if u.shape == ():
        return False
    inside = np.all(np.stack([vals[k].real <= 1.0 for k in names]), axis=0)
    a = np.abs(u)
    if not np.all(np.isfinite(a)):
        return True
    scale = max(float(np.max(a[inside])), 1e-12)
    return bool(np.max(a) > BLOWUP * scale)


# ─── Formulas ─────────────────────────────────────────────────────────────────
def pisr_tree_and_formula(b, d):
    """(arbol aprendido, formula completa) del campeon de PISR en la semilla d."""
    p = os.path.join(d, f"expr_{b.label}_PISR-EMOAD.sym")
    if not os.path.exists(p) or not is_fresh(p):
        return None, None
    meta = dict(line.rstrip("\n").split("=", 1) for line in open(p) if "=" in line)
    try:
        tree = sp.sympify(meta["tree"], locals=SYMS)
    except Exception:
        return None, None
    return tree, I.pisr_formula(b, d)


def measure(job):
    label, d = job
    b = BY_LABEL[label]
    run = int(d.rsplit("_", 1)[1])
    rows = []
    tree, full = pisr_tree_and_formula(b, d)
    if full is not None:
        rows.append(("PISR-NSGA-II", full, tree))
    e, _ = I.pysr_formula(b, d)
    if e is not None:
        rows.append(("PySR best", e, e))
    return [{"problem": label, "run": run, "method": m,
             "depth": depth(f), "constants": n_constants(learned),
             "singular": singular(f, b.dim, b.unsteady)} for m, f, learned in rows]


def measure_pars(row):
    b = BY_LABEL[row["problem"]]
    try:
        e = sp.sympify(row["pars_expr"], locals={"x": I.x, "y": I.y, "t": I.t, "I": sp.I})
    except Exception:
        return None
    return {"problem": row["problem"], "run": int(row["run"]), "method": "PySR pars.",
            "depth": depth(e), "constants": n_constants(e), "singular": singular(e, b.dim, b.unsteady)}


# ─── Tabla ────────────────────────────────────────────────────────────────────
def make_table(df):
    def med(sub, col):
        v = sub[col].dropna()
        return float(v.median()) if len(v) else np.nan

    def pct(sub):
        v = sub["singular"].dropna()
        return f"{100 * v.astype(bool).mean():.0f}\\,\\%" if len(v) else r"\textemdash"

    def fmt(v):
        return r"\textemdash" if not np.isfinite(v) else f"{v:.0f}"

    def bold_low(vals, f):
        cells = [f(v) for v in vals]
        fin = [v for v in vals if np.isfinite(v)]
        if len(fin) > 1:
            lo = min(fin)
            cells = [r"\textbf{" + c + "}" if np.isfinite(v) and v == lo else c for c, v in zip(cells, vals)]
        return cells

    rows = []
    for _, title, items in by_family():
        rows.append(rf"\multicolumn{{10}}{{@{{}}l}}{{\familyrow{{{title}}}}} \\")
        for b in items:
            subs = [df[(df.problem == b.label) & (df.method == m)] for m in METHODS]
            dep = bold_low([med(s, "depth") for s in subs], fmt)
            con = bold_low([med(s, "constants") for s in subs], fmt)
            sing = [pct(s) for s in subs]
            rows.append(rf"{b.label_tex} & " + " & ".join(dep + con + sing) + r" \\")
    tot = [df[df.method == m] for m in METHODS]
    total = (r"\midrule \textbf{Todas} & " + " & ".join(
        [fmt(med(s, "depth")) for s in tot] + [fmt(med(s, "constants")) for s in tot] + [pct(s) for s in tot])
        + r" \\")
    head = "PISR & best & pars."
    lines = [
        r"\begin{table}[H]", r"\centering\small",
        r"\caption{Legibilidad de las fórmulas, medianas sobre las semillas. \emph{Profundidad}: máximo de "
        r"funciones anidadas (exp, log, trigonométricas, hiperbólicas, especiales, potencias no enteras) en un "
        r"camino de la expresión; la aritmética no cuenta. \emph{Constantes}: constantes numéricas ajustadas por "
        r"el método (en PISR con ansatz, solo las del árbol $N$). \emph{Singular}: porcentaje de semillas cuya "
        r"fórmula es no finita o explota ($|u|>10^3\max_{[0,1]^d}|u|$) en $[0,1.5]^d$. PySR \emph{best} es su "
        r"criterio por defecto y \emph{pars.} la selección parsimoniosa de su frente. Menor valor en negrita.}",
        r"\label{tab:readability}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
        r" & \multicolumn{3}{c}{\textbf{Profundidad}} & \multicolumn{3}{c}{\textbf{Constantes}} "
        r"& \multicolumn{3}{c}{\textbf{Singular}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
        rf"\textbf{{Problema}} & {head} & {head} & {head} \\", r"\midrule",
        *rows, total, r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}",
    ]
    with open(os.path.join(TABLES, "readability_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  [OK] report/tables/readability_table.tex")


def main():
    dirs = [d for d in seed_dirs() if any(is_fresh(os.path.join(d, f)) for f in os.listdir(d))]
    jobs = [(b.label, d) for b in BENCHMARKS for d in dirs]
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        rows = [r for chunk in pool.map(measure, jobs, chunksize=4) for r in chunk]
        front = os.path.join(RESULTS_DIR, "pysr_front_analysis.csv")
        if os.path.exists(front):
            pars = pd.read_csv(front).to_dict("records")
            rows += [r for r in pool.map(measure_pars, pars, chunksize=8) if r]
    if not rows:
        print("  [--] sin formulas: se omite")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "readability.csv"), index=False)
    make_table(df)


if __name__ == "__main__":
    main()
