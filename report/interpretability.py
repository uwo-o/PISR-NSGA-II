#!/usr/bin/env python3
"""Interpretabilidad: PISR-NSGA-II contra PINN+PySR, los dos metodos que
entregan una FORMULA sin datos de la solucion.

Por problema y semilla, para cada metodo:
  error         solution_mse contra la referencia (los mismos puntos);
  nodos         tamano del arbol simbolico de la formula COMPLETA tal como la
                entrega el metodo (con el ansatz de borde incluido en PISR),
                contado igual para ambos tras evaluar las constantes;
  recuperada    la formula, con sus constantes redondeadas a valores exactos,
                coincide con la solucion analitica a 1e-10 (solo problemas con
                forma cerrada);
  extrapolacion error relativo maximo en (1, 1.5] (problemas 1D estacionarios;
                misma referencia que report/extrapolation.py).

Salidas: results/interpretability.csv (una fila por problema, metodo y
semilla), report/tables/interpretability_table.tex y
report/figures/interpretability_tradeoff.pdf.
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import BENCHMARKS, RESULTS_DIR, by_family, is_fresh, seed_dirs  # noqa: E402
from simplify_expressions import SYMS, boundary_parts, nice_value, to_real, with_timeout, x, y, t  # noqa: E402
import extrapolation as ex  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "report", "tables")
FIGS = os.path.join(ROOT, "report", "figures")
PISR, PYSR = "PISR-NSGA-II", "PINN+PySR"
RECOVERY_TOL = 1e-10

# Solucion analitica de los problemas con forma cerrada (mismas formulas que
# PDEProblem::exact en src/pde_problems.cpp).
_RE = 20
_LAM = sp.Rational(_RE, 2) - sp.sqrt(sp.Rational(_RE ** 2, 4) + 4 * sp.pi ** 2)
EXACT = {
    "Laplace_1D": x,
    "Poisson_1D": sp.sin(sp.pi * x),
    "HarmonicOscillator_1D": sp.exp(-x ** 2 / 2),
    "Schrodinger_1D": sp.exp(2 * sp.I * x),
    "Schrodinger_2D": sp.exp(2 * sp.I * (x + y)),
    "Schrodinger-Unsteady_1D": sp.exp(sp.I * sp.pi * (x - sp.pi * t)),
    "Schrodinger-Unsteady_2D": sp.exp(sp.I * sp.pi * (x + y - 2 * sp.pi * t)),
    "Ginzburg-Landau_1D": sp.tanh(x / sp.sqrt(2)),
    "Navier-Stokes_2D": y - sp.exp(_LAM * x) * sp.sin(2 * sp.pi * y) / (2 * sp.pi),
    "Navier-Stokes-Unsteady_2D": sp.sin(sp.pi * x) * sp.sin(sp.pi * y) * sp.exp(-2 * sp.pi ** 2 * sp.Rational(1, 100) * t),
}


# ─── Formulas ─────────────────────────────────────────────────────────────────
def _rename(e):
    by_name = {"x": x, "y": y, "t": t}
    return e.xreplace({s: by_name[s.name] for s in e.free_symbols if s.name in by_name})


def pisr_formula(b, d):
    p = os.path.join(d, f"expr_{b.label}_PISR-EMOAD.sym")
    if not os.path.exists(p) or not is_fresh(p):
        return None
    meta = dict(line.rstrip("\n").split("=", 1) for line in open(p) if "=" in line)
    try:
        N = sp.sympify(meta["tree"], locals=SYMS)
    except Exception:
        return None
    U = N
    if meta.get("uses_ansatz") == "1":
        ab = boundary_parts(b)
        if ab is None:
            return None
        U = ab[0] + ab[1] * N
    if not b.complex_field and U.has(sp.I):
        U = to_real(U)
    return U


def pysr_formula(b, d):
    p = os.path.join(d, f"{b.label}_pinn_pysr.csv")
    if not os.path.exists(p) or not is_fresh(p):
        return None, np.nan
    row = pd.read_csv(p).iloc[0]
    err = float(row["solution_mse"])
    if not bool(row.get("ok", True)) or not isinstance(row.get("expr_srepr"), str):
        return None, err
    try:
        e = eval(row["expr_srepr"], sp.__dict__.copy())  # srepr escrito por competitors/pinn_pysr.py
    except Exception:
        return None, err
    return _rename(sp.sympify(e)), err


# ─── Metricas ─────────────────────────────────────────────────────────────────
def nodes(e):
    """Nodos del arbol simbolico con las constantes evaluadas (pi, Ai(0), ...
    cuentan como un numero), igual para ambos metodos."""
    v = with_timeout(lambda: sp.nfloat(e, 12), 10)
    v = e if v is None else v
    return sum(1 for _ in sp.preorder_traversal(v))


class Points:
    """Puntos de validacion del problema (results/reference/<EDP>.csv)."""
    def __init__(self, b):
        p = os.path.join(RESULTS_DIR, "reference", f"{b.label}.csv")
        df = pd.read_csv(p)
        df = df[df.kind == "dom"]
        self.x, self.y, self.t = (df[c].to_numpy(float) for c in ("x", "y", "t"))

    def eval(self, e):
        try:
            f = sp.lambdify((x, y, t), e, modules=["numpy", "scipy"])
            v = np.asarray(f(self.x.astype(complex), self.y.astype(complex), self.t.astype(complex)), dtype=complex)
        except Exception:
            f = sp.lambdify((x, y, t), e, modules=["mpmath"])
            v = np.array([complex(f(a, c, d)) for a, c, d in zip(self.x, self.y, self.t)])
        return np.full(self.x.shape, complex(v)) if v.shape == () else v


CONST_TOL = 1e-3   # constantes a ~3 decimales del valor exacto (como el simplificador)


def _candidates(v, exact_consts):
    out = list(nice_value(v))
    for c in exact_consts:
        for k in (c, -c, 2 * c, c / 2, 1 / c if c else c):
            if k:
                out.append(sp.Float(k, 17))
    out = [c for c in out if abs(float(c) - v) <= CONST_TOL * max(abs(v), 1e-6)]
    if abs(v) <= 1e-6:  # residuo numerico (p. ej. tanh(x/sqrt 2) + 1e-10): cuenta como 0
        out.insert(0, sp.Integer(0))
    return out


def recovered(e, exact, pts):
    """La ESTRUCTURA es la de la solucion analitica: redondeando cada constante
    al valor exacto mas cercano (a CONST_TOL relativo; enteros, fracciones
    simples, pi, sqrt 2, ... o constantes de la propia solucion; las de
    magnitud <= 1e-6 pueden ir a 0), la formula coincide con ella a
    RECOVERY_TOL."""
    ue = pts.eval(exact)
    scale = max(float(np.max(np.abs(ue))), 1e-12)
    dev = lambda f: float(np.max(np.abs(pts.eval(f) - ue))) / scale
    try:
        if not dev(e) <= 1e-2:
            return False
        consts = [float(c) for c in sp.nfloat(exact, 17).atoms(sp.Float)]

        def snap(f0):
            for c0 in sorted(f0.atoms(sp.Float), key=lambda c: -abs(float(c))):
                cands = _candidates(float(c0), consts)
                if cands:
                    f0 = min((f0.xreplace({c0: c}) for c in cands), key=dev)
            return f0
        # Formas equivalentes con otras constantes (p. ej. e^{a x} e^{b x} con
        # a + b = pi): tambien se prueba tras combinar exponenciales y potencias.
        for form in (lambda: e, lambda: sp.powsimp(sp.expand(e), force=True),
                     lambda: sp.simplify(sp.powsimp(e, force=True))):
            f0 = with_timeout(form, 15)
            if f0 is not None and dev(snap(f0)) <= RECOVERY_TOL:
                return True
        return False
    except Exception:
        return False


_REF_CACHE = {}
XS = np.linspace(0, ex.X_MAX, 601)


def extrapolation_error(b, e):
    """Error relativo maximo en (1, 1.5]; misma referencia que extrapolation.py."""
    if b.dim != 1 or b.unsteady:
        return np.nan
    if b.label not in _REF_CACHE:
        cf = ex.closed_form(b)
        _REF_CACHE[b.label] = ex.eval_expr(cf, XS) if cf is not None else ex.numeric_reference(b, XS).astype(complex)
    ref = _REF_CACHE[b.label]
    try:
        v = ex.eval_expr(e.subs({y: 0, t: 0}), XS)
    except Exception:
        return np.nan
    m = (XS > 1.0) & np.isfinite(ref)
    if not m.any() or not np.all(np.isfinite(v[m])):
        return np.inf
    return float(np.max(np.abs(v[m] - ref[m]))) / float(np.nanmax(np.abs(ref)))


# ─── Recoleccion ──────────────────────────────────────────────────────────────
def pisr_errors():
    p = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    if not os.path.exists(p):
        return {}
    d = pd.read_csv(p)
    d = d[d.method == PISR]
    return {(r.pde, int(r.run)): float(r.solution_mse) for r in d.itertuples()}


def collect():
    errs = pisr_errors()
    rows = []
    dirs = [dd for dd in seed_dirs() if any(is_fresh(os.path.join(dd, f)) for f in os.listdir(dd))]
    for b in BENCHMARKS:
        pts = Points(b)
        exact = EXACT.get(b.label)
        for d in dirs:
            run = int(d.rsplit("_", 1)[1])
            for method in (PISR, PYSR):
                if method == PISR:
                    e, err = pisr_formula(b, d), errs.get((b.label, run), np.nan)
                else:
                    e, err = pysr_formula(b, d)
                if e is None and not np.isfinite(err):
                    continue
                rows.append({
                    "problem": b.label, "method": method, "run": run,
                    "solution_mse": err if np.isfinite(err) and err >= 0 else np.nan,
                    "nodes": nodes(e) if e is not None else np.nan,
                    "recovered": (recovered(e, exact, pts) if e is not None else False) if exact is not None else np.nan,
                    "extrapolation": extrapolation_error(b, e) if e is not None else np.nan,
                })
        n = sum(1 for r in rows if r["problem"] == b.label)
        print(f"  [OK] {b.label:<27} {n} formulas")
    return pd.DataFrame(rows)


# ─── Tabla y figura ───────────────────────────────────────────────────────────
def sci(v):
    if v is None or not np.isfinite(v):
        return r"\textemdash"
    if v == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(v))))
    return rf"${v / 10 ** e:.1f}\times10^{{{e}}}$"


def med(df, method, label, col):
    v = df[(df.method == method) & (df.problem == label)][col].astype(float).replace([np.inf], np.nan).dropna()
    return float(v.median()) if len(v) else np.nan


def rate(df, method, label):
    v = df[(df.method == method) & (df.problem == label)]["recovered"].dropna()
    return f"{int(v.astype(bool).sum())}/{len(v)}" if len(v) else r"\textemdash"


def bold_min(a, b, fmt):
    """Formatea dos valores y pone en negrita el menor (si ambos existen)."""
    def bold(v):
        return "$\\mathbf{" + v[1:-1] + "}$" if v.startswith("$") else r"\textbf{" + v + "}"
    sa, sb = fmt(a), fmt(b)
    if np.isfinite(a) and np.isfinite(b) and a != b:
        if a < b:
            sa = bold(sa)
        else:
            sb = bold(sb)
    return sa, sb


def make_table(df):
    has_pysr = (df.method == PYSR).any()
    rows = []
    for _, title, items in by_family():
        rows.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\familyrow{{{title}}}}} \\")
        for b in items:
            e1, e2 = bold_min(med(df, PISR, b.label, "solution_mse"), med(df, PYSR, b.label, "solution_mse"), sci)
            n1, n2 = bold_min(med(df, PISR, b.label, "nodes"), med(df, PYSR, b.label, "nodes"),
                              lambda v: f"{v:.0f}" if np.isfinite(v) else r"\textemdash")
            x1, x2 = bold_min(med(df, PISR, b.label, "extrapolation"), med(df, PYSR, b.label, "extrapolation"), sci)
            r1, r2 = (rate(df, PISR, b.label), rate(df, PYSR, b.label)) if b.label in EXACT else (r"\textemdash",) * 2
            rows.append(rf"{b.label_tex} & {e1} & {e2} & {n1} & {n2} & {r1} & {r2} & {x1} & {x2} \\")
    note = "" if has_pysr else (r" \emph{PINN+PySR results are not available yet "
                                r"(\texttt{run\_competitors.sh}); only PISR-NSGA-II is shown.}")
    lines = [
        r"\begin{table}[H]", r"\centering\small",
        r"\caption{Interpretability: the two methods that return a formula without solution data. "
        r"Medians over seeds. \emph{Nodes}: size of the symbolic tree of the complete formula as returned "
        r"(PISR includes its boundary ansatz). \emph{Exact}: seeds whose formula, with constants rounded to "
        r"exact values, equals the analytic solution to $10^{-10}$ (problems with a closed form). "
        r"\emph{Extrap.}: maximum relative error on $(1, 1.5]$ (1D steady problems). Better value in bold."
        + note + "}",
        r"\label{tab:interpretability}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{Solution MSE}} & \multicolumn{2}{c}{\textbf{Nodes}} "
        r"& \multicolumn{2}{c}{\textbf{Exact}} & \multicolumn{2}{c}{\textbf{Extrap.}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
        r"\textbf{Problem} & PISR & P+P & PISR & P+P & PISR & P+P & PISR & P+P \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}",
    ]
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, "interpretability_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  [OK] report/tables/interpretability_table.tex")


def make_figure(df):
    out = os.path.join(FIGS, "interpretability_tradeoff.pdf")
    if not (df.method == PYSR).any():
        if os.path.exists(out):
            os.remove(out)  # sin PINN+PySR no hay comparacion que graficar
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.0), layout="constrained")
    for b in BENCHMARKS:
        p = (med(df, PISR, b.label, "nodes"), med(df, PISR, b.label, "solution_mse"))
        q = (med(df, PYSR, b.label, "nodes"), med(df, PYSR, b.label, "solution_mse"))
        if not all(np.isfinite(v) for v in p + q):
            continue
        p, q = (p[0], max(p[1], 1e-32)), (q[0], max(q[1], 1e-32))
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#BBB", lw=0.8, zorder=1)
        ax.annotate(b.label.replace("_", " "), p, fontsize=6.5, color="#555", xytext=(3, 2), textcoords="offset points")
    for method, color, marker in [(PISR, "#2E86C1", "o"), (PYSR, "#E67E22", "s")]:
        pts = [(med(df, method, b.label, "nodes"), med(df, method, b.label, "solution_mse")) for b in BENCHMARKS]
        pts = [(a, max(c, 1e-32)) for a, c in pts if np.isfinite(a) and np.isfinite(c)]
        if pts:
            ax.scatter(*zip(*pts), color=color, marker=marker, s=34, label=method, zorder=2, edgecolors="white", lw=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("formula size (nodes, median over seeds)")
    ax.set_ylabel("solution MSE (median over seeds)")
    ax.set_title("Interpretability vs. accuracy (lower-left is better; grey lines join the same problem)", fontsize=9.5)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print("  [OK] report/figures/interpretability_tradeoff.pdf")


def main():
    df = collect()
    if df.empty:
        print("  [--] sin formulas por semilla (results/run_*): se omite")
        return
    df.to_csv(os.path.join(RESULTS_DIR, "interpretability.csv"), index=False)
    make_table(df)
    make_figure(df)


if __name__ == "__main__":
    main()
