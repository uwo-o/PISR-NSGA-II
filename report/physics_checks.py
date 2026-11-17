#!/usr/bin/env python3
"""
report/physics_checks.py — ¿La formula captura lo que le importa a un fisico?

El MSE mide cercania punto a punto; la tesis de la memoria es
interpretabilidad. Aqui se extraen de cada formula (el campeon tal como se
encontro, ver simplify_expressions.py) cantidades con significado fisico y se
comparan con su valor conocido: pendientes en el origen, energia, radio de la
estrella, numero de onda, relacion de dispersion, tasa de decaimiento.

Las cantidades se calculan sobre la formula (derivadas simbolicas, raices,
limites), nunca sobre la grilla numerica.

Salida: results/physics_checks.csv y report/tables/physics_checks.tex.
"""
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import solve_bvp
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import BY_LABEL  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(ROOT, "report", "tables")
GOOD, FAIR = 0.01, 0.10   # error relativo: <1 % bien, <10 % aceptable


def load_exprs():
    path = os.path.join(RESULTS, "simplification_report.json")
    rep = json.load(open(path)) if os.path.exists(path) else {}
    out = {}
    for lab, r in rep.items():
        if r.get("status") == "ok" and "expr_unsimplified" in r:
            e = sp.sympify(eval(r["expr_unsimplified"], sp.__dict__.copy()))
            names = {s.name: s for s in e.free_symbols}
            out[lab] = (e, names)
    return out


def sym(names, n):
    return names.get(n, sp.Symbol(n, nonnegative=True))


def num(e, **pt):
    """Valor complejo de e en el punto dado (por nombre de variable)."""
    sub = {s: v for s in e.free_symbols for k, v in pt.items() if s.name == k}
    return complex(sp.N(e.subs(sub), 20))


def deriv_at(e, names, var, order, **pt):
    """Derivada simbolica; si es singular en el punto, limite numerico por la derecha."""
    v = sym(names, var)
    d = sp.diff(e, v, order)
    try:
        val = num(d, **pt)
        if np.isfinite(val.real) and np.isfinite(val.imag):
            return val
    except Exception:
        pass
    pt2 = dict(pt)
    pt2[var] = pt[var] + 1e-9
    return num(d, **pt2)


def ode_slope0(f, ua, ub):
    """u'(0) de la solucion de u''=f(x,u) con u(0)=ua, u(1)=ub (referencia numerica)."""
    m = np.linspace(0, 1, 400)
    sol = solve_bvp(lambda s, y: np.vstack([y[1], f(s, y[0])]),
                    lambda a, b: np.array([a[0] - ua, b[0] - ub]),
                    m, np.vstack([ua + (ub - ua) * m, np.full_like(m, ub - ua)]), tol=1e-10, max_nodes=200000)
    return float(sol.sol(0.0)[1])


def lane_emden_radius():
    """Primer cero de Lane-Emden n=3 (referencia, integrando la EDO)."""
    from scipy.integrate import solve_ivp
    x0 = 1e-4
    ev = lambda s, y: y[0]
    ev.terminal = True
    sol = solve_ivp(lambda s, y: [y[1], -2 * y[1] / s - y[0] ** 3], (x0, 20), [1 - x0 ** 2 / 6, -x0 / 3],
                    rtol=1e-12, atol=1e-14, events=ev)
    return float(sol.t_events[0][0])


# (problema, cantidad, significado, valor esperado, funcion(e, names) -> valor)
def checks():
    sqrt2 = math.sqrt(2)
    return [
        ("HarmonicOscillator_1D", r"$E=-u''(0)/2u(0)$", "ground-state energy", 0.5,
         lambda e, n: (-deriv_at(e, n, "x", 2, x=0.0) / (2 * num(e, x=0.0))).real),
        ("Airy_1D", r"$u'(0)$", r"$\mathrm{Ai}'(0)$", -0.2588194037928068,
         lambda e, n: deriv_at(e, n, "x", 1, x=0.0).real),
        ("Thomas-Fermi_1D", r"$u(0)$", "nuclear charge normalisation", 1.0,
         lambda e, n: num(e, x=0.0).real),
        ("Thomas-Fermi_1D", r"$u'(0)$", "initial slope (energy of the atom)", -1.588071022611375,
         lambda e, n: deriv_at(e, n, "x", 1, x=0.0).real),
        ("Schrodinger_1D", r"$|u|$", "probability density (constant)", 1.0,
         lambda e, n: float(np.mean([abs(num(e, x=v)) for v in np.linspace(0, 1, 21)]))),
        ("Schrodinger_1D", r"$k=\partial_x\arg u$", "wavenumber", 2.0,
         lambda e, n: (deriv_at(e, n, "x", 1, x=0.5) / num(e, x=0.5)).imag),
        ("Schrodinger_2D", r"$k_x=\partial_x\arg u$", "wavenumber", 2.0,
         lambda e, n: (deriv_at(e, n, "x", 1, x=0.5, y=0.5) / num(e, x=0.5, y=0.5)).imag),
        ("Schrodinger-Unsteady_1D", r"$\omega/k^2$", "free-particle dispersion", 1.0,
         lambda e, n: (-(deriv_at(e, n, "t", 1, x=0.5, t=0.5) / num(e, x=0.5, t=0.5)).imag)
         / ((deriv_at(e, n, "x", 1, x=0.5, t=0.5) / num(e, x=0.5, t=0.5)).imag ** 2)),
        ("Schrodinger-Unsteady_2D", r"$\omega/|k|^2$", "free-particle dispersion", 1.0,
         lambda e, n: (-(deriv_at(e, n, "t", 1, x=0.5, y=0.5, t=0.5) / num(e, x=0.5, y=0.5, t=0.5)).imag)
         / ((deriv_at(e, n, "x", 1, x=0.5, y=0.5, t=0.5) / num(e, x=0.5, y=0.5, t=0.5)).imag ** 2
            + (deriv_at(e, n, "y", 1, x=0.5, y=0.5, t=0.5) / num(e, x=0.5, y=0.5, t=0.5)).imag ** 2)),
        ("Navier-Stokes-Unsteady_2D", r"$-\partial_t \ln\|\psi\|$", r"viscous decay rate $2\pi^2\nu$",
         2 * math.pi ** 2 * 0.01, lambda e, n: decay_rate(e)),
        ("Lane-Emden_1D", r"$u''(0)$", "central curvature", -1.0 / 3.0,
         lambda e, n: deriv_at(e, n, "x", 2, x=0.0).real),
        ("Lane-Emden_1D", r"$\xi_1$ (first zero)", "stellar radius", lane_emden_radius(),
         lambda e, n: first_zero(e, 1.0, 30.0)),
        ("Troesch_1D", r"$u'(0)$", "classic Troesch quantity",
         ode_slope0(lambda s, u: 3 * np.sinh(3 * u), 0.0, 1.0),
         lambda e, n: deriv_at(e, n, "x", 1, x=0.0).real),
        ("Ginzburg-Landau_1D", r"$u'(0)$", r"wall slope $1/\sqrt2$", 1 / sqrt2,
         lambda e, n: deriv_at(e, n, "x", 1, x=0.0).real),
        ("Ginzburg-Landau_1D", r"$u(4)$", r"far field $\to 1$ ($\tanh(4/\sqrt2)$)", math.tanh(4 / sqrt2),
         lambda e, n: num(e, x=4.0).real),
        ("Painleve-I_1D", r"$u'(0)$", "initial slope", ode_slope0(lambda s, u: u ** 2 + s, 0.0, 0.5),
         lambda e, n: deriv_at(e, n, "x", 1, x=0.0).real),
    ]


def first_zero(e, a, b):
    f = sp.lambdify(sp.Symbol("x", nonnegative=True), sp.re(e), "numpy")
    g = lambda v: float(np.real(f(complex(v))))
    xs = np.linspace(a, b, 3000)
    vals = [g(v) for v in xs]
    for i in range(len(xs) - 1):
        if np.sign(vals[i]) != np.sign(vals[i + 1]) and np.isfinite(vals[i]) and np.isfinite(vals[i + 1]):
            return brentq(g, xs[i], xs[i + 1])
    return math.nan


def decay_rate(e):
    """Pendiente de -ln||psi(.,t)|| en t in [0,1] (norma L2 en una grilla de [0,1]^2)."""
    names = {s.name: s for s in e.free_symbols}
    X, Y, T = names.get("x"), names.get("y"), names.get("t")
    f = sp.lambdify([X or sp.Symbol("x"), Y or sp.Symbol("y"), T or sp.Symbol("t")], e, "numpy")
    g = np.linspace(0.05, 0.95, 19)
    gx, gy = np.meshgrid(g, g)
    ts = np.linspace(0, 1, 11)
    norms = [np.sqrt(np.mean(np.abs(np.asarray(f(gx.astype(complex), gy.astype(complex), complex(tt)))) ** 2))
             for tt in ts]
    slope = np.polyfit(ts, np.log(norms), 1)[0]
    return -slope


def main():
    exprs = load_exprs()
    rows = []
    for lab, qty, meaning, expected, fn in checks():
        if lab not in exprs:
            rows.append((lab, qty, meaning, expected, math.nan))
            continue
        e, names = exprs[lab]
        try:
            v = float(fn(e, names))
        except Exception as ex:
            print(f"  [WARN] {lab} {qty}: {ex}")
            v = math.nan
        rows.append((lab, qty, meaning, expected, v))
    df = pd.DataFrame(rows, columns=["problem", "quantity", "meaning", "expected", "formula"])
    df["rel_error"] = (df["formula"] - df["expected"]).abs() / df["expected"].abs()
    df.to_csv(os.path.join(RESULTS, "physics_checks.csv"), index=False)
    for _, r in df.iterrows():
        print(f"  {r['problem']:<26} {r['quantity']:<28} esperado {r['expected']:+.5g}  formula {r['formula']:+.5g}  err {r['rel_error']:.1%}")
    make_table(df)


def fmt(v):
    if not np.isfinite(v):
        return r"\textemdash"
    return f"${v:.4g}$" if abs(v) >= 1e-3 or v == 0 else f"${v:.2e}$"


def make_table(df):
    rows, last = [], None
    for _, r in df.iterrows():
        b = BY_LABEL[r["problem"]]
        e = r["rel_error"]
        st = r"\stnone" if not np.isfinite(e) else (r"\stgood" if e < GOOD else (r"\stfair" if e < FAIR else r"\stpoor"))
        err = r"\textemdash" if not np.isfinite(e) else (f"${100 * e:.1f}\\,\\%$" if e >= 1e-3 else "$<0.1\\,\\%$")
        name = b.label_tex if r["problem"] != last else ""
        last = r["problem"]
        rows.append(rf"{name} & {r['quantity']} & {r['meaning']} & {fmt(r['expected'])} & {fmt(r['formula'])} & {st}\ {err} \\")
    lines = [
        r"\begin{table}[H]", r"\centering\small",
        r"\caption{Physical consistency of the discovered formulas: quantities a physicist would extract from "
        r"the solution, computed \emph{from the formula} (symbolic derivatives, roots, limits) and compared "
        r"with their known values. Status by relative error: \stgood\ $<1\,\%$, \stfair\ $<10\,\%$, "
        r"\stpoor\ $\geq10\,\%$, \stnone\ no result. The Lane--Emden radius and the Ginzburg--Landau far field "
        r"lie outside the domain $[0,1]$ used in the search.}",
        r"\label{tab:physics}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{@{}lllccr@{}}", r"\toprule",
        r"\textbf{Problem} & \textbf{Quantity} & \textbf{Meaning} & \textbf{Known} & \textbf{From formula} & \textbf{Error} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}",
    ]
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, "physics_checks.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  [OK] report/tables/physics_checks.tex")


if __name__ == "__main__":
    main()
