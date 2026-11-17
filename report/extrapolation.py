#!/usr/bin/env python3
"""
report/extrapolation.py — ¿Que pasa fuera del dominio de entrenamiento?

Todos los problemas 1D estacionarios se plantean en [0,1]. Aqui se evalua la
formula reportada por PISR-NSGA-II en [0, 1.5] y se compara con:

  * Referencia en [0, 1.5]: forma cerrada si existe; si no, la solucion
    numerica del MISMO problema en [0,1] (scipy solve_bvp, o IVP para
    Lane-Emden) continuada mas alla de x=1 integrando la EDO con los valores
    u(1), u'(1). Es la continuacion unica de esa solucion.
  * Polinomio de Chebyshev ajustado a la referencia en [0,1], con el MENOR
    grado que iguala (o mejora) la precision de PISR-NSGA-II dentro del
    dominio. Representa a cualquier sustituto que solo interpola datos (grilla
    numerica + interpolante, red neuronal, regresion polinomial): igual de
    preciso adentro, sin informacion sobre la ecuacion afuera.
  * La grilla RK4/FDM no se puede evaluar fuera de [0,1]: no hay nada que
    extrapolar.

Salida: results/extrapolation.csv, report/tables/extrapolation_table.tex,
report/figures/extrapolation_solutions.pdf y extrapolation_errors.pdf.
"""
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from numpy.polynomial import chebyshev as C
from scipy.integrate import solve_bvp, solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import BENCHMARKS, BOUNDARY, latest_result  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(ROOT, "report", "tables")
FIGS = os.path.join(ROOT, "report", "figures")
X_MAX = 1.5
x = sp.Symbol("x", nonnegative=True)

# EDO u'' = f(x, u, u') de cada problema sin forma cerrada (mismas ecuaciones
# que src/pde_problems.cpp; ver la tabla de la suite en el reporte).
ODES = {
    "Fisher_1D": lambda s, u, v: -u * (1 - u),
    "Duffing_1D": lambda s, u, v: np.sin(np.pi * s) - u - u ** 3,
    "Thomas-Fermi_1D": lambda s, u, v: np.abs(u) ** 1.5 / np.sqrt(np.maximum(s, 1e-300)),
    "Troesch_1D": lambda s, u, v: 3.0 * np.sinh(3.0 * u),
    "Painleve-I_1D": lambda s, u, v: u ** 2 + s,
    "Lane-Emden_1D": lambda s, u, v: -2.0 * v / np.maximum(s, 1e-300) - u ** 3,
}


def closed_form(b):
    """Solucion exacta (sympy) para problemas con forma cerrada: es g."""
    if b.numerical:
        return None
    kind, g = BOUNDARY[b.label]
    return sp.sympify(g, locals={"x": x})


def numeric_reference(b, xs):
    """Solucion en xs (subconjunto de [0, X_MAX]); NaN donde la continuacion explota."""
    f = ODES[b.label]
    kind, g = BOUNDARY[b.label]
    if kind == "ivp":
        # Lane-Emden: arranque por serie en x0 (u = 1 - x^2/6 + x^4/40), luego IVP.
        x0 = 1e-3
        u0, v0 = 1 - x0 ** 2 / 6 + x0 ** 4 / 40, -x0 / 3 + x0 ** 3 / 10
        sol = solve_ivp(lambda s, y: [y[1], f(s, y[0], y[1])], (x0, X_MAX), [u0, v0],
                        method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True)
        out = np.full_like(xs, np.nan)
        m = (xs >= x0) & (xs <= sol.t[-1])
        out[m] = sol.sol(xs[m])[0]
        out[xs < x0] = 1 - xs[xs < x0] ** 2 / 6
        return out
    gx = sp.lambdify(x, sp.sympify(g, locals={"x": x}), "numpy")
    ua, ub = float(gx(0.0)), float(gx(1.0))
    a = 1e-8 if b.label == "Thomas-Fermi_1D" else 0.0
    mesh = np.linspace(a, 1, 400)
    guess = np.vstack([ua + (ub - ua) * mesh, np.full_like(mesh, ub - ua)])
    bvp = solve_bvp(lambda s, y: np.vstack([y[1], f(s, y[0], y[1])]),
                    lambda ya, yb: np.array([ya[0] - ua, yb[0] - ub]),
                    mesh, guess, tol=1e-9, max_nodes=200000)
    out = np.full_like(xs, np.nan)
    inside = xs <= 1.0
    out[inside] = bvp.sol(np.maximum(xs[inside], a))[0]
    u1, v1 = bvp.sol(1.0)
    blow = lambda s, y: abs(y[0]) - 1e3
    blow.terminal = True
    ivp = solve_ivp(lambda s, y: [y[1], f(s, y[0], y[1])], (1.0, X_MAX), [u1, v1], method="DOP853",
                    rtol=1e-12, atol=1e-14, dense_output=True, events=blow)
    m = (xs > 1.0) & (xs <= ivp.t[-1])
    out[m] = ivp.sol(xs[m])[0]
    return out


def pisr_formula(b, report):
    r = report.get(b.label, {})
    if r.get("status") != "ok" or "expr_unsimplified" not in r:
        return None
    # Formula TAL COMO LA ENCONTRO el algoritmo (antes de redondear constantes a
    # valores exactos como 1/2 o pi): la extrapolacion mide al campeon, no a la
    # presentacion.
    e = sp.sympify(eval(r["expr_unsimplified"], sp.__dict__.copy()))  # srepr -> expresion
    e = e.subs({s: x for s in e.free_symbols if s.name == "x"})
    return e, r["latex"]


def eval_expr(e, xs):
    f = sp.lambdify(x, e, modules=["numpy", "scipy"])
    v = np.asarray(f(xs.astype(complex)), dtype=complex)
    return np.full_like(xs, complex(v), dtype=complex) if v.shape == () else v


def matched_chebyshev(xs_in, u_in, target, max_deg=30):
    """Menor grado cuyo error maximo en [0,1] es <= target (el de PISR)."""
    t = 2 * xs_in - 1
    for d in range(1, max_deg + 1):
        cr = C.chebfit(t, u_in.real, d)
        ci = C.chebfit(t, u_in.imag, d)
        fit = C.chebval(t, cr) + 1j * C.chebval(t, ci)
        if np.max(np.abs(fit - u_in)) <= target:
            return d, cr, ci
    return max_deg, cr, ci


def nice(b):
    return (f"{b.display} ({b.dim}D)").replace("\\'e", "é").replace('\\"o', "ö").replace("--", "–").replace("$", "")


def main():
    rep_path = os.path.join(RESULTS, "simplification_report.json")
    report = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    probs = [b for b in BENCHMARKS if b.dim == 1 and not b.unsteady]
    xs = np.linspace(0, X_MAX, 601)
    xin = xs <= 1.0 + 1e-12
    rows, panels = [], []
    for b in probs:
        F = pisr_formula(b, report)
        if F is None:
            print(f"  [--] {b.label}: sin formula simplificada")
            continue
        expr, latex = F
        cf = closed_form(b)
        ref = eval_expr(cf, xs) if cf is not None else numeric_reference(b, xs).astype(complex)
        # control: la referencia numerica en [0,1] debe coincidir con la del binario
        grid = pd.read_csv(latest_result(f"grid_{b.label}_PISR-EMOAD.csv")) if latest_result(f"grid_{b.label}_PISR-EMOAD.csv") else None
        ref_check = np.nan
        if grid is not None and cf is None:
            gx = grid["x"].to_numpy(float)
            ref_check = float(np.max(np.abs(np.interp(gx, xs, ref.real) - grid["u_exact"].to_numpy(float))))
        pisr = eval_expr(expr, xs)
        err_p = np.abs(pisr - ref)
        in_err = float(np.nanmax(err_p[xin]))
        deg, cr, ci = matched_chebyshev(xs[xin], ref[xin], max(in_err, 1e-14))
        cheb = C.chebval(2 * xs - 1, cr) + 1j * C.chebval(2 * xs - 1, ci)
        err_c = np.abs(cheb - ref)
        scale = float(np.nanmax(np.abs(ref)))
        valid = np.isfinite(ref)
        x_end = float(xs[valid].max())

        def band(err, lo, hi):
            m = (xs > lo) & (xs <= hi) & valid
            return float(np.nanmax(err[m])) / scale if m.any() else np.nan

        rows.append({"problem": b.label, "closed_form": cf is not None, "cheb_degree": deg,
                     "in_pisr": in_err / scale, "in_cheb": float(np.nanmax(err_c[xin])) / scale,
                     "out125_pisr": band(err_p, 1.0, 1.25), "out125_cheb": band(err_c, 1.0, 1.25),
                     "out150_pisr": band(err_p, 1.0, 1.5), "out150_cheb": band(err_c, 1.0, 1.5),
                     "x_end": x_end, "ref_check": ref_check})
        panels.append((b, ref, pisr, cheb, deg, err_p, err_c, valid))
        print(f"  [OK] {b.label:<24} dentro: PISR {in_err/scale:.1e} | x<=1.5: PISR {rows[-1]['out150_pisr']:.1e} "
              f"vs Chebyshev(g{deg}) {rows[-1]['out150_cheb']:.1e}"
              + (f"  [ref vs binario {ref_check:.1e}]" if np.isfinite(ref_check) else ""))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "extrapolation.csv"), index=False)
    make_table(df)
    make_figures(panels)


def sci(v):
    if v is None or not np.isfinite(v):
        return r"\textemdash"
    if v == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(v))))
    return rf"${v / 10 ** e:.1f}\times10^{{{e}}}$"


def make_table(df):
    from benchmarks import BY_LABEL
    rows = []
    for _, r in df.iterrows():
        b = BY_LABEL[r["problem"]]
        win = r["out150_pisr"] < r["out150_cheb"]
        p150 = sci(r["out150_pisr"]); c150 = sci(r["out150_cheb"])
        if win:
            p150 = r"\textbf{" + p150 + "}"
        else:
            c150 = r"\textbf{" + c150 + "}"
        note = "" if r["x_end"] >= 1.5 - 1e-9 else rf"$^\dagger$"
        rows.append(rf"{b.label_tex}{note} & {sci(r['in_pisr'])} & {int(r['cheb_degree'])} & {sci(r['in_cheb'])} & "
                    rf"{sci(r['out125_pisr'])} & {sci(r['out125_cheb'])} & {p150} & {c150} \\")
    lines = [
        r"\begin{table}[H]", r"\centering\small",
        r"\caption{Extrapolation beyond the domain $[0,1]$ used in the search. Maximum error relative to "
        r"$\max|u|$. The Chebyshev polynomial is fitted to the reference on $[0,1]$ with the \emph{lowest degree "
        r"that matches PISR-NSGA-II's accuracy inside the domain}, so both start equally accurate; the difference "
        r"outside is what knowing the equation buys. Lower error in $[1,1.5]$ in bold. "
        r"$^\dagger$the reference solution blows up before $x=1.5$; the interval is truncated there.}",
        r"\label{tab:extrapolation}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
        r" & \multicolumn{3}{c}{\textbf{Inside $[0,1]$}} & \multicolumn{2}{c}{\textbf{$(1,1.25]$}} & \multicolumn{2}{c}{\textbf{$(1,1.5]$}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
        r"\textbf{Problem} & PISR & degree & Chebyshev & PISR & Chebyshev & PISR & Chebyshev \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}",
    ]
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, "extrapolation_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  [OK] report/tables/extrapolation_table.tex")


def make_figures(panels):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                         "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})
    n = len(panels)
    cols = 4
    rows = (n + cols - 1) // cols
    for kind in ("solutions", "errors"):
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.1 * rows), layout="constrained")
        axes = np.atleast_1d(axes).ravel()
        for ax, (b, ref, pisr, cheb, deg, ep, ec, valid) in zip(axes, panels):
            xs = np.linspace(0, X_MAX, len(ref))
            ax.axvspan(1.0, X_MAX, color="#F4D03F", alpha=0.12, lw=0)
            ax.axvline(1.0, color="#B7950B", lw=0.8, ls=":")
            if kind == "solutions":
                lo, hi = np.nanmin(ref.real[valid]), np.nanmax(ref.real[valid])
                pad = 0.35 * (hi - lo + 1e-9)
                ax.plot(xs, ref.real, color="#1B1B1B", lw=3, alpha=0.3, label="reference")
                ax.plot(xs, pisr.real, color="#2E86C1", lw=1.6, ls="--", label="PISR-NSGA-II")
                ax.plot(xs, cheb.real, color="#E67E22", lw=1.3, ls=":", label=f"Chebyshev (deg {deg})")
                ax.set_ylim(lo - pad, hi + pad)
                ax.set_ylabel("Re u" if b.complex_field else "u")
            else:
                ax.semilogy(xs, np.clip(ep, 1e-16, None), color="#2E86C1", lw=1.4, label="PISR-NSGA-II")
                ax.semilogy(xs, np.clip(ec, 1e-16, None), color="#E67E22", lw=1.2, ls=":", label=f"Chebyshev (deg {deg})")
                ax.set_ylabel("|error|")
            ax.set_title(nice(b))
            ax.set_xlabel("x")
            ax.set_xlim(0, X_MAX)
            ax.legend(loc="best", fontsize=7)
        for ax in axes[n:]:
            ax.set_visible(False)
        title = ("Inside and outside the training domain (shaded: x > 1)" if kind == "solutions"
                 else "Pointwise error inside and outside the training domain (shaded: x > 1)")
        fig.suptitle(title, fontsize=13, fontweight="bold")
        out = os.path.join(FIGS, f"extrapolation_{kind}.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
