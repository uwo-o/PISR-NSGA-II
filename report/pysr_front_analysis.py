#!/usr/bin/env python3
"""Analisis justo de PINN+PySR: que encuentra PySR en su frente de Pareto, no
solo lo que elige su criterio por defecto ("best").

Para cada semilla y problema, desde <EDP>_pinn_pysr_front.csv (complejidad,
perdida contra el PINN, formula), tres selecciones:
  best          la de PySR por defecto (columna expr de <EDP>_pinn_pysr.csv);
  parsimoniosa  la MAS CORTA cuyo ajuste al PINN esta a <= PARS_FACTOR veces la
                mejor perdida del frente. Solo usa datos del PINN: es una regla
                que cualquier usuario de PySR podria aplicar;
  oraculo       alguna formula del frente ES la solucion exacta (usa la
                referencia: cota superior de lo que PySR llego a encontrar, no
                un metodo aplicable).
Campos complejos: cada parte (re, im) por separado; la formula es re + i im.

Salidas: results/pysr_front_analysis.csv y report/tables/pysr_front_table.tex.
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
import interpretability as I  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "report", "tables")
PARS_FACTOR = 10.0
LOC = {"x": I.x, "y": I.y, "t": I.t}


def parse(s):
    try:
        return sp.sympify(str(s), locals=LOC)
    except Exception:
        return None


def part_exact(exact, part):
    e = sp.expand_complex(exact)
    return sp.re(e) if part == "re" else sp.im(e)


def analyse(job):
    label, d = job
    b = BY_LABEL[label]
    fpath = os.path.join(d, f"{label}_pinn_pysr_front.csv")
    if not os.path.exists(fpath) or not is_fresh(fpath):
        return None
    front = pd.read_csv(fpath)
    pts = I.Points(b)
    ref = pd.read_csv(os.path.join(RESULTS_DIR, "reference", f"{label}.csv"))
    ref = ref[ref.kind == "dom"]
    uref = ref.u_re.to_numpy() + 1j * ref.u_im.to_numpy()
    exact = I.EXACT.get(label)
    parts = ["re", "im"] if (front.part == "im").any() else ["re"]

    chosen, oracle = {}, {}
    for part in parts:
        f = front[front.part == part].copy()
        f["expr"] = [parse(s) for s in f.equation]
        f = f[f.expr.notna()]
        if f.empty:
            return None
        lmin = float(f.loss.min())
        ok = f[f.loss <= PARS_FACTOR * max(lmin, 1e-300)]
        chosen[part] = ok.sort_values("complexity").iloc[0].expr
        if exact is not None:
            oracle[part] = any(I.recovered(e, part_exact(exact, part), pts) for e in f.expr)

    U = chosen["re"] + (sp.I * chosen["im"] if "im" in chosen else 0)
    u = pts.eval(U)
    sol = float(np.mean(np.abs(u - uref) ** 2)) if np.all(np.isfinite(u)) else np.nan
    return {
        "problem": label, "run": int(d.rsplit("_", 1)[1]),
        "pars_solution_mse": sol,
        "pars_nodes": I.nodes(U),
        "pars_recovered": I.recovered(U, exact, pts) if exact is not None else np.nan,
        "pars_extrapolation": I.extrapolation_error(b, U),
        "oracle_recovered": all(oracle.values()) if exact is not None else np.nan,
        "pars_expr": str(U),
    }


def make_table(df, interp):
    def med(sub, col):
        v = sub[col].astype(float).replace([np.inf], np.nan).dropna()
        return float(v.median()) if len(v) else np.nan

    def rate(sub, col):
        v = sub[col].dropna()
        return f"{int(v.astype(bool).sum())}/{len(v)}" if len(v) else r"\textemdash"

    rows = []
    for _, title, items in by_family():
        rows.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\familyrow{{{title}}}}} \\")
        for b in items:
            a = interp[(interp.problem == b.label) & (interp.method == I.PISR)]
            p = interp[(interp.problem == b.label) & (interp.method == I.PYSR)]
            q = df[df.problem == b.label]
            has = b.label in I.EXACT
            rec = (lambda s, c: rate(s, c)) if has else (lambda s, c: r"{\color{muted}n/a}")
            rows.append(" & ".join([
                b.label_tex,
                rec(a, "recovered"), rec(p, "recovered"), rec(q, "pars_recovered"), rec(q, "oracle_recovered"),
                f"{med(a, 'nodes'):.0f}", f"{med(p, 'nodes'):.0f}", f"{med(q, 'pars_nodes'):.0f}",
            ]) + r" \\")
    lines = [
        r"\begin{table}[H]", r"\centering\small",
        r"\caption{Una mirada justa a PINN+PySR: lo que PySR encuentra en su propio frente de Pareto, no solo lo "
        r"que entrega su criterio por defecto. \emph{best}: la selección por defecto de PySR. \emph{pars.}: la "
        rf"fórmula más corta del frente cuyo ajuste al PINN está a menos de {PARS_FACTOR:.0f}$\times$ la mejor "
        r"pérdida (usa solo datos del PINN; una regla que cualquier usuario de PySR podría aplicar). "
        r"\emph{oráculo}: alguna fórmula del frente es la solución exacta (usa la referencia; una cota superior, "
        r"no un método aplicable). Recuperación exacta en semillas (problemas con forma cerrada) y tamaño "
        r"mediano de la fórmula en nodos. \emph{n/a}: el problema no tiene una forma cerrada expresable.}",
        r"\label{tab:pysrfront}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
        r" & \multicolumn{4}{c}{\textbf{Recuperación exacta}} & \multicolumn{3}{c}{\textbf{Nodos}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(l){6-8}",
        r"\textbf{Problema} & PISR & best & pars. & oráculo & PISR & best & pars. \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}",
    ]
    with open(os.path.join(TABLES, "pysr_front_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  [OK] report/tables/pysr_front_table.tex")


def main():
    dirs = [d for d in seed_dirs() if any(is_fresh(os.path.join(d, f)) for f in os.listdir(d))]
    jobs = [(b.label, d) for b in BENCHMARKS for d in dirs]
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        rows = [r for r in pool.map(analyse, jobs, chunksize=4) if r]
    if not rows:
        print("  [--] sin frentes de PySR: se omite")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "pysr_front_analysis.csv"), index=False)
    interp_path = os.path.join(RESULTS_DIR, "interpretability.csv")
    if os.path.exists(interp_path):
        make_table(df, pd.read_csv(interp_path))


if __name__ == "__main__":
    main()
