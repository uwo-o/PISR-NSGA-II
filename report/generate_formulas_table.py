#!/usr/bin/env python3
"""
report/generate_formulas_table.py — Tabla de expresiones descubiertas.

Una fila por problema de la suite (report/benchmarks.py), agrupada por familia:
expresion del campeon de PISR-NSGA-II y solucion de referencia. Las formulas de
PySR/PySINDy se agregan solo si sus logs son posteriores a la redefinicion de
los problemas (benchmarks.is_fresh); antes resolvian otro problema.
"""
import glob

import pandas as pd
import os

from benchmarks import by_family, is_fresh, latest_result

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(REPORT_DIR)
TABLES_DIR = os.path.join(REPORT_DIR, "tables")


def pisr_formula(b):
    """Forma simplificada y verificada (simplify_expressions.py) si existe y es
    posterior al campeon; si no, la expresion tal como la imprimio el binario."""
    simp = latest_result(f"expr_{b.label}_simplified.tex")
    raw = latest_result(f"expr_{b.label}_PISR-EMOAD.tex")
    if simp and (not raw or os.path.getmtime(simp) >= os.path.getmtime(raw)):
        return open(simp).read().strip().replace("$$", "").strip() or None
    path = raw
    if not path:
        return None
    s = open(path).read().strip().replace("$$", "").replace("$", "")
    for prefix in (r"\hat{u}(x,y) =", r"\hat{u}(x) ="):
        s = s.replace(prefix, "")
    return s.strip() or None


def pysr_formula(b):
    """Formula de PINN+PySR en su semilla mediana (por solution_mse), en LaTeX con 4 cifras."""
    import sympy as sp
    rows = []
    for f in glob.glob(os.path.join(ROOT_DIR, "results", "run_*", f"{b.label}_pinn_pysr.csv")):
        if not is_fresh(f):
            continue
        r = pd.read_csv(f).iloc[0]
        if bool(r.get("ok", True)) and isinstance(r.get("expr_srepr"), str) and float(r["solution_mse"]) == float(r["solution_mse"]):
            rows.append((float(r["solution_mse"]), r["expr_srepr"]))
    if not rows:
        return None
    rows.sort(key=lambda t: t[0])
    try:
        e = sp.sympify(eval(rows[(len(rows) - 1) // 2][1], sp.__dict__.copy()))
        e = e.xreplace({c: sp.Float(float(f"{float(c):.4g}"), 6) for c in e.atoms(sp.Float)})
        return sp.latex(e)
    except Exception:
        return None


def cell(expr):
    """Expresion en modo matematico, reducida al ancho de la columna si es larga."""
    return r"\adjustbox{max width=\linewidth}{$\displaystyle " + expr + r"$}"


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    rows = []
    any_sota = False
    for _, title, items in by_family():
        rows.append(rf"\multicolumn{{3}}{{l}}{{\familyrow{{{title}}}}} \\")
        for b in items:
            entries = [("PISR-NSGA-II", pisr_formula(b))]
            f = pysr_formula(b)
            if f:
                entries.append(("PINN+PySR", f))
                any_sota = True
            entries.append(("Reference", b.reference))
            for k, (method, f) in enumerate(entries):
                name = b.label_tex if k == 0 else ""
                body = cell(f) if f else r"\textit{sin campeón}"
                if method == "Reference":
                    method = r"\textcolor{muted}{Referencia}"
                    body = r"\textcolor{muted}{" + body + "}"
                rows.append(rf"{name} & {method} & {body} \\")
            rows.append(r"\addlinespace[3pt]")

    caption = (r"Soluciones en forma cerrada descubiertas $\hat u$ (campeón del Hall of Fame después del pulido "
               r"de constantes, semilla mediana de cada problema), simplificadas con SymPy y verificadas contra el "
               r"campeón en su grilla de exportación (diferencia relativa máxima bajo $10^{-3}$; constantes con 4 "
               r"cifras significativas). En los problemas 2D cuyo campeón usa el ansatz de frontera, "
               r"$\mathcal{C}[g]$ es la interpolación transfinita (de Coons) de los datos de borde $g$, así que "
               r"$\hat u = g$ exactamente en $\partial[0,1]^2$.")
    if any_sota:
        caption += (r" PINN+PySR: fórmula elegida por PySR (criterio \emph{best}) en su propia semilla "
                    r"mediana, con constantes a 4 cifras significativas.")
    lines = [
        r"\begin{longtable}{@{}L{0.19\linewidth}L{0.17\linewidth}L{0.60\linewidth}@{}}",
        rf"\caption{{{caption}}}\label{{tab:formulas}}\\",
        r"\toprule",
        r"\textbf{Problema} & \textbf{Método} & \textbf{Expresión} $\hat u$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Problema} & \textbf{Método} & \textbf{Expresión} $\hat u$ \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
        *rows,
        r"\end{longtable}",
    ]
    out = os.path.join(TABLES_DIR, "formulas_table.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [OK] {os.path.relpath(out, ROOT_DIR)}")


if __name__ == "__main__":
    main()
