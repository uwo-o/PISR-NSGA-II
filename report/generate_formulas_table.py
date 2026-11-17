#!/usr/bin/env python3
"""
report/generate_formulas_table.py — Tabla de expresiones descubiertas.

Una fila por problema de la suite (report/benchmarks.py), agrupada por familia:
expresion del campeon de PISR-NSGA-II y solucion de referencia. Las formulas de
PySR/PySINDy se agregan solo si sus logs son posteriores a la redefinicion de
los problemas (benchmarks.is_fresh); antes resolvian otro problema.
"""
import glob
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


def sota_formula(b, method):
    prefix = "pysr" if method == "PySR" else "sindy"
    base = os.path.join(ROOT_DIR, "sota", "results", f"{prefix}_{b.base}_{b.dim}D")
    cands = [base + ".log"] + sorted(glob.glob(base + "_run*.log"))
    cands = [c for c in cands if os.path.exists(c) and is_fresh(c)]
    if not cands:
        return None
    lines = open(cands[-1]).read().splitlines()
    for i, line in enumerate(lines):
        if "Best Discovered Equation" in line and i + 1 < len(lines):
            eq = lines[i + 1].strip()
            try:
                import sympy as sp
                expr = sp.sympify(eq)
                expr = expr.xreplace({n: round(n, 3) for n in expr.atoms(sp.Number)})
                return sp.latex(expr)
            except Exception:
                return r"\texttt{" + eq.replace("_", r"\_").replace("^", r"\^{}")[:100] + "}"
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
            for m in ("PySR", "PySINDy"):
                f = sota_formula(b, m)
                if f:
                    entries.append((m, f))
                    any_sota = True
            entries.append(("Reference", b.reference))
            for k, (method, f) in enumerate(entries):
                name = b.label_tex if k == 0 else ""
                body = cell(f) if f else r"\textit{no champion}"
                if method == "Reference":
                    method = r"\textcolor{muted}{Reference}"
                    body = r"\textcolor{muted}{" + body + "}"
                rows.append(rf"{name} & {method} & {body} \\")
            rows.append(r"\addlinespace[3pt]")

    caption = (r"Discovered closed-form solutions $\hat u$ (hall-of-fame champion after constant polishing), "
               r"simplified with SymPy and verified against the champion on its export grid "
               r"(maximum relative difference below $10^{-3}$; constants to 4 significant figures). "
               r"For 2D problems whose champion uses the boundary ansatz, $\mathcal{C}[g]$ is the transfinite "
               r"(Coons) interpolation of the boundary data $g$, so $\hat u = g$ on $\partial[0,1]^2$ exactly.")
    if not any_sota:
        caption += (r" External symbolic baselines (PySR, PySINDy) are omitted: their runs predate the "
                    r"current problem definitions.")
    lines = [
        r"\begin{longtable}{@{}L{0.19\linewidth}L{0.17\linewidth}L{0.60\linewidth}@{}}",
        rf"\caption{{{caption}}}\label{{tab:formulas}}\\",
        r"\toprule",
        r"\textbf{Problem} & \textbf{Method} & \textbf{Expression} $\hat u$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Problem} & \textbf{Method} & \textbf{Expression} $\hat u$ \\",
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
