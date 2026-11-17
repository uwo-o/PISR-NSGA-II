#!/usr/bin/env python3
"""
report/simplify_expressions.py — Forma final simplificada de cada campeon.

Para cada problema de la suite (report/benchmarks.py):
  1. Lee el arbol del campeon:
       * results/expr_<P>_PISR-EMOAD.sym  (exacto, exportado por el binario), o
       * results/expr_<P>_PISR-EMOAD.tex  (corridas viejas: LaTeX con 3 decimales).
  2. Arma la solucion final U: aplica el ansatz de frontera si el campeon lo
     usaba (1D: g(0) + x(g(1)-g(0)) + x(x-1)N; IVP: u0 + u0'x + x^2 N;
     2D: Coons[g] + x(1-x)y(1-y)N) y toma Re(U) si el campo es real.
  3. Simplifica con SymPy: varias estrategias (expand, simplify, together,
     collect, trigsimp...), constantes a 4 cifras, terminos despreciables fuera
     y constantes "limpias" (enteros, racionales chicos, multiplos de pi).
  4. VERIFICA cada candidato contra la grilla exportada del campeon
     (results/grid_<P>_PISR-EMOAD.csv): solo se acepta si la diferencia maxima
     con el campeon queda dentro de la precision de presentacion. Nunca se usa
     la solucion de referencia: la simplificacion no puede "mejorar" la
     respuesta, solo escribirla mejor.

Salida: results/expr_<P>_simplified.tex y results/simplification_report.json.
"""
import json
import os
import re
import signal
import sys
import warnings

import numpy as np
import pandas as pd
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import BENCHMARKS, BOUNDARY, latest_result  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# Dominio [0,1]^d x [0,1]: variables no negativas (x^{1.455} es real, sin ramas).
x, y, t = sp.symbols("x y t", nonnegative=True)
SYMS = {"x": x, "y": y, "t": t, "z": sp.Symbol("z", real=True)}
PARSE_MISMATCH = 0.05   # si el LaTeX leido no reproduce la grilla, no se simplifica
DISPLAY_TOL = 1e-3      # tolerancia relativa de presentacion (≈ 3 decimales)
SIG_FIGS = 4


class Timeout(Exception):
    pass


def with_timeout(fn, seconds=20):
    def handler(*_):
        raise Timeout()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        return fn()
    except Timeout:
        return None
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ─── Lector del LaTeX que produce print_formal (corridas viejas) ──────────────
TOKEN_RE = re.compile(r"""
    (?P<ws>\s+)|
    (?P<sum>\\sum_\{n=1\}\^\{(?P<nsum>\d+)\}\s*C_n)|
    (?P<frac>\\frac)|(?P<cdot>\\cdot)|
    (?P<lpar>\\left\(|\()|(?P<rpar>\\right\)|\))|
    (?P<re>\\operatorname\{Re\})|
    (?P<func>\\(?:exp|sin|cos|tanh|sinh|cosh|log|ln))(?![a-zA-Z])|
    (?P<pi>\\pi)|
    (?P<phys>\\hbar|k_B|\\varepsilon_0|G|c(?![a-z]))|
    (?P<poly>(?P<pname>[PHTL])_(?P<pdeg>\d+))|
    (?P<num>\d+\.\d*|\d+)|
    (?P<var>[xyt])|(?P<imag>i)|(?P<e>e)|
    (?P<op>[+\-^{}])
""", re.X)
POLY = {"P": sp.legendre, "H": sp.hermite, "T": sp.chebyshevt, "L": sp.laguerre}


class LatexParser:
    def __init__(self, s):
        s = s.strip().replace("$$", "").replace("$", "")
        s = re.sub(r"\\hat\{u\}\([^)]*\)\s*=", "", s)
        self.toks, pos = [], 0
        while pos < len(s):
            m = TOKEN_RE.match(s, pos)
            if not m:
                raise ValueError(f"token desconocido en: {s[pos:pos + 20]!r}")
            pos = m.end()
            kind = m.lastgroup
            if kind in ("nsum", "pname", "pdeg"):
                kind = {"nsum": "sum", "pname": "poly", "pdeg": "poly"}[kind]
            if m.group("ws"):
                continue
            for k in ("sum", "frac", "cdot", "lpar", "rpar", "re", "func", "pi", "phys", "poly", "num", "var", "imag", "e", "op"):
                if m.group(k) is not None:
                    self.toks.append((k, m.group(0), m))
                    break
        self.i = 0
        self.unknowns = []

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else ("eof", "", None)

    def take(self, val=None):
        tok = self.peek()
        if val is not None and tok[1] != val and not (val == "(" and tok[0] == "lpar") and not (val == ")" and tok[0] == "rpar"):
            raise ValueError(f"se esperaba {val!r}, vino {tok[1]!r}")
        self.i += 1
        return tok

    def parse(self):
        e = self.expr()
        if self.peek()[0] != "eof":
            raise ValueError(f"sobra: {self.peek()[1]!r}")
        return e

    def top_terms(self):
        """Sumandos de primer nivel con su signo impreso: [(+1|-1, termino)]."""
        out = []
        sign = 1
        if self.peek()[0] == "op" and self.peek()[1] in "+-":
            sign = -1 if self.take()[1] == "-" else 1
        out.append((sign, self.term()))
        while self.peek()[0] == "op" and self.peek()[1] in "+-":
            sign = -1 if self.take()[1] == "-" else 1
            out.append((sign, self.term()))
        if self.peek()[0] != "eof":
            raise ValueError("sobra texto")
        return out

    def expr(self):
        sign = 1
        if self.peek()[1] in "+-" and self.peek()[0] == "op":
            sign = -1 if self.take()[1] == "-" else 1
        acc = sign * self.term()
        while self.peek()[0] == "op" and self.peek()[1] in "+-":
            op = self.take()[1]
            acc = acc + self.term() if op == "+" else acc - self.term()
        return acc

    def starts_factor(self):
        k, v, _ = self.peek()
        return k in ("sum", "frac", "lpar", "re", "func", "pi", "phys", "poly", "num", "var", "imag", "e") or (k == "op" and v == "{")

    def term(self):
        acc = self.factor()
        while True:
            if self.peek()[0] == "cdot":
                self.take()
                sign = 1
                if self.peek()[1] == "-":
                    self.take(); sign = -1
                acc = acc * sign * self.factor()
            elif self.starts_factor():
                acc = acc * self.factor()
            else:
                return acc

    def factor(self):
        a = self.atom()
        while self.peek()[1] == "^":
            self.take("^"); self.take("{")
            e = self.expr()
            self.take("}")
            a = a ** e
        return a

    def atom(self):
        k, v, m = self.take()
        if k == "num":
            return sp.Float(v)
        if k == "var":
            return SYMS[v]
        if k == "imag":
            return sp.I
        if k == "e":
            return sp.E
        if k == "pi":
            return sp.pi
        if k == "phys":  # mismos valores que TerminalNode::eval_t
            return sp.Float({"G": "6.67430e-11", "c": "299792458.0", "\\hbar": "1.0545718e-34",
                             "k_B": "1.380649e-23", "\\varepsilon_0": "8.8541878e-12"}[v])
        if k == "lpar":
            e = self.expr(); self.take(")")
            return e
        if k == "op" and v == "{":
            e = self.expr(); self.take("}")
            return e
        if k == "frac":
            self.take("{"); a = self.expr(); self.take("}")
            self.take("{"); b = self.expr(); self.take("}")
            return a / b
        if k == "re":
            self.take("("); e = self.expr(); self.take(")")
            return sp.re(e)
        if k == "func":
            f = {"exp": sp.exp, "sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh, "sinh": sp.sinh,
                 "cosh": sp.cosh, "log": sp.log, "ln": sp.log}[v[1:]]
            self.take("(")
            # GAUSSIAN se imprime "\exp(-0.5 {arg}^{2})": el parser general lo cubre.
            e = self.expr(); self.take(")")
            return f(e)
        if k == "poly":
            self.take("("); e = self.expr(); self.take(")")
            return POLY[m.group("pname")](int(m.group("pdeg")), e)
        if k == "sum":
            # \sum_{n=1}^{N} C_n f : los C_n no se imprimen; se ajustan luego
            # contra la grilla del campeon (U es lineal en ellos).
            c = sp.Symbol(f"C{len(self.unknowns)}")
            self.unknowns.append(c)
            return c * self.factor()
        raise ValueError(f"token inesperado {v!r}")


# ─── Ansatz ───────────────────────────────────────────────────────────────────
def boundary_parts(b):
    """(A, B) con U = A + B*N, o None si el problema no usa ansatz."""
    entry = BOUNDARY.get(b.label)
    if not entry:
        return None
    kind, g = entry
    if kind == "ivp":
        u0, du0 = (sp.sympify(v) for v in g.split(","))
        return u0 + du0 * x, x ** 2
    gx = sp.sympify(g, locals={"x": x, "y": y})
    if b.dim == 1:
        g0, g1 = gx.subs(x, 0), gx.subs(x, 1)
        return g0 + x * (g1 - g0), x * (x - 1)
    G = lambda a, c: gx.subs({x: a, y: c}, simultaneous=True)
    L = ((1 - x) * G(0, y) + x * G(1, y) + (1 - y) * G(x, 0) + y * G(x, 1)
         - (1 - x) * (1 - y) * G(0, 0) - x * (1 - y) * G(1, 0) - (1 - x) * y * G(0, 1) - x * y * G(1, 1))
    return L, x * (1 - x) * y * (1 - y)


# ─── Evaluacion sobre la grilla del campeon ──────────────────────────────────
class Grid:
    def __init__(self, b):
        p = latest_result(f"grid_{b.label}_PISR-EMOAD.csv")
        df = pd.read_csv(p)
        self.x = df["x"].to_numpy(float)
        self.y = df["y"].to_numpy(float) if "y" in df.columns else np.zeros_like(self.x)
        self.t = np.full_like(self.x, 0.5 if b.unsteady else 0.0)
        u = df["u_approx"].to_numpy(float)
        if "u_approx_im" in df.columns:
            u = u + 1j * df["u_approx_im"].to_numpy(float)
        self.u = u
        self.scale = max(float(np.nanmax(np.abs(u))), 1e-12)

    def eval(self, e):
        e = sp.sympify(e)
        try:
            f = sp.lambdify((x, y, t), e, modules=["numpy", "scipy"])
            v = np.asarray(f(self.x.astype(complex), self.y.astype(complex), self.t.astype(complex)), dtype=complex)
            if v.shape == ():
                v = np.full_like(self.u, complex(v), dtype=complex)
            return v
        except Exception:
            # funciones sin traduccion compleja en numpy/scipy (p. ej. airyai):
            # evaluacion punto a punto con mpmath
            f = sp.lambdify((x, y, t), e, modules=["mpmath"])
            return np.array([complex(f(a, c, d)) for a, c, d in zip(self.x, self.y, self.t)])

    def dev(self, e):
        """max |e - campeon| / escala, sobre los puntos finitos de la grilla."""
        v = self.eval(e)
        ok = np.isfinite(self.u)
        if not np.all(np.isfinite(v[ok])):
            return np.inf
        return float(np.max(np.abs(v[ok] - self.u[ok]))) / self.scale


# ─── Simplificacion ───────────────────────────────────────────────────────────
def round_floats(e, sig=SIG_FIGS):
    rep = {}
    for f in e.atoms(sp.Float):
        v = float(f)
        rep[f] = sp.Float(0) if v == 0 else sp.Float(float(f"{v:.{sig - 1}e}"), sig + 2)
    return e.xreplace(rep)


def nice_value(v):
    """Candidatos 'limpios' para la constante v, del mas simple al menos."""
    out = []
    for q in (1, 2, 3, 4, 6):
        p = round(v * q)
        if p != 0 or abs(v) < 1e-12:
            out.append(sp.Rational(p, q))
    for q in (1, 2):
        for base in (sp.pi, sp.pi ** 2, sp.sqrt(2), sp.sqrt(3)):
            p = round(v * q / float(base))
            if p != 0 and abs(p) <= 4:
                out.append(sp.Rational(p, q) * base)
    out = [c for c in out if abs(float(c) - v) <= 2e-3 * max(abs(v), 1e-3)]
    return sorted(set(out), key=lambda c: (sp.count_ops(c), abs(float(c) - v)))


def complexity(e):
    """Operaciones + penalizacion por constantes 'feas' (decimales, enteros grandes, fracciones)."""
    ugly = len(e.atoms(sp.Float))
    ugly += sum(1 for n in e.atoms(sp.Integer) if abs(int(n)) > 12)
    ugly += sum(1 for n in e.atoms(sp.Rational) if not n.is_Integer and n.q > 6)
    return sp.count_ops(e, visual=False) + 0.4 * ugly


def drop_negligible(e, grid, tol):
    """Quita sumandos cuyo aporte maximo en la grilla es despreciable."""
    if not isinstance(e, sp.Add):
        return e
    keep = []
    for term in e.args:
        try:
            m = np.nanmax(np.abs(grid.eval(term))) / grid.scale
        except Exception:
            m = np.inf
        if not (m < 0.2 * tol):
            keep.append(term)
    return sp.Add(*keep) if keep else sp.Integer(0)


def snap_constants(e, grid, tol):
    for f in sorted(e.atoms(sp.Float), key=lambda c: -abs(float(c))):
        for cand in nice_value(float(f)):
            trial = e.xreplace({f: cand})
            if grid.dev(trial) <= tol:
                e = trial
                break
    return e


def normalize_fraction(e):
    """p/q con el termino constante de q igual a 1 (forma de Pade legible)."""
    n, d = sp.fraction(sp.cancel(sp.together(e)))
    vars_ = [v for v in (x, y, t) if d.has(v)]
    if not vars_:
        return e
    c = d.subs({v: 0 for v in vars_})
    if c == 0:
        return e
    return sp.expand(n / c) / sp.expand(d / c)


def strategies(e):
    vars_ = [v for v in (x, y, t) if e.has(v)]
    out = [("original", lambda: e),
           ("expand", lambda: sp.expand(e)),
           ("simplify", lambda: sp.simplify(e)),
           ("together", lambda: sp.cancel(sp.together(e))),
           ("collect", lambda: sp.factor_terms(sp.collect(sp.expand(e), vars_))),
           ("trigsimp", lambda: sp.trigsimp(sp.expand(e))),
           ("powsimp", lambda: sp.powsimp(sp.expand(e), force=True)),
           ("pade", lambda: normalize_fraction(e)),
           ("factor", lambda: sp.factor(e)),
           ("factor-collect", lambda: sp.factor_terms(sp.collect(sp.expand(e), [sp.factor(x * (x - 1))] + vars_))),
           ("apart", lambda: sp.apart(sp.cancel(sp.together(e)), x) if e.has(x) else e),
           ("exp-rewrite", lambda: sp.simplify(sp.powsimp(sp.expand(e.rewrite(sp.exp)), force=True)))]
    return out


def simplify_expr(e, grid, tol):
    """Mejor forma (menor complejidad) cuya distancia al campeon es <= tol."""
    best, best_c, best_name = None, np.inf, None
    for name, fn in strategies(e):
        cand = with_timeout(fn, 25)
        if cand is None:
            continue
        for post in (lambda c: c, lambda c: sp.expand(c)):
            c = with_timeout(lambda: round_floats(post(cand)), 10)
            if c is None:
                continue
            c = drop_negligible(c, grid, tol)
            c = snap_constants(c, grid, tol)
            c2 = with_timeout(lambda: sp.simplify(c), 15)
            for opt in (c, c2):
                if opt is None:
                    continue
                opt = round_floats(opt)
                try:
                    d = grid.dev(opt)
                except Exception:
                    continue
                if d <= tol and complexity(opt) < best_c:
                    best, best_c, best_name = opt, complexity(opt), name
    return best, best_name


def to_real(e):
    return sp.re(sp.expand_complex(e))


def load_champion(b):
    sym = latest_result(f"expr_{b.label}_PISR-EMOAD.sym")
    tex = latest_result(f"expr_{b.label}_PISR-EMOAD.tex")
    if sym and (not tex or os.path.getmtime(sym) >= os.path.getmtime(tex)):
        meta = dict(line.rstrip("\n").split("=", 1) for line in open(sym) if "=" in line)
        tree = sp.sympify(meta["tree"], locals=SYMS)
        return "sym", tree, [], meta.get("uses_ansatz") == "1"
    if tex:
        p = LatexParser(open(tex).read())
        return "latex", p.parse(), p.unknowns, None
    return None


def recover_signs(b, grid, tex):
    try:
        terms = [term for _, term in LatexParser(tex).top_terms()]
    except Exception:
        return None
    if b.complex_field is False:
        terms = [to_real(tm) for tm in terms]
    if len(terms) > 14:
        return None
    A = np.stack([grid.eval(tm) for tm in terms], axis=1)
    ok = np.all(np.isfinite(A), axis=1) & np.isfinite(grid.u)
    A, u = A[ok], grid.u[ok]
    # Busqueda exhaustiva de signos (+-1 por sumando; los terminos pueden ser
    # colineales, asi que minimos cuadrados no queda determinado).
    n = len(terms)
    combos = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    devs = np.max(np.abs(A @ combos.T - u[:, None]), axis=0) / grid.scale
    k = int(np.argmin(devs))
    if devs[k] > PARSE_MISMATCH:
        return None
    return sp.Add(*[int(s_) * tm for s_, tm in zip(combos[k], terms)])


def process(b):
    loaded = load_champion(b)
    if not loaded:
        return {"status": "sin expresion"}
    source, tree, unknowns, uses_ansatz = loaded
    grid = Grid(b)
    parts = boundary_parts(b)

    # Representacion: el .sym la trae; para el LaTeX viejo, en 1D el .tex ya es U
    # y en 2D puede ser N (ansatz) o U (raw): se decide contra la grilla.
    options = []
    if source == "sym":
        options = [("ansatz" if uses_ansatz else "direct", parts if uses_ansatz else None)]
    elif b.dim == 1 or parts is None:
        options = [("direct", None)]
    else:
        options = [("direct", None), ("ansatz", parts)]

    best = None
    for rep, ab in options:
        N = tree
        if unknowns:  # coeficientes de series no impresos: minimos cuadrados
            U_lin = N if ab is None else ab[0] + ab[1] * N
            cols = [grid.eval(sp.diff(U_lin, c)) for c in unknowns]
            base = grid.eval(U_lin.subs({c: 0 for c in unknowns}))
            A = np.stack(cols, axis=1)
            coef, *_ = np.linalg.lstsq(A, grid.u - base, rcond=None)
            N = N.subs({c: sp.Float(complex(v).real, 15) + sp.I * sp.Float(complex(v).imag, 15)
                        for c, v in zip(unknowns, coef)})
        U = N if ab is None else ab[0] + ab[1] * N
        if b.complex_field is False:
            U, N = to_real(U), (to_real(N) if ab is not None else to_real(N))
        d0 = grid.dev(U)
        if best is None or d0 < best[0]:
            best = (d0, rep, ab, N, U)
    d0, rep, ab, N, U = best
    if d0 > PARSE_MISMATCH and source == "latex":
        # El printer viejo podia perder parentesis en "A - (B + C)": solo se
        # alteran SIGNOS de los sumandos de primer nivel. Se recuperan como
        # coeficientes de minimos cuadrados contra la grilla del campeon y se
        # aceptan solo si salen +-1 exactos.
        fixed = recover_signs(b, grid, open(latest_result(f"expr_{b.label}_PISR-EMOAD.tex")).read())
        if fixed is not None:
            U, d0, rep, ab, N = fixed, grid.dev(fixed), "direct", None, fixed
    if d0 > PARSE_MISMATCH:
        # El LaTeX viejo era ambiguo (printer sin parentesis en A-(B+C)): lo
        # leido no es el campeon. No se publica una forma "simplificada" falsa.
        return {"status": "parse_mismatch", "source": source, "dev_parsed": d0}
    tol = max(2.0 * d0, DISPLAY_TOL)

    if ab is None:
        S, how = simplify_expr(U, grid, tol)
        body = S if S is not None else U
        latex = sp.latex(body)
        final = body
    else:
        # U = A + B N: se simplifica solo N (A y B son la construccion de borde
        # conocida) y se verifica U completa.
        class Shifted:
            def __init__(self):
                self.scale = grid.scale
            def eval(self, e):
                return grid.eval(ab[0] + ab[1] * e)
            def dev(self, e):
                return grid.dev(ab[0] + ab[1] * e)
        sg = Shifted()
        S, how = simplify_expr(N, sg, tol)
        Nf = S if S is not None else N
        final = ab[0] + ab[1] * Nf
        if b.dim == 1:
            S2, how2 = simplify_expr(final, grid, tol)
            if S2 is not None:
                final, how = S2, how2
            latex = sp.latex(final)
        else:
            latex = (r"\mathcal{C}[g](x,y) + x(1-x)\,y(1-y)\left(" + sp.latex(Nf) + r"\right)")
    d_final = grid.dev(final)
    return {"status": "ok", "source": source, "representation": rep, "strategy": how,
            "dev_parsed": d0, "dev_final": d_final, "tolerance": tol,
            "ops_before": int(sp.count_ops(U)), "ops_after": int(sp.count_ops(final)),
            "latex": latex, "expr": sp.srepr(final), "expr_unsimplified": sp.srepr(U)}


def main():
    only = sys.argv[1:]
    report = {}
    for b in BENCHMARKS:
        if only and b.label not in only:
            continue
        try:
            r = process(b)
        except Exception as ex:  # nunca bloquea el reporte
            r = {"status": f"error: {ex}"}
        report[b.label] = r
        if r.get("status") == "ok":
            with open(os.path.join(RESULTS, f"expr_{b.label}_simplified.tex"), "w") as f:
                f.write(f"$$ {r['latex']} $$\n")
            print(f"  [OK] {b.label:<28} ops {r['ops_before']:>4} -> {r['ops_after']:<4} "
                  f"dev {r['dev_final']:.1e} ({r['representation']}, {r['source']})  {r['latex'][:90]}")
        else:
            print(f"  [--] {b.label:<28} {r['status']}")
    path = os.path.join(RESULTS, "simplification_report.json")
    old = json.load(open(path)) if os.path.exists(path) and only else {}
    old.update(report)
    with open(path, "w") as f:
        json.dump(old, f, indent=2, default=float)


if __name__ == "__main__":
    main()
