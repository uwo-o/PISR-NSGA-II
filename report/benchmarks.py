"""
report/benchmarks.py — Catalogo unico de la suite de benchmarks.

Fuente de verdad para TODOS los scripts del reporte y de graficos: orden,
agrupacion por familia, ecuacion, condiciones de borde y solucion de
referencia. Refleja benchmark_suite() en src/pde_problems.cpp; si se agrega
o cambia un problema alli, se actualiza aqui y todo el reporte lo recoge.
"""
import datetime
import os
from dataclasses import dataclass

# Fecha en que se redefinieron los problemas (Airy verdadero, Lane-Emden como
# IVP, Thomas-Fermi en [0,1], Schrodinger no resonante, GL tanh(x/sqrt2), NS
# con residuo real). Resultados de lineas base externas (DeepXDE, PySR,
# PySINDy) anteriores a esta fecha resolvieron OTRO problema y no se comparan.
PROBLEMS_REDEFINED = datetime.datetime(2026, 10, 20)


def is_fresh(path: str) -> bool:
    """True si el archivo es posterior a la redefinicion de los problemas."""
    return datetime.datetime.fromtimestamp(os.path.getmtime(path)) >= PROBLEMS_REDEFINED


@dataclass(frozen=True)
class Benchmark:
    base: str          # nombre base tal como lo escribe el binario (prob.name())
    dim: int           # 1 o 2 (dimension espacial)
    family: str        # clave de FAMILIES
    display: str       # nombre legible
    equation: str      # ecuacion (LaTeX, modo matematico)
    conditions: str    # condiciones de borde / iniciales (LaTeX, modo texto)
    reference: str     # solucion de referencia (LaTeX, modo matematico)
    numerical: bool    # referencia numerica (sin forma cerrada)
    unsteady: bool = False
    complex_field: bool = False

    @property
    def label(self) -> str:
        """Etiqueta de archivo: Airy_1D, Navier-Stokes_2D, ..."""
        return f"{self.base}_{self.dim}D"

    @property
    def label_tex(self) -> str:
        return f"{self.display} ({self.dim}D)"


# Orden de presentacion de las familias.
FAMILIES = [
    ("linear",   "Problemas lineales de calibración"),
    ("nonlin",   "Problemas de contorno no lineales"),
    ("complex",  "Problemas de campo complejo (cuánticos)"),
    ("fluids",   "Dinámica de fluidos incompresibles"),
    ("singular", "Problemas singulares, rígidos y de tipo Painlev\\'e"),
]
FAMILY_TITLE = dict(FAMILIES)

_D01 = r"$u(0),\,u(1)$ dados"
_SQ = r"$u$ en $\partial[0,1]^2$"

BENCHMARKS = [
    # ── Calibracion lineal ──────────────────────────────────────────────────
    Benchmark("Laplace", 1, "linear", "Laplace",
              r"u'' = 0", _D01, r"x", False),
    Benchmark("Poisson", 1, "linear", "Poisson",
              r"u'' = -\pi^2 \sin(\pi x)", _D01, r"\sin(\pi x)", False),
    Benchmark("HarmonicOscillator", 1, "linear", "Oscilador armónico",
              r"u'' - x^2 u + u = 0", _D01, r"e^{-x^2/2}", False),
    Benchmark("Airy", 1, "linear", "Airy",
              r"u'' = x\,u", _D01, r"\mathrm{Ai}(x)", False),
    Benchmark("Airy", 2, "linear", "Airy",
              r"\nabla^2 u = (x+y)\,u", _SQ, r"\mathrm{Ai}\!\left(2^{-1/3}(x+y)\right)", False),
    # ── No lineales ─────────────────────────────────────────────────────────
    Benchmark("Fisher", 1, "nonlin", "Fisher--KPP (estacionaria)",
              r"u'' + u(1-u) = 0", _D01, r"\text{numérica}", True),
    Benchmark("Fisher", 2, "nonlin", "Fisher--KPP (estacionaria)",
              r"\nabla^2 u + u(1-u) = 0", _SQ, r"\text{numérica}", True),
    Benchmark("Duffing", 1, "nonlin", "Tipo Duffing (contorno)",
              r"u'' + u + u^3 = \sin(\pi x)", _D01, r"\text{numérica}", True),
    Benchmark("Duffing", 2, "nonlin", "Tipo Duffing (contorno)",
              r"\nabla^2 u + u + u^3 = \sin(\pi x)\sin(\pi y)", _SQ, r"\text{numérica}", True),
    Benchmark("Thomas-Fermi", 1, "nonlin", "Thomas--Fermi",
              r"u'' = u^{3/2}/\sqrt{x}", r"$u(0)=1,\ u(1)=0.424$", r"\text{numérica}", True),
    Benchmark("Thomas-Fermi", 2, "nonlin", "Tipo Thomas--Fermi",
              r"\nabla^2 u = u^{3/2}/\sqrt{r}", _SQ, r"\text{numérica}", True),
    # ── Complejos ───────────────────────────────────────────────────────────
    Benchmark("Schrodinger", 1, "complex", "Schr\\\"odinger",
              r"u'' + 4u = 0", _D01, r"e^{2ix}", False, complex_field=True),
    Benchmark("Schrodinger", 2, "complex", "Schr\\\"odinger",
              r"\nabla^2 u + 8u = 0", _SQ, r"e^{2i(x+y)}", False, complex_field=True),
    Benchmark("Schrodinger-Unsteady", 1, "complex", "Schr\\\"odinger (no estacionaria)",
              r"i\,u_t + u_{xx} = 0", r"$u(x,0)$, $u(0,t)$, $u(1,t)$",
              r"e^{i\pi(x-\pi t)}", False, unsteady=True, complex_field=True),
    Benchmark("Schrodinger-Unsteady", 2, "complex", "Schr\\\"odinger (no estacionaria)",
              r"i\,u_t + \nabla^2 u = 0", r"$u(x,y,0)$, $u$ en $\partial[0,1]^2$",
              r"e^{i\pi(x+y-2\pi t)}", False, unsteady=True, complex_field=True),
    # ── Fluidos ─────────────────────────────────────────────────────────────
    Benchmark("Navier-Stokes", 2, "fluids", "Navier--Stokes (Kovasznay)",
              r"\psi_y \nabla^2\psi_x - \psi_x \nabla^2\psi_y = \nu \nabla^4 \psi",
              r"$\psi,\ \partial_n\psi$ en $\partial[0,1]^2$, $\nu=0.05$ ($\mathrm{Re}=20$)",
              r"y - \tfrac{1}{2\pi}e^{\lambda x}\sin(2\pi y),\ \lambda=\tfrac{\mathrm{Re}}{2}-\sqrt{\tfrac{\mathrm{Re}^2}{4}+4\pi^2}", False),
    Benchmark("Navier-Stokes-Unsteady", 2, "fluids", "Navier--Stokes (Taylor--Green)",
              r"\omega_t + \mathbf{u}\cdot\nabla\omega = \nu \nabla^2 \omega,\ \omega = -\nabla^2\psi",
              r"$\psi(x,y,0)$, $\psi,\ \partial_n\psi$ en $\partial[0,1]^2$",
              r"\sin(\pi x)\sin(\pi y)\,e^{-2\pi^2 \nu t}", False, unsteady=True),
    # ── Singulares / rigidos ────────────────────────────────────────────────
    Benchmark("Lane-Emden", 1, "singular", "Lane--Emden ($n=3$)",
              r"u'' + \tfrac{2}{x}u' + u^3 = 0", r"$u(0)=1,\ u'(0)=0$ (IVP)", r"\text{numérica}", True),
    Benchmark("Troesch", 1, "singular", "Troesch ($\\lambda=3$)",
              r"u'' = 3\sinh(3u)", r"$u(0)=0,\ u(1)=1$", r"\text{numérica}", True),
    Benchmark("Ginzburg-Landau", 1, "singular", "Ginzburg--Landau",
              r"u'' + u - u^3 = 0", _D01, r"\tanh(x/\sqrt{2})", False),
    Benchmark("Painleve-I", 1, "singular", "Painlev\\'e I",
              r"u'' = u^2 + x", r"$u(0)=0,\ u(1)=\tfrac12$", r"\text{numérica}", True),
]

BY_LABEL = {b.label: b for b in BENCHMARKS}


def by_family():
    """[(clave, titulo, [Benchmark, ...]), ...] en orden de presentacion."""
    return [(k, t, [b for b in BENCHMARKS if b.family == k]) for k, t in FAMILIES]


def labels():
    return [b.label for b in BENCHMARKS]


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


_SEED_CACHE = {}


def seed_errors(label: str):
    """[(run, solution_mse)] de PISR-NSGA-II para `label`, ordenado por error,
    desde results/all_runs_summary.csv (solo corridas con campeon)."""
    if label in _SEED_CACHE:
        return _SEED_CACHE[label]
    import csv
    path = os.path.join(RESULTS_DIR, "all_runs_summary.csv")
    out = []
    if os.path.exists(path) and is_fresh(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("method") != "PISR-NSGA-II" or row.get("pde") != label:
                    continue
                try:
                    run, err = int(row["run"]), float(row["solution_mse"])
                except (KeyError, ValueError):
                    continue
                if err >= 0 and err == err:
                    out.append((run, err))
    out.sort(key=lambda t: t[1])
    _SEED_CACHE[label] = out
    return out


def representative_run(label: str):
    """Semilla mediana de `label` (la de rango ceil(n/2) por error; con 30
    semillas, la 15.a) o None si no hay corridas multi-semilla. Es la que el
    reporte usa para mostrar UNA formula: el caso tipico, ni el mejor ni uno
    al azar."""
    errs = seed_errors(label)
    if not errs or not os.path.isdir(os.path.join(RESULTS_DIR, "run_0")):
        return None
    return errs[(len(errs) - 1) // 2][0]


def seed_dirs():
    """Directorios results/run_<k> vigentes (multi-semilla), ordenados por k."""
    import glob, re
    ds = [d for d in glob.glob(os.path.join(RESULTS_DIR, "run_*")) if re.search(r"run_\d+$", d)]
    return sorted(ds, key=lambda d: int(d.rsplit("_", 1)[1]))


def _label_of(filename: str):
    hits = [b.label for b in BENCHMARKS if b.label in filename]
    return max(hits, key=len) if hits else None


def latest_result(filename: str):
    """Ruta de `filename` para mostrar en el reporte (o None).

    Con corridas multi-semilla (results/run_<k>/) devuelve el archivo de la
    semilla MEDIANA del problema (representative_run). Si ahi no existe (p. ej.
    los derivados que escribe el propio reporte en results/, como
    expr_<EDP>_simplified.tex) o no hay multi-semilla, el mas reciente entre
    results/ y results/run_*/."""
    import glob
    label = _label_of(filename)
    rep = representative_run(label) if label else None
    if rep is not None:
        p = os.path.join(RESULTS_DIR, f"run_{rep}", filename)
        if os.path.exists(p) and is_fresh(p):
            return p
    cands = [os.path.join(RESULTS_DIR, filename)] + glob.glob(os.path.join(RESULTS_DIR, "run_*", filename))
    # Solo resultados posteriores a la redefinicion de los problemas: un archivo
    # viejo de results/run_* resolvio OTRA ecuacion.
    cands = [c for c in cands if os.path.exists(c) and is_fresh(c)]
    return max(cands, key=os.path.getmtime) if cands else None


# Datos de borde y forma del ansatz de cada problema (misma construccion que
# apply_boundary_ansatz en src/pi_solver.cpp). Expresiones SymPy en x, y.
#   "dirichlet": 1D  U = g(0) + x (g(1)-g(0)) + x (x-1) N
#                2D  U = Coons[g] + x(1-x) y(1-y) N
#   "ivp"      : 1D  U = u0 + u0' x + x^2 N      (g = "u0,du0")
# Problemas sin entrada (no estacionarios, Navier-Stokes) no usan ansatz.
BOUNDARY = {
    "Laplace_1D": ("dirichlet", "x"),
    "Poisson_1D": ("dirichlet", "sin(pi*x)"),
    "HarmonicOscillator_1D": ("dirichlet", "exp(-x**2/2)"),
    "Airy_1D": ("dirichlet", "airyai(x)"),
    "Airy_2D": ("dirichlet", "airyai(2**(-Rational(1,3))*(x+y))"),
    "Fisher_1D": ("dirichlet", "1/(1+exp(-x))"),
    "Fisher_2D": ("dirichlet", "1/(1+exp(-(x+y)))"),
    "Duffing_1D": ("dirichlet", "1/cosh(x)"),
    "Duffing_2D": ("dirichlet", "1/cosh(x+y)"),
    "Thomas-Fermi_1D": ("dirichlet", "1-Rational(576,1000)*x"),
    "Thomas-Fermi_2D": ("dirichlet", "1-Rational(576,1000)*sqrt(x**2+y**2)"),
    "Schrodinger_1D": ("dirichlet", "exp(2*I*x)"),
    "Schrodinger_2D": ("dirichlet", "exp(2*I*(x+y))"),
    "Lane-Emden_1D": ("ivp", "1,0"),
    "Troesch_1D": ("dirichlet", "sinh(3*x)/sinh(3)"),
    "Ginzburg-Landau_1D": ("dirichlet", "tanh(x/sqrt(2))"),
    "Painleve-I_1D": ("dirichlet", "x**2/2"),
}



# Notas de fidelidad fisica (se imprimen bajo la tabla de la suite). Distinguen
# los modelos fisicos de las construcciones matematicas con nombre fisico.
PHYSICS_NOTES = [
    ("Modelos físicos",
     r"Oscilador armónico (estado base, $E=\tfrac12$ en unidades $\hbar=m=\omega=1$); Schr\"odinger "
     r"(partícula libre, estacionaria y dependiente del tiempo); Lane--Emden $n=3$ (politropo degenerado "
     r"relativista); Thomas--Fermi 1D (potencial atómico apantallado; $u(1)=0.424$ es el valor tabulado de la "
     r"solución semi-infinita, así que el problema truncado tiene la misma solución en $[0,1]$); Troesch "
     r"(confinamiento de plasma); Ginzburg--Landau (pared de dominio)."),
    ("Construcciones matemáticas con nombre físico",
     r"\emph{Tipo Duffing (contorno)}: el oscilador de Duffing es una EDO en el tiempo con amortiguamiento y "
     r"forzamiento periódico; aquí la no linealidad cúbica se usa en un problema de contorno espacial, y su "
     r"versión 2D no tiene contraparte física. \emph{Fisher--KPP (estacionaria)}: la física de Fisher--KPP es "
     r"el frente viajero de $u_t=u_{xx}+u(1-u)$; aquí solo se resuelve la ecuación estacionaria. \emph{Tipo "
     r"Thomas--Fermi (2D)}: Thomas--Fermi es una ecuación radial en 3D; la versión cartesiana 2D es un "
     r"problema de prueba."),
    ("Equivalencias y advertencias",
     r"Painlev\'e I en forma estándar es $w''=6w^2+z$; $u''=u^2+x$ es la misma ecuación con "
     r"$u=6^{3/5}w$, $x=6^{1/5}z$. En ambas soluciones de Navier--Stokes el término no lineal degenera "
     r"(Taylor--Green: $J(\psi,\omega)=0$; Kovasznay: $J(\psi,\omega)=\lambda\omega$ es lineal), así que son "
     r"soluciones exactas de Navier--Stokes que no ponen a prueba la no linealidad."),
]
