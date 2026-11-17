#pragma once
// =============================================================================
// pde_problems.hpp  —  Definición de Laplace, Poisson y Helmholtz
//   Dominio Ω = [0,1]², condiciones de Dirichlet.
// =============================================================================

#include "common.hpp"
#include "dimensions.hpp"
#include <vector>
#include <random>

class Node;

// ─── Análisis a priori de la física (Physics-Guided Search) ───────────────
// Declarado antes de PDEProblem para que este pueda cachear su propio resultado
// (ver campo `priors` más abajo) y así llegar hasta PIIndividual::evaluate sin
// tener que agregar un parámetro nuevo a esa función en todos sus call sites.
// PRINCIPIO: los priors pueden SESGAR la busqueda (inicializacion, eleccion de
// esqueletos), nunca EXCLUIR candidatos. Que el operador tenga una propiedad
// (separabilidad, simetria) no implica que la solucion la tenga: depende de la
// frontera, del forzamiento y, en problemas no lineales, de posibles rupturas
// de simetria. Por eso ningun prior se usa como restriccion de factibilidad.
struct PDEPriors {
    bool autonomous_x = false;
    bool autonomous_y = false;
    bool autonomous_t = false;    // el residuo no depende explícitamente de t
    bool pole_at_origin = false;  // solo sesgo de esqueletos (random_tree_special); la formulacion del borde la declara el problema
    bool even_parity_x = false;

    // Simetría especular x <-> 1-x (resp. y <-> 1-y). Se marca si el operador
    // es invariante bajo la reflexion Y los datos de borde (evaluados SOLO en
    // la frontera, nunca en el interior) tambien lo son. Uso: una fraccion de
    // la poblacion inicial arranca simetrica. No es restriccion.
    bool mirror_symmetric_x = false;
    bool mirror_symmetric_y = false;

    // Priors de Nivel Superior
    bool scale_invariant = false; // Invarianza de escala (Auto-similaridad)
    int  max_deriv_order = 0;     // ADN Diferencial (Orden máximo de derivación)

    // Separabilidad del OPERADOR (sondeada con funciones genericas, nunca con
    // la solucion): u(x,y)=f(x)+g(y) o u(x,y)=f(x)g(y). Uso: una fraccion de la
    // poblacion inicial arranca con esa forma. No es restriccion: la solucion
    // puede no ser separable aunque el operador si lo sea.
    bool additive_separable = false;
    bool multiplicative_separable = false;

    // Separabilidad triple (2D dependiente del tiempo): u(x,y,t)=f(x)g(y)h(t).
    // Mismo sondeo generico sobre el OPERADOR (no la solucion) que la
    // separabilidad de 2 variables, extendido a probar las 3 derivadas
    // cruzadas (xy, xt, yt). Se activa para CUALQUIER EDP dependiente del
    // tiempo que pase el sondeo, sin excluir ningun tipo por nombre — asi no
    // es una solucion "hecha a medida" de un solo problema del benchmark.
    bool triple_separable = false;

    // Suma de un termino puro + un termino producto: u(x,y)=f(v)+g(x)h(y),
    // con v in {x,y}. Generaliza additive/multiplicative_separable (expansion
    // en modos/autofunciones, ej. Fourier) — cubre estructuras como
    // y - exp(lambda*x)*sin(2*pi*y)/(2*pi*Re) (Navier-Stokes) que no son ni
    // puramente aditivas ni puramente multiplicativas. Sondeado sobre el
    // OPERADOR con una funcion generica (ver probe_priors), nunca sobre la
    // solucion real. No excluye ningun tipo de EDP por nombre.
    bool modal_sum_separable = false;

    // El operador trata x e y de forma intercambiable (isotropo/simetrico),
    // sondeado evaluando el residuo con una funcion de sondeo generica de una
    // sola variable en un punto y su version con roles x<->y intercambiados
    // en el punto reflejado — si coinciden, el operador es consistente con
    // soluciones tipo "cresta"/onda plana f(a*x+b*y) para cualquier angulo,
    // no solo un eje. Ver skeleton_rotated en tree_node.cpp.
    bool rotation_invariant = false;

    // Operador lineal homogeneo (R(a u + b v) = a R(u) + b R(v), R(0) = 0) y
    // sin dependencia explicita de las variables: las exponenciales
    // e^{lambda x + mu y + nu t} (lambda complejo) son autofunciones del
    // operador. Sondeado con jets genericos, nunca con la solucion. Uso: una
    // fraccion de la poblacion inicial arranca como suma de 1-2 modos
    // exponenciales con exponentes al azar. No es restriccion.
    bool exponential_modes = false;

    // Termino de amortiguamiento lineal en la primera derivada (du/dx o
    // du/dy) mas alla de lo que ya aporta el Laplaciano — la firma
    // caracteristica de un oscilador amortiguado (u''+2*zeta*omega*u'+
    // omega^2*u=0). Sondeado perturbando SOLO esa derivada (dejando el resto
    // fijo) y viendo si el residuo reacciona — ver skeleton_damped_oscillator
    // en tree_node.cpp.
    bool damped_oscillator = false;
};

// ─── Problema PDE ─────────────────────────────────────────────────────────────
struct PDEProblem {
    PDE    type;
    int    dim = 2;    // 1 o 2 dimensiones espaciales
    double k2 = 0.0;   // parametro escalar del operador (k² en Helmholtz/Schrodinger, nu en Navier-Stokes, a en la ecuacion ad-hoc)
    bool   is_numerical = false; // la referencia no tiene forma cerrada: se calcula con NumericalSolver::reference
    bool   is_unsteady = false;  // depende del tiempo (dominio en (x[,y],t) con t in [0,1])
    std::vector<Complex> numerical_truth; // Malla de la referencia numerica (ver NumericalSolver::reference)
    int    truth_n = 0;                   // nodos por eje de numerical_truth
    PDEPriors priors; // Cacheado por PISolver al construirse (probe_priors). Solo sesgo, nunca restriccion.

    // ── Definicion del problema bien planteado (datos, no heuristicas) ──
    BCKind bc_kind = BCKind::DIRICHLET;
    Complex ivp_u0 = 0.0, ivp_du0 = 0.0;  // IVP_LEFT: u(0), u'(0). ANCHOR_LEFT: solo ivp_u0.
    // Campo incognita complejo (Schrodinger). Si es false, el campo es real:
    // lo complejo del arbol es solo representacion, y residuo, frontera y
    // reporte se calculan sobre Re(U) (ver project_real en common.hpp).
    bool complex_field = false;
    // Orden maximo de las derivadas espaciales del operador. Con orden 4
    // (Navier-Stokes en funcion de corriente) un problema de Dirichlet bien
    // planteado necesita u Y du/dn en la frontera; ademas el ansatz (que solo
    // esta implementado hasta segundo orden) no aplica.
    int spatial_order = 2;

    // Dimensiones para coherencia física
    Dimension dim_u = Units::None;
    Dimension dim_x = Units::Length;
    Dimension dim_y = Units::Length;
    Dimension dim_t = Units::Time;

    // Grilla FIJA y densa de validacion en el interior del dominio. Distinta de
    // los puntos de entrenamiento (que se muestrean al azar en PISolver).
    std::vector<Point> validation_points() const;
    // Puntos de frontera deterministas y densos (validacion) / un punto de
    // frontera al azar (entrenamiento). Ambos describen el MISMO conjunto de
    // datos de borde del problema: frontera espacial (para todo t si es no
    // estacionario) y, si es no estacionario, condicion inicial en t=0.
    std::vector<Point> boundary_points(int n) const;
    Point sample_boundary_point(std::mt19937& rng) const;

    // Desajuste cuadratico de los datos de borde en el punto p, dado el jet
    // (ya proyectado) del candidato en p. Unica fuente de verdad de "que
    // significa cumplir la frontera" para este problema (valor; derivada
    // normal si spatial_order>2; u'(0) si IVP_LEFT).
    double boundary_mismatch_sq(const AD& jet, const Point& p) const;

    // Solución exacta conocida u*(x,y,t) (solo tiene sentido si !is_numerical,
    // o como dato de borde via bc()).
    Complex exact(double x, double y, double t = 0.0) const;
    // Referencia numerica interpolada (cubica) de numerical_truth.
    Complex numerical_exact(double x, double y, double t = 0.0) const;
    // Referencia para EVALUAR resultados (nunca para buscar): exacta si
    // existe, numerica si no.
    Complex reference(double x, double y, double t = 0.0) const;

    // Término fuente f(x,y,t) del lado derecho
    Complex source(double x, double y, double t = 0.0) const;

    // Dato de borde: valor de u en la frontera (o en t=0). Solo debe
    // evaluarse sobre la frontera/condicion inicial.
    Complex bc(double x, double y, double t = 0.0) const;

    // Residuo del operador sobre un jet. Unica fuente de verdad de la
    // ecuacion: la usan la busqueda, la validacion, los priors y el solver de
    // referencia. Para spatial_order>2 el jet debe venir de ad_eval_ext_t.
    Complex pde_residual_ad(const AD& ad, double x, double y, double t = 0.0) const;

    // Nombre legible
    std::string name() const;
};

PDEPriors probe_priors(const PDEProblem& prob);

// ─── Fabricación de problemas ─────────────────────────────────────────────────
PDEProblem make_laplace(int dim = 2);
PDEProblem make_poisson(int dim = 2);
PDEProblem make_helmholtz(int dim = 2, double k = 1.0);
PDEProblem make_schrodinger(int dim = 2);
PDEProblem make_schrodinger_unsteady(int dim = 2);
PDEProblem make_nonlinear_poisson();
PDEProblem make_liouville();
PDEProblem make_sine_gordon();
PDEProblem make_airy(int dim);
PDEProblem make_harmonic_oscillator(int dim);
PDEProblem make_navier_stokes();
PDEProblem make_navier_stokes_unsteady();
PDEProblem make_fisher(int dim);
PDEProblem make_duffing(int dim);
PDEProblem make_thomas_fermi(int dim);

// Corrida ad-hoc, NO parte de la suite de benchmarks (ver
// custom_runs/thesis_eq.cpp): -u'' + 3f'/f u' + a*u = u^3, con f(x)=cosh(x)
// como placeholder (no especificado por el problema original) y a libre.
PDEProblem make_custom_thesis_eq(double a);

// La suite de benchmarks (unica fuente: la usan main.cpp y los tests).
std::vector<PDEProblem> benchmark_suite();
// Etiqueta "Nombre_1D" / "Nombre_2D".
std::string problem_label(const PDEProblem& p);

// Funcion de Airy Ai(z) por su serie de Maclaurin (converge para todo z; con
// |z|<=2 bastan ~20 terminos para precision de maquina). Referencia exacta de
// los benchmarks de Airy.
double airy_ai(double z);
PDEProblem make_bratu();
PDEProblem make_allen_cahn();
PDEProblem make_lane_emden();
PDEProblem make_troesch();
PDEProblem make_ginzburg_landau();
PDEProblem make_painleve1();
