#pragma once
// =============================================================================
// common.hpp  —  Tipos de datos y constantes globales
// =============================================================================

#include <vector>
#include <string>
#include <limits>
#include <complex>

using Complex = std::complex<double>;
const Complex I_COMPLEX(0.0, 1.0);
const double PI_VAL = 3.14159265358979323846;
const double E_VAL  = 2.71828182845904523536;

// ─── Tipos de PDE soportados ──────────────────────────────────────────────────
enum class PDE { 
    LAPLACE, POISSON, HELMHOLTZ, SCHRODINGER, 
    NONLINEAR_POISSON, LIOUVILLE, SINE_GORDON,
    AIRY, HARMONIC_OSCILLATOR, GROSS_PITAEVSKII,
    NAVIER_STOKES, NAVIER_STOKES_UNSTEADY,
    SCHRODINGER_UNSTEADY,
    FISHER, DUFFING, THOMAS_FERMI,
    BRATU, ALLEN_CAHN, LANE_EMDEN,
    TROESCH, GINZBURG_LANDAU, PAINLEVE1,
    CUSTOM_THESIS // Corrida ad-hoc (-u''+3f'/f u'+au=u^3) -- NO es parte de
                  // la suite de benchmarks, ver custom_runs/thesis_eq.cpp.
};

// ─── Tipo de condicion de borde (parte de la DEFINICION del problema) ─────────
// Cada benchmark es un problema bien planteado: ecuacion + dominio + datos de
// borde/iniciales. El tipo de dato no se detecta ni se infiere de la ecuacion:
// se declara en la fabrica del problema, igual que los valores.
//   DIRICHLET   : u dado en toda la frontera espacial (y, si es no estacionario,
//                 ademas la condicion inicial u(x,0)). Para operadores de orden
//                 espacial 4 se da tambien du/dn (ver PDEProblem::spatial_order).
//   IVP_LEFT    : 1D estacionario, u(0) y u'(0) dados (problema de valor inicial).
//   ANCHOR_LEFT : 1D, solo u(0) dado. Problema INDETERMINADO a proposito; solo
//                 lo usa la corrida ad-hoc custom_runs/thesis_eq.cpp, nunca un
//                 benchmark.
enum class BCKind { DIRICHLET, IVP_LEFT, ANCHOR_LEFT };

inline std::string pde_name(PDE t) {
    switch (t) {
        case PDE::LAPLACE:     return "Laplace";
        case PDE::POISSON:     return "Poisson";
        case PDE::HELMHOLTZ:   return "Helmholtz";
        case PDE::SCHRODINGER:       return "Schrodinger";
        case PDE::SCHRODINGER_UNSTEADY: return "Schrodinger-Unsteady";
        case PDE::NONLINEAR_POISSON: return "NonlinearPoisson";
        case PDE::LIOUVILLE:         return "Liouville";
        case PDE::SINE_GORDON:       return "Sine-Gordon";
        case PDE::AIRY:              return "Airy";
        case PDE::HARMONIC_OSCILLATOR: return "HarmonicOscillator";
        case PDE::GROSS_PITAEVSKII:  return "Gross-Pitaevskii";
        case PDE::NAVIER_STOKES:     return "Navier-Stokes";
        case PDE::NAVIER_STOKES_UNSTEADY: return "Navier-Stokes-Unsteady";
        case PDE::FISHER:            return "Fisher";
        case PDE::DUFFING:           return "Duffing";
        case PDE::THOMAS_FERMI:      return "Thomas-Fermi";
        case PDE::BRATU:             return "Bratu";
        case PDE::ALLEN_CAHN:        return "Allen-Cahn";
        case PDE::LANE_EMDEN:        return "Lane-Emden";
        case PDE::TROESCH:           return "Troesch";
        case PDE::GINZBURG_LANDAU:   return "Ginzburg-Landau";
        case PDE::PAINLEVE1:         return "Painleve-I";
        case PDE::CUSTOM_THESIS:     return "CustomThesisEq";
        default: return "Unknown";
    }
}
// ─── Tipos de Nodo del árbol ──────────────────────────────────────────────────
enum class NodeType {
    ADD, SUB, MUL, DIV, POW,
    SIN, COS, SINH, COSH, EXP, SQR, LOG, TANH,
    LEGENDRE, HERMITE, CHEBYSHEV, LAGUERRE,
    GAUSSIAN,
    VAR_X, VAR_Y, VAR_T, VAR_Z, VAR_N,
    ERC, CONST_I, CONST_PI, CONST_E,
    CONST_G, CONST_C, CONST_HBAR, CONST_KB, CONST_EPS0,
    SERIES,
    UNKNOWN
    };
// ─── Estructura Dual (Valor + Derivadas) para AD ──────────────────────────────
// Campos de 2do/3er/4to orden agregados para el residuo EXACTO de Navier-Stokes
// (biharmonico + gradiente de vorticidad, ver PDEProblem::compute_residual),
// que antes se aproximaba con diferencias finitas. dxy/dxt/dyt no los usa
// ningun residuo directamente, pero son necesarios como andamiaje: la regla de
// Leibniz para MUL y la de Faa di Bruno para composiciones (SIN, EXP, ...)
// requieren esos mixtos de 2do orden para propagar correctamente dxxy, dxyy,
// dxxyy, dxxt, dyyt de orden superior (ver apply_composition/apply_mul_ad en
// tree_node.cpp). Todos por defecto en 0, igual que los campos existentes —
// esto es estrictamente aditivo: ninguna EDP fuera de Navier-Stokes/-Unsteady
// llega a leer estos campos, asi que su comportamiento no cambia en nada.
struct AD {
    Complex v;   // valor
    Complex dx, dy, dt;
    Complex dxx, dyy, dtt;
    Complex dxy, dxt, dyt;             // 2do orden mixto (andamiaje)
    Complex dxxx, dxxy, dxyy, dyyy;    // 3er orden espacial puro/mixto
    Complex dxxt, dyyt;                // 3er orden espacio(2)-tiempo(1)
    Complex dxxxx, dxxyy, dyyyy;       // 4to orden espacial puro/mixto (biharmonico)

    AD(Complex val = 0.0) : v(val), dx(0), dy(0), dt(0), dxx(0), dyy(0), dtt(0),
        dxy(0), dxt(0), dyt(0), dxxx(0), dxxy(0), dxyy(0), dyyy(0),
        dxxt(0), dyyt(0), dxxxx(0), dxxyy(0), dyyyy(0) {}
};

// Parte real de todo el jet. Las derivadas respecto de x, y, t (variables
// reales) conmutan con Re, asi que Re(U)_xx == Re(U_xx), etc. Se aplica ANTES
// del operador (posiblemente no lineal) en los problemas de campo real: lo
// complejo queda solo como recurso de representacion del arbol (CONST_I,
// rotaciones de fase), y lo que se optimiza es exactamente lo que se reporta,
// Re(U). Sin esto, un arbol complejo con residuo bajo no garantizaba que Re(U)
// resolviera la ecuacion, porque Re(U)^3 != Re(U^3).
inline AD project_real(const AD& a) {
    AD r;
    r.v = a.v.real();
    r.dx = a.dx.real();     r.dy = a.dy.real();     r.dt = a.dt.real();
    r.dxx = a.dxx.real();   r.dyy = a.dyy.real();   r.dtt = a.dtt.real();
    r.dxy = a.dxy.real();   r.dxt = a.dxt.real();   r.dyt = a.dyt.real();
    r.dxxx = a.dxxx.real(); r.dxxy = a.dxxy.real(); r.dxyy = a.dxyy.real(); r.dyyy = a.dyyy.real();
    r.dxxt = a.dxxt.real(); r.dyyt = a.dyyt.real();
    r.dxxxx = a.dxxxx.real(); r.dxxyy = a.dxxyy.real(); r.dyyyy = a.dyyyy.real();
    return r;
}

// ─── Punto en el dominio ──────────────────────────────────────────────────────
struct Point {
    double x, y, t;
};

// ─── Clase base para individuo (para NSGA-II genérico) ────────────────────────
struct Individual {
    virtual ~Individual() = default;
    double mse_domain   = std::numeric_limits<double>::max();
    double mse_boundary = std::numeric_limits<double>::max();
    int    rank         = 0;
    double crowding     = 0.0;
    int    tree_size    = 0;
    NodeType root_type  = NodeType::UNKNOWN;

    // Restricciones NSGA-II (Deb, 2002)
    bool   is_feasible = true;
    double constraint_violation = 0.0; // 0 = feasible, >0 = infeasible
};

// ─── Estadísticas de convergencia por generación ─────────────────────────────
struct ConvergenceStats {
    int    gen;
    double best_mse_domain;   // Mejor MSE de dominio del batch de entrenamiento de esta generacion (ruidoso: mini-batch cambia cada gen)
    double best_mse_boundary; // Idem, frontera
    double best_total_mse;    // best_mse_domain + best_mse_boundary (mismo batch, mismo ruido)
    double best_val_score;    // Puntaje de validacion del campeon: residuo + frontera en la grilla FIJA de validacion. Solo informacion de la EDP.
    double champion_ref_err;  // Error del campeon contra la referencia. SOLO REGISTRO para analisis: ningun camino del algoritmo lo lee.
};

// ─── Parámetros globales ──────────────────────────────────────────────────────
namespace Config {
    extern int    POP_SIZE;   
    extern int    MAX_GEN;   
    extern int    N_DOMAIN;  
    extern int    N_BOUNDARY;   
    extern double ERC_SIGMA;  
    extern int    MAX_TREE_DEPTH;
    
    extern double CROSSOVER_PROB;  
    extern double MUTATION_PROB;   
    extern int    TOURNAMENT_SIZE;    
    extern double STOP_THRESHOLD; 

    // Hybrid Committee RAR
    extern int    RAR_INTERVAL;      
    extern int    RAR_CANDIDATES;    
    extern double RAR_ADAPTIVE_RATIO;
    extern int    RAR_ELITE_COUNT;   
    extern double RAR_RANDOM_RATIO;  
    extern bool   GENERAL_MODE;

    // Si es false, se desactiva el ansatz de frontera exacta (U=L+B*N) — la
    // frontera vuelve a ser una penalizacion blanda real (compute_boundary_error)
    // en vez de forzarse a 0 por construccion, y el arbol resuelve la EDP
    // "a pelo". Distinto de GENERAL_MODE (que ademas apaga la frontera por
    // completo, ni siquiera como penalizacion blanda).
    extern bool   USE_ANSATZ;

    // Representacion hibrida (solo si el ansatz aplica): parte de la poblacion
    // representa U directamente, con la frontera medida, y compite con los
    // individuos de ansatz. --no-hybrid lo apaga (ablacion).
    extern bool   HYBRID_REPR;

    extern int    CORES;
    }