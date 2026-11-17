#ifndef NUMERICAL_SOLVER_HPP
#define NUMERICAL_SOLVER_HPP

#include "common.hpp"
#include "pde_problems.hpp"
#include <string>
#include <vector>

// Solver de referencia. La ecuacion sale SIEMPRE de PDEProblem::pde_residual_ad
// (la misma que usa la busqueda) y los datos de borde de bc()/ivp_*, asi que la
// referencia resuelve exactamente el problema que se le plantea al algoritmo.
// Solo se usa para EVALUAR resultados (y para el oraculo), nunca dentro de la
// busqueda.
namespace NumericalSolver {
    struct Report {
        bool ok = false;
        std::string method;
        int n = 0;                    // nodos por eje de la grilla guardada
        double newton_residual = 0.0; // max |R| (discreto) al converger
        double richardson_diff = 0.0; // max |u_h - u_{h/2}|: estimacion del error de discretizacion
        std::string note;
    };

    // Calcula la referencia de alta precision y la guarda en
    // prob.numerical_truth / prob.truth_n. Solo problemas estacionarios.
    Report reference(PDEProblem& prob);

    // Solucion a una resolucion dada sobre la grilla uniforme de [0,1]^dim
    // (linea de base "RK4/FDM" de las tablas). Vacio si no aplica.
    std::vector<Complex> solve(const PDEProblem& prob, int resolution);
}

#endif
