#pragma once
// =============================================================================
// oracle.hpp — Prueba de oraculo de los benchmarks.
//
// Para cada problema verifica, ANTES de cualquier experimento, que el
// planteamiento y la maquinaria son coherentes:
//   1. la referencia satisface la EDP (residuo por diferencias finitas sobre
//      prob.reference(), independiente del motor AD y de los arboles);
//   2. la referencia satisface los datos de borde / iniciales;
//   3. (forma cerrada, estacionario, orden 2) el solver numerico, partiendo
//      solo de la ecuacion y los datos de borde, reproduce la referencia:
//      detecta problemas mal planteados (p. ej. resonancia) y errores del solver;
//   4. (forma cerrada) el arbol de la solucion conocida, pasado por
//      PIIndividual::evaluate, es factible, con residuo ~0 y frontera ~0,
//      con y sin ansatz;
//   5. el orden de derivacion detectado por los priors coincide con el
//      declarado, y se informa que priors detectados se cumplen en la solucion.
// =============================================================================
#include "pde_problems.hpp"
#include "tree_node.hpp"
#include <string>
#include <vector>

struct OracleCheck {
    std::string name;
    bool pass = true;
    bool informative = false; // true: no cuenta para aprobar/reprobar
    std::string detail;
};

struct OracleReport {
    std::string problem;
    bool pass = true;
    std::vector<OracleCheck> checks;
};

// Arbol de la solucion en forma cerrada del benchmark (nullptr si no tiene,
// p. ej. Airy, cuya solucion Ai no es elemental, o los de referencia numerica).
NodePtr exact_solution_tree(const PDEProblem& prob);

// `prob` debe traer la referencia numerica ya calculada si is_numerical.
OracleReport run_oracle(const PDEProblem& prob);

void print_oracle_report(const OracleReport& r);
