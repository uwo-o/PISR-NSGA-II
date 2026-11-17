// =============================================================================
// custom_runs/thesis_eq.cpp -- Corrida ad-hoc 2026-10-20, NO parte de la
// suite de benchmarks (no se agrega a main.cpp/run_once).
//
// Ecuacion de la tesis de magister de una amiga del usuario:
//     -u'' + 3(f'/f)u' + a*u = u^3
//
// Datos faltantes, acordados explicitamente con el usuario ante la falta de
// info real del problema original:
//   - f(x): no especificada. Se usa el placeholder f(x)=cosh(x) (f'/f=tanh(x),
//     suave, nunca se anula) -- ver pde_residual_ad, case PDE::CUSTOM_THESIS.
//   - Dominio: x in [0,1], el estandar del proyecto.
//   - a: parametro libre -> se fija por CLI para esta corrida exploratoria.
//   - Condiciones de frontera: NINGUNA real. u=0 satisface la ecuacion
//     exactamente para cualquier f,a, asi que se fija UN solo anclaje u(0)=1
//     (BCKind::ANCHOR_LEFT), impuesto por el ansatz U = 1 + x N(x).
//     El arbol del campeon es N; la solucion es U (ver eval_solution).
//
// Uso: ./thesis_eq [a] [pop] [gen]
// =============================================================================
#include "pde_problems.hpp"
#include "pi_solver.hpp"
#include <iostream>
#include <cstdlib>

int last_special_choice = -1; // ver extern en tree_node.cpp

int main(int argc, char** argv) {
    double a = (argc > 1) ? std::atof(argv[1]) : 1.0;
    int pop  = (argc > 2) ? std::atoi(argv[2]) : 500;
    int gen  = (argc > 3) ? std::atoi(argv[3]) : 1000;

    PDEProblem prob = make_custom_thesis_eq(a);

    std::cout << "=== Corrida ad-hoc: -u'' + 3(f'/f)u' + a*u = u^3 ===\n";
    std::cout << "    f(x)=cosh(x) (placeholder), a=" << a
              << ", pop=" << pop << ", gen=" << gen << "\n";
    std::cout << "    dominio x in [0,1], unico anclaje u(0)=1\n\n";

    PISolver solver(prob, 42);
    solver.run(pop, gen);

    if (!solver.has_champion()) {
        std::cout << "\n[!] No se encontro ningun campeon factible.\n";
        return 1;
    }

    const auto& champ = solver.champion();
    std::cout << "\n=== Campeon encontrado ===\n";
    std::cout << "mse_domain=" << champ.mse_domain
              << "  mse_boundary=" << champ.mse_boundary
              << "  tree_size=" << champ.tree_size << "\n";
    std::cout << (solution_uses_ansatz(prob, champ.raw) ? "N(x) [u = 1 + x N(x)] = " : "u(x) = ");
    champ.tree->print(std::cout);
    std::cout << "\n\nEvaluacion en una grilla (x, u(x)):\n";
    for (int i = 0; i <= 10; ++i) {
        double x = i / 10.0;
        std::cout << "  x=" << x << "  u=" << std::real(eval_solution(prob, champ.tree.get(), x, 0.0, 0.0, champ.raw)) << "\n";
    }
    return 0;
}
