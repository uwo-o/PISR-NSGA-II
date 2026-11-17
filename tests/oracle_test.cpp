// =============================================================================
// tests/oracle_test.cpp — prueba de oraculo de toda la suite (ver oracle.hpp).
// Uso: ./oracle_test [Nombre_1D ...]   (sin argumentos: toda la suite)
// Sale con codigo != 0 si algun problema falla.
// =============================================================================
#include "oracle.hpp"
#include "numerical_solver.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

int last_special_choice = -1;

int main(int argc, char** argv) {
    std::vector<std::string> only(argv + 1, argv + argc);
    int fails = 0, total = 0;
    for (PDEProblem p : benchmark_suite()) {
        std::string lbl = problem_label(p);
        if (!only.empty() && std::find(only.begin(), only.end(), lbl) == only.end()) continue;
        bool ref_ok = true;
        if (p.is_numerical) {
            auto r = NumericalSolver::reference(p);
            ref_ok = r.ok;
            std::cout << "[Referencia] " << lbl << ": " << r.method << "  n=" << r.n
                      << "  residuo discreto=" << r.newton_residual << "  dif. Richardson=" << r.richardson_diff
                      << (r.ok ? "" : "  [NO CONVERGIO -> FALLA]") << "\n";
        }
        OracleReport rep = run_oracle(p);
        print_oracle_report(rep);
        ++total;
        if (!rep.pass || !ref_ok) ++fails;
    }
    std::cout << "\nORACULO: " << (total - fails) << "/" << total << " problemas aprobados\n";
    return fails ? 1 : 0;
}
