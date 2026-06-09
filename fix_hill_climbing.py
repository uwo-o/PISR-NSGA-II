import re

with open("src/pi_solver.cpp", "r") as f:
    content = f.read()

# 1. Replace solve_qr_internal and hill_climb_constants
qr_start = content.find("// ─── Solucionador de Mínimos Cuadrados vía Householder QR ──────────────────────")
if qr_start == -1:
    print("Could not find solve_qr_internal")
    exit(1)

hill_start = content.find("void PISolver::hill_climb_constants")
if hill_start == -1:
    print("Could not find hill_climb_constants")
    exit(1)

end_marker = "void PISolver::update_hall_of_fame() {"
hill_end = content.find(end_marker)

new_hill_climb = """// ─── Optimización Local Pura (Hill-Climbing) ───────────────────────────────────
void PISolver::hill_climb_constants(PIIndividual& ind, int iterations) {
    if (!ind.tree) return;
    std::vector<Complex*> ercs;
    ind.tree->collect_ercs(ercs);
    if (ercs.empty()) return;

    double best_err = ind.mse_domain + ind.mse_boundary;
    std::vector<Complex> best_vals(ercs.size());
    for (size_t i = 0; i < ercs.size(); ++i) best_vals[i] = *ercs[i];

    std::normal_distribution<double> dist(0.0, Config::ERC_SIGMA * 0.2);

    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < ercs.size(); ++i) {
            *ercs[i] = best_vals[i] + dist(gen_);
        }
        
        ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
        double cur_err = ind.mse_domain + ind.mse_boundary;

        if (std::isfinite(cur_err) && cur_err < best_err) {
            best_err = cur_err;
            for (size_t i = 0; i < ercs.size(); ++i) best_vals[i] = *ercs[i];
        } else {
            for (size_t i = 0; i < ercs.size(); ++i) *ercs[i] = best_vals[i];
        }
    }
    
    for (size_t i = 0; i < ercs.size(); ++i) *ercs[i] = best_vals[i];
    ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
}

"""

content = content[:qr_start] + new_hill_climb + content[hill_end:]

# 2. Add Hill Climbing call to the top 25% in the main loop
run_marker = "population_ = nsga2_select_next(std::move(combined), pop_size);"
run_replace = """population_ = nsga2_select_next(std::move(combined), pop_size);

        // Hill Climbing robusto en el top 25% del Frente de Pareto
        int n_top = pop_size * 0.25;
        #pragma omp parallel for
        for (int i = 0; i < n_top; ++i) {
            hill_climb_constants(population_[i], 50);
        }"""

if run_marker in content:
    content = content.replace(run_marker, run_replace)
else:
    print("Could not find the nsga2_select_next line")

# 3. Clean up the hill climbing calls during population initialization if we are applying it in the loop instead
# (Actually, leaving it in init for seeds is good, but we changed the signature to current_gen_)
content = content.replace("hill_climb_constants(ind, 100); // Optimizar coeficientes de la semilla", "")
content = content.replace("ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_); // Re-evaluar", "")
content = content.replace("hill_climb_constants(special_ind, 50);", "")

with open("src/pi_solver.cpp", "w") as f:
    f.write(content)

print("pi_solver.cpp updated successfully via script")
