// =============================================================================
// main.cpp  —  Orquestador: PI-NSGA-II vs Koza-BNF
//   Prueba Laplace, Poisson y Helmholtz en Omega = [0,1]^2
//   Genera CSVs de frentes de Pareto para analisis comparativo.
// =============================================================================

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <numeric>

#include "common.hpp"
#include "pde_problems.hpp"
#include "koza_bnf.hpp"
#include "pi_solver.hpp"

namespace fs = std::filesystem;

// ─── Guardar frente de Pareto como CSV ───────────────────────────────────────
template<typename Ind>
void save_pareto_csv(const std::vector<Ind>& pop,
                     const std::string& path,
                     const std::string& method,
                     const std::string& pde_label)
{
    std::ofstream f(path);
    f << std::fixed << std::setprecision(8);
    f << "method,pde,mse_domain,mse_boundary,rank\n";
    for (auto& ind : pop) {
        if (ind.rank == 1)
            f << method << "," << pde_label << ","
              << ind.mse_domain << "," << ind.mse_boundary << ","
              << ind.rank << "\n";
    }
    std::cout << "  -> Guardado: " << path << "\n";
}

// ─── Estadísticas ─────────────────────────────────────────────────────────────
struct Stats {
    std::string method, pde;
    int    front_size  = 0;
    double best_domain = 1e18;
    double best_bnd    = 1e18;
    double mean_domain = 0.0;
    double mean_bnd    = 0.0;
    double runtime_s   = 0.0;
};

template<typename Ind>
Stats compute_stats(const std::vector<Ind>& pop,
                    const std::string& method,
                    const std::string& pde,
                    double rt)
{
    Stats s;
    s.method = method; s.pde = pde; s.runtime_s = rt;
    for (auto& ind : pop) {
        if (ind.rank == 1) {
            s.front_size++;
            s.best_domain = std::min(s.best_domain, ind.mse_domain);
            s.best_bnd    = std::min(s.best_bnd,    ind.mse_boundary);
            s.mean_domain += ind.mse_domain;
            s.mean_bnd    += ind.mse_boundary;
        }
    }
    if (s.front_size > 0) {
        s.mean_domain /= s.front_size;
        s.mean_bnd    /= s.front_size;
    }
    return s;
}

// ─── Validación con solución exacta ──────────────────────────────────────────
void validate_exact(const PDEProblem& prob) {
    auto dom = prob.domain_points(Config::N_DOMAIN);
    double mse = 0.0;
    for (auto& p : dom) {
        double u = prob.exact(p.x, p.y);
        double h   = 1e-5;
        auto fn = [&](double x, double y){ return prob.exact(x, y); };
        double lap = (fn(p.x+h,p.y) - 2*fn(p.x,p.y) + fn(p.x-h,p.y))/(h*h)
                   + (fn(p.x,p.y+h) - 2*fn(p.x,p.y) + fn(p.x,p.y-h))/(h*h);
        double res = lap + prob.k2 * u - prob.source(p.x, p.y);
        mse += res * res;
    }
    mse /= (double)dom.size();
    std::cout << "  [Validacion] Residuo PDE de u* (" << prob.name()
              << "): " << std::scientific << mse << " (esperado ~0)\n";
}

// ─── Guardar resumen comparativo ─────────────────────────────────────────────
void save_summary(const std::vector<Stats>& stats, const std::string& path) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(8);
    f << "method,pde,pareto_size,best_mse_domain,best_mse_boundary,"
         "mean_mse_domain,mean_mse_boundary,runtime_s\n";
    for (auto& s : stats) {
        f << s.method << "," << s.pde << "," << s.front_size << ","
          << s.best_domain << "," << s.best_bnd << ","
          << s.mean_domain << "," << s.mean_bnd << ","
          << s.runtime_s << "\n";
    }
    std::cout << "  -> Resumen: " << path << "\n";
}

// ─── Tabla en consola ─────────────────────────────────────────────────────────
void print_table(const std::string& lbl, const Stats& k, const Stats& p) {
    std::cout << "\n+------------------+-------------+-------------+----------+----------+\n";
    std::cout << "| " << std::left << std::setw(48) << ("  Ecuacion: " + lbl) << "|\n";
    std::cout << "+------------------+-------------+-------------+----------+----------+\n";
    std::cout << "| Metodo           | MSE Dom.    | MSE Bnd.    | Pareto   | Tiempo   |\n";
    std::cout << "+------------------+-------------+-------------+----------+----------+\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "| Koza (BNF)       | " << std::setw(11) << k.best_domain
              << " | " << std::setw(11) << k.best_bnd
              << " | " << std::setw(8)  << k.front_size
              << " | " << std::setw(7)  << k.runtime_s << "s |\n";
    std::cout << "| PI-NSGA-II       | " << std::setw(11) << p.best_domain
              << " | " << std::setw(11) << p.best_bnd
              << " | " << std::setw(8)  << p.front_size
              << " | " << std::setw(7)  << p.runtime_s << "s |\n";
    std::cout << "+------------------+-------------+-------------+----------+----------+\n";
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=============================================================\n";
    std::cout << "  PI-NSGA-II vs Koza BNF --- Benchmark de Ecuaciones PDE\n";
    std::cout << "  Laplace | Poisson | Helmholtz  en  Omega = [0,1]^2\n";
    std::cout << "  Pop=" << Config::POP_SIZE
              << "  Gen=" << Config::MAX_GEN
              << "  Pts_dom=" << Config::N_DOMAIN
              << "  Pts_bnd=" << Config::N_BOUNDARY << "\n";
    std::cout << "=============================================================\n\n";

    fs::create_directories("results");

    std::vector<PDEProblem> problems = {
        make_laplace(),
        make_poisson(),
        make_helmholtz(1.0)
    };

    // ── Validar soluciones exactas ──────────────────────────────────────────
    std::cout << "--- Validacion de soluciones exactas ---\n";
    for (auto& prob : problems) validate_exact(prob);
    std::cout << "\n";

    std::vector<Stats> all_stats;

    for (auto& prob : problems) {
        std::string lbl = prob.name();
        std::cout << "=====================================================\n";
        std::cout << "  " << lbl << "\n";
        std::cout << "=====================================================\n";

        // ── Koza ─────────────────────────────────────────────────────────────
        std::cout << "\n[1/2] Koza (BNF + Diferencias Finitas)\n";
        auto t0 = std::chrono::steady_clock::now();
        KozaSolver koza(prob, 12345u);
        auto koza_pop = koza.run(Config::POP_SIZE, Config::MAX_GEN);
        double koza_rt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        save_pareto_csv(koza_pop, "results/" + lbl + "_koza_pareto.csv",
                        "Koza", lbl);
        Stats ks = compute_stats(koza_pop, "Koza", lbl, koza_rt);

        // ── PI-NSGA-II ────────────────────────────────────────────────────────
        std::cout << "\n[2/2] PI-NSGA-II (AD Simbolico + ERCs)\n";
        t0 = std::chrono::steady_clock::now();
        PISolver pi(prob, 12345u);
        auto pi_pop = pi.run(Config::POP_SIZE, Config::MAX_GEN);
        double pi_rt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        save_pareto_csv(pi_pop, "results/" + lbl + "_pi_pareto.csv",
                        "PI-NSGA-II", lbl);
        Stats ps = compute_stats(pi_pop, "PI-NSGA-II", lbl, pi_rt);

        print_table(lbl, ks, ps);
        all_stats.push_back(ks);
        all_stats.push_back(ps);
        std::cout << "\n";
    }

    save_summary(all_stats, "results/comparison_summary.csv");

    std::cout << "\n=============================================================\n";
    std::cout << "  Listo. Archivos CSV en ./results/\n";
    std::cout << "  Para graficar:  python3 plot_pareto.py\n";
    std::cout << "=============================================================\n";
    return 0;
}
