// =============================================================================
// main.cpp  —  Orquestador: PISR-NSGA-II (Solo nuestro Algoritmo)
//   Modos:
//     ./build/pi_nsga2              → 1 corrida (comportamiento estándar)
//     ./build/pi_nsga2 --runs N     → N corridas independientes (análisis estadístico)
//
//   Genera CSVs de frentes de Pareto para análisis comparativo.
//   Ecuaciones: Laplace, Poisson, Helmholtz, Liouville, Sine-Gordon.
// =============================================================================

#include "pi_solver.hpp"
int last_special_choice = -1;
#include "numerical_solver.hpp"
#include "oracle.hpp"
#include <set>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <omp.h>

#include "common.hpp"
#include "pde_problems.hpp"
#include "pi_solver.hpp"

namespace fs = std::filesystem;

// ─── Guardar frente de Pareto como CSV ───────────────────────────────────────
template<typename Ind>
void save_pareto_csv(const std::vector<Ind>& pop,
                     const std::string& path,
                     const std::string& method,
                     const std::string& pde_label,
                     int dim)
{
    std::ofstream f(path);
    f << std::fixed << std::setprecision(10);
    f << "method,pde,dim,mse_domain,mse_boundary,tree_size,rank\n";
    for (auto& ind : pop) {
        f << method << "," << pde_label << "," << dim << ","
          << ind.mse_domain << "," << ind.mse_boundary << ","
          << ind.tree_size << "," << ind.rank << "\n";
    }
}

// ─── Guardar el archivo historico de poblacion (todas las generaciones) ──────
// A diferencia de save_pareto_csv (una sola generacion, la final), esto trae
// una fila por individuo por CADA generacion — ver PISolver::population_archive()
// y su comentario en pi_solver.hpp para la motivacion (el pulido de constantes
// colapsa la diversidad visual incluso en la poblacion final; generaciones
// anteriores, antes de converger del todo, tienen combinaciones (tree_size,
// mse_domain) que ya no sobreviven al final).
void save_population_archive_csv(const std::vector<PISolver::ArchiveEntry>& archive,
                                  const std::string& path,
                                  const std::string& method,
                                  const std::string& pde_label,
                                  int dim)
{
    std::ofstream f(path);
    f << std::fixed << std::setprecision(10);
    f << "method,pde,dim,gen,mse_domain,mse_boundary,tree_size,rank\n";
    for (auto& e : archive) {
        f << method << "," << pde_label << "," << dim << "," << e.gen << ","
          << e.mse_domain << "," << e.mse_boundary << ","
          << e.tree_size << "," << e.rank << "\n";
    }
}

// ─── Guardar Historial de Convergencia ────────────────────────────────────────
void save_convergence_csv(const std::vector<ConvergenceStats>& history,
                           const std::string& path)
{
    std::ofstream f(path);
    // best_val_score: puntaje de validacion (fisica) del campeon.
    // champion_ref_err: error contra la referencia, solo registro.
    f << "gen,best_mse_domain,best_mse_boundary,best_total_mse,best_val_score,champion_ref_err\n";
    for (auto& s : history) {
        f << s.gen << "," << s.best_mse_domain << ","
          << s.best_mse_boundary << "," << s.best_total_mse << ","
          << s.best_val_score << "," << s.champion_ref_err << "\n";
    }
}

// ─── Guardar Mejor Expresión en LaTeX ────────────────────────────────────────
// Recibe directamente al campeon del Hall of Fame (ya validado contra la
// grilla fija dentro de PISolver) en vez de re-escanear la poblacion final
// por mse de entrenamiento — ese escaneo podia elegir un individuo que
// simplemente sobreajustaba el mini-batch de la ultima generacion.
template<typename Ind>
void save_best_expression(const Ind& champion, const PDEProblem& prob, const std::string& path, int dim = 2) {
    if (!champion.tree) return;
    std::ofstream f(path);
    // Redondeo ANTES de simplificar (no al reves): tras el pulido por
    // gradiente, dos sub-arboles algebraicamente iguales (ej. "1.0000003*x"
    // y "0.9999997*x", ambos de una mutacion/cruzamiento que dejo ERCs casi-
    // pero-no-exactamente 1.0) imprimen igual ("x") por la tolerancia de
    // print_formal, pero simplify() los ve como distintos y no cancela
    // "1.0000003x - 0.9999997x" a 0. Redondeando primero (1e-3, cosmetico —
    // NO el 0.05 agresivo de antes que snapeaba constantes legitimas a 0)
    // ambos quedan en 1.0 exacto, y simplify() sí puede cancelar/colapsar.
    // Arbol exacto (sin redondear) para la simplificacion simbolica posterior.
    {
        std::string sym_path = path.substr(0, path.size() - 4) + ".sym";
        std::ofstream fs(sym_path);
        fs << "raw=" << (champion.raw ? 1 : 0) << "\n";
        fs << "uses_ansatz=" << (solution_uses_ansatz(prob, champion.raw) ? 1 : 0) << "\n";
        fs << "real_field=" << (prob.complex_field ? 0 : 1) << "\n";
        fs << "tree=";
        champion.tree->print_sympy(fs);
        fs << "\n";
    }
    auto final_tree = champion.tree->clone();
    final_tree->round_constants(1e-3);
    final_tree = final_tree->simplify(); // Limpieza simbólica

    // Mismo criterio que la busqueda (solution_uses_ansatz). Si el campo es
    // real pero el arbol usa la unidad imaginaria (representacion), la
    // solucion reportada es Re(.), exactamente lo que se optimizo.
    bool use_ansatz = solution_uses_ansatz(prob, champion.raw);
    auto emit = [&](const Node* expr) {
        bool wrap_re = !prob.complex_field && expr->uses_variable(NodeType::CONST_I);
        f << "$$ ";
        if (wrap_re) f << "\\operatorname{Re}\\left(";
        expr->print_formal(f, 0);
        if (wrap_re) f << "\\right)";
        f << " $$" << std::endl;
    };

    if (!use_ansatz) { emit(final_tree.get()); return; }

    // 1D: se imprime SIEMPRE la formula U(x) final fusionada y simplificada
    // (no N(x) sola), con la forma del ansatz que corresponde al tipo de dato
    // de borde del problema.
    if (dim == 1) {
        NodePtr U;
        if (prob.bc_kind == BCKind::IVP_LEFT) {
            // U = u0 + u0' x + x^2 N
            NodePtr lin = make_binary(NodeType::ADD, make_erc(prob.ivp_u0),
                                       make_binary(NodeType::MUL, make_erc(prob.ivp_du0), make_var('x')));
            NodePtr x2 = make_binary(NodeType::MUL, make_var('x'), make_var('x'));
            U = make_binary(NodeType::ADD, std::move(lin),
                            make_binary(NodeType::MUL, std::move(x2), std::move(final_tree)));
        } else if (prob.bc_kind == BCKind::ANCHOR_LEFT) {
            // U = u0 + x N
            U = make_binary(NodeType::ADD, make_erc(prob.ivp_u0),
                            make_binary(NodeType::MUL, make_var('x'), std::move(final_tree)));
        } else {
            // U = u0 + x(u1-u0) + x(x-1) N
            Complex u0 = prob.bc(0.0, 0.0, 0.0), u1 = prob.bc(1.0, 0.0, 0.0);
            NodePtr L = make_binary(NodeType::ADD, make_erc(u0),
                                     make_binary(NodeType::MUL, make_var('x'), make_erc(u1 - u0)));
            NodePtr B = make_binary(NodeType::MUL, make_var('x'),
                                     make_binary(NodeType::SUB, make_var('x'), make_erc(1.0)));
            U = make_binary(NodeType::ADD, std::move(L),
                             make_binary(NodeType::MUL, std::move(B), std::move(final_tree)));
        }
        U = U->simplify();
        U->round_constants(1e-3);
        U = U->simplify();
        emit(U.get());
        return;
    }
    // 2D: U(x,y) = L(x,y) + x(1-x)y(1-y) N(x,y), con L la interpolacion
    // transfinita de los datos de borde (sin forma cerrada generica); se
    // reporta N(x,y).
    emit(final_tree.get());
}

// ─── Guardar Rejilla de Evaluación (para plots 3D/1D) ─────────────────────────
template<typename Ind>
void save_best_grid(const Ind& champion, const PDEProblem& prob, const std::string& path) {
    if (!champion.tree) return;
    std::ofstream f(path);
    // u_approx sale de eval_solution: la MISMA cadena (arbol -> ansatz ->
    // parte real si el campo es real) que optimiza la busqueda, sin copias de
    // las formulas del ansatz aca. u_exact es la referencia (exacta o
    // numerica). Problemas no estacionarios: se exporta t=0.5. Campos
    // complejos: columnas extra con la parte imaginaria.
    const double t = prob.is_unsteady ? 0.5 : 0.0;
    const bool cplx = prob.complex_field;
    if (prob.dim == 1) {
        f << "x,u_exact,u_approx" << (cplx ? ",u_exact_im,u_approx_im" : "") << "\n";
        for (int i = 0; i <= 100; ++i) {
            double x = i / 100.0;
            Complex ue = prob.reference(x, 0.0, t);
            Complex ua = eval_solution(prob, champion.tree.get(), x, 0.0, t, champion.raw);
            f << x << "," << ue.real() << "," << ua.real();
            if (cplx) f << "," << ue.imag() << "," << ua.imag();
            f << "\n";
        }
    } else {
        f << "x,y,u_exact,u_approx" << (cplx ? ",u_exact_im,u_approx_im" : "") << "\n";
        for (int i = 0; i <= 30; ++i) {
            for (int j = 0; j <= 30; ++j) {
                double x = i / 30.0, y = j / 30.0;
                Complex ue = prob.reference(x, y, t);
                Complex ua = eval_solution(prob, champion.tree.get(), x, y, t, champion.raw);
                f << x << "," << y << "," << ue.real() << "," << ua.real();
                if (cplx) f << "," << ue.imag() << "," << ua.imag();
                f << "\n";
            }
        }
    }
}

// ─── Estadísticas de una corrida ─────────────────────────────────────────────
struct Stats {
    std::string method, pde;
    int    front_size  = 0;
    double best_domain = 1e18;
    double best_bnd    = 1e18;
    double mean_domain = 0.0;
    double mean_bnd    = 0.0;
    double runtime_s   = 0.0;
    double hypervolume = 0.0;
    double min_bic     = 1e18;
    double infeasible_ratio = 0.0;
    double bloat_factor = 0.0;
    // Error de SOLUCION del campeon (vs. verdad exacta/numerica, en la grilla
    // fija de validacion) — distinto de best_domain (residuo de la EDP, no
    // supervisado). -1 si no hay campeon. Esta es la metrica comparable con
    // el MSE que reportan PySR/PySINDy (que sí entrenan contra la solucion).
    double solution_mse = -1.0;
};

// ─── Hipervolumen 3D: mse_domain × mse_boundary × tree_size (normalizado) ─────
// Algoritmo: slice-by-slice sweep sobre el tercer objetivo (tree_size).
// Complejidad: O(n² log n), exacto para 3 objetivos.
// Referencia: Emmerich et al. (2006), WFG algorithm.
//
// Objetivos (todos a minimizar):
//   f1 = mse_domain     (ref: ref_dom)
//   f2 = mse_boundary   (ref: ref_bnd)
//   f3 = tree_size_norm (ref: 1.0, normalizado al máximo permitido)
// ─────────────────────────────────────────────────────────────────────────────

// Calcula HV 2D del conjunto de puntos 2D respecto al punto de referencia (rx, ry).
// Los puntos deben estar en el espacio (minimización); solo se consideran los que
// dominan al punto de referencia (xi < rx AND yi < ry).
static double hv2d(std::vector<std::pair<double,double>> pts, double rx, double ry)
{
    // Filtrar puntos que no dominan la referencia
    pts.erase(std::remove_if(pts.begin(), pts.end(),
        [rx, ry](const std::pair<double,double>& p){
            return p.first >= rx || p.second >= ry;
        }), pts.end());

    if (pts.empty()) return 0.0;

    // Ordenar por primer objetivo (ascendente)
    std::sort(pts.begin(), pts.end(),
              [](const std::pair<double,double>& a, const std::pair<double,double>& b){
                  return a.first < b.first;
              });

    // Sweepline: acumular área
    double hv = 0.0;
    double prev_x = pts[0].first;
    double cur_min_y = pts[0].second;

    for (size_t i = 1; i < pts.size(); ++i) {
        double x = pts[i].first;
        if (cur_min_y < ry)                         // solo si contribuye
            hv += (x - prev_x) * (ry - cur_min_y);
        cur_min_y = std::min(cur_min_y, pts[i].second);
        prev_x = x;
    }
    if (cur_min_y < ry)
        hv += (rx - prev_x) * (ry - cur_min_y);

    return hv;
}

template<typename Ind>
double compute_hypervolume(const std::vector<Ind>& pop,
                           const PDEProblem& prob,
                           double ref_dom  = 1e4,
                           double ref_bnd  = 1e4,
                           double ref_size = 1.2)   // tree_size normalizado
{
    // ── Recopilar puntos del frente de Pareto (rank == 1) ──────────────────
    struct Pt3 { double f1, f2, f3; };
    std::vector<Pt3> pts;

    // Determinar max tree_size para normalización
    double max_ts = 1.0;
    for (auto& ind : pop)
        if (ind.rank == 1 && ind.is_physically_complete(prob))
            max_ts = std::max(max_ts, (double)ind.tree_size);

    for (auto& ind : pop) {
        if (ind.rank != 1 || !ind.is_physically_complete(prob)) continue;
        double f1 = ind.mse_domain;
        double f2 = ind.mse_boundary;
        double f3 = (double)ind.tree_size / max_ts;   // normalizar a [0,1]

        // Filtrar puntos fuera de la caja de referencia
        if (f1 >= ref_dom || f2 >= ref_bnd || f3 >= ref_size) continue;
        pts.push_back({f1, f2, f3});
    }

    if (pts.empty()) return 0.0;

    // ── Slice-by-slice sobre f3 (ascendente = menor tree_size primero) ─────
    // Ordenar por f3 ascendente
    std::sort(pts.begin(), pts.end(),
              [](const Pt3& a, const Pt3& b){ return a.f3 < b.f3; });

    double hv3 = 0.0;
    double prev_f3 = pts[0].f3;

    // Para cada "slice" entre f3[i-1] y f3[i], calcular el HV 2D del
    // conjunto acumulado de puntos proyectados sobre (f1, f2).
    std::vector<std::pair<double,double>> slice_pts;

    for (size_t i = 0; i < pts.size(); ++i) {
        slice_pts.push_back({pts[i].f1, pts[i].f2});

        // Grosor del slice en la dimensión f3
        double next_f3 = (i + 1 < pts.size()) ? pts[i + 1].f3 : ref_size;
        double thickness = next_f3 - prev_f3;

        if (thickness > 0.0) {
            double area = hv2d(slice_pts, ref_dom, ref_bnd);
            hv3 += area * thickness;
        }
        prev_f3 = next_f3;
    }

    // Normalizar por el volumen total de la caja de referencia
    return hv3 / (ref_dom * ref_bnd * ref_size);
}

// `champion`, si se entrega, es el individuo del Hall of Fame (ya validado en
// grilla fija) del que se toman best_domain/best_bnd/min_bic — el resto de las
// estadisticas (front_size, medias, infeasible_ratio, bloat_factor, hipervolumen)
// siguen viniendo del escaneo de toda la poblacion final, que es diversidad
// real y no necesita el campeon validado.
template<typename Ind>
Stats compute_stats(const std::vector<Ind>& pop,
                    const PDEProblem& prob,
                    const std::string& method,
                    const std::string& pde,
                    double rt,
                    const Ind* champion = nullptr,
                    double champion_solution_mse = -1.0)
{
    Stats s;
    s.method = method; s.pde = pde; s.runtime_s = rt;
    int infeasible = 0;
    int total_nodes = 0;
    for (auto& ind : pop) {
        if (!ind.is_feasible) {
            infeasible++;
        }
        total_nodes += ind.tree_size;

        // Análisis del Frente de Pareto (Individuos de Rank 1 y Físicamente Completos)
        if (ind.rank == 1 && ind.is_feasible && ind.is_physically_complete(prob)) {
            s.front_size++;
            s.mean_domain += ind.mse_domain;
            s.mean_bnd += ind.mse_boundary;
        }
    }
    if (champion && champion->tree) {
        double n_samples = 400.0; // aprox n_domain + n_boundary
        double current_total = champion->mse_domain + champion->mse_boundary;
        s.best_domain = champion->mse_domain;
        s.best_bnd = champion->mse_boundary;
        s.min_bic = n_samples * std::log(std::max(current_total, 1e-16)) + champion->tree_size * std::log(n_samples);
        s.solution_mse = champion_solution_mse;
    }
    if (!pop.empty()) {
        s.infeasible_ratio = 100.0 * infeasible / pop.size();
        s.bloat_factor = (double)total_nodes / pop.size();
    }
    if (s.front_size > 0) {
        s.mean_domain /= s.front_size;
        s.mean_bnd    /= s.front_size;
    }
    s.hypervolume = compute_hypervolume(pop, prob);
    return s;
}

// ─── Guardar resumen comparativo ─────────────────────────────────────────────
// Reescribe `path` conservando las filas de EDPs que esta corrida NO toco (la
// columna `pde_col` identifica la EDP). Asi `--only X` reemplaza solo las
// filas de X en vez de borrar el resumen de las demas.
static void merge_csv(const std::string& path, const std::string& header, int pde_col,
                      const std::set<std::string>& replaced, const std::vector<std::string>& new_rows) {
    std::vector<std::string> kept;
    {
        std::ifstream in(path);
        std::string line;
        bool first = true;
        while (std::getline(in, line)) {
            if (first) { first = false; if (line != header) break; continue; } // esquema viejo: se descarta
            if (line.empty()) continue;
            std::stringstream ss(line); std::string cell; int c = 0; std::string pde;
            while (std::getline(ss, cell, ',')) { if (c == pde_col) { pde = cell; break; } ++c; }
            if (!replaced.count(pde)) kept.push_back(line);
        }
    }
    std::ofstream f(path);
    f << header << "\n";
    for (auto& l : kept) f << l << "\n";
    for (auto& l : new_rows) f << l << "\n";
}

static std::string fmt_num(double v) {
    std::ostringstream o; o << std::setprecision(10) << std::scientific << v; return o.str();
}

void save_summary(const std::vector<Stats>& stats, const std::string& path) {
    const std::string header = "method,pde,pareto_size,best_mse_domain,best_mse_boundary,"
                               "mean_mse_domain,mean_mse_boundary,best_bic,runtime_s,solution_mse";
    std::set<std::string> replaced; std::vector<std::string> rows;
    for (auto& s : stats) {
        replaced.insert(s.pde);
        rows.push_back(s.method + "," + s.pde + "," + std::to_string(s.front_size) + "," +
                       fmt_num(s.best_domain) + "," + fmt_num(s.best_bnd) + "," +
                       fmt_num(s.mean_domain) + "," + fmt_num(s.mean_bnd) + "," +
                       fmt_num(s.min_bic) + "," + fmt_num(s.runtime_s) + "," + fmt_num(s.solution_mse));
    }
    merge_csv(path, header, 1, replaced, rows);
}

// ─── Tabla en consola (Solo PISR-NSGA-II) ──────────────────────────────────────
void print_table(const std::string& lbl, const Stats& p) {
    // MSE Bnd. ya no se imprime: con el ansatz de frontera exacta aplicado,
    // ese objetivo es 0 constante para casi toda la suite y desde el fix
    // anterior tampoco es un objetivo real de NSGA-II ahi (ver
    // use_boundary_objective_ en pi_solver.cpp) — mostrar siempre "0.0000e+00"
    // no aportaba nada. Sigue calculandose internamente (Navier-Stokes SI lo
    // usa como objetivo real) y exportandose en los CSV, solo se saca de esta
    // tabla resumen.
    std::cout << "\n+------------------+-------------+----------+-------------+-------------+----------+\n";
    std::cout << "| " << std::left << std::setw(78) << ("  Ecuacion: " + lbl) << "|\n";
    std::cout << "+------------------+-------------+----------+-------------+-------------+----------+\n";
    std::cout << "| Metodo           | MSE Dom.    | Pareto   | Best BIC    | MSE Solucion| Tiempo   |\n";
    std::cout << "+------------------+-------------+----------+-------------+-------------+----------+\n";
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "| PISR-NSGA-II     | " << std::setw(11) << p.best_domain
              << std::defaultfloat << std::setprecision(4)
              << " | " << std::setw(8)  << p.front_size
              << std::scientific << std::setprecision(4)
              << " | " << std::setw(11) << p.min_bic
              << " | " << std::setw(11) << p.solution_mse
              << std::defaultfloat << std::setprecision(4)
              << " | " << std::setw(7)  << p.runtime_s << "s |\n";
    std::cout << "+------------------+-------------+----------+-------------+-------------+----------+\n";
    std::cout << "| Health Analytics | Infeasible: " << std::setw(5) << std::fixed << std::setprecision(1) << p.infeasible_ratio << "% | Bloat Factor (Avg nodes): " << std::setw(5) << p.bloat_factor << "         |\n";
    std::cout << "+------------------+-------------+----------+-------------+-------------+----------+\n";
}

static bool g_skip_oracle = false;

// ─── Una corrida completa (Solo PISR-NSGA-II) ──────────────────────────────────
std::vector<Stats> run_once(int run_id, const std::string& out_dir, bool verbose, bool is_test, const std::string& only_pde, bool replicable) {
    unsigned seed_base = replicable ? (1000u * (unsigned)(run_id + 1)) : std::random_device{}();
    // Suite unica (la misma que recorre oracle_test).
    std::vector<PDEProblem> all_problems = benchmark_suite();

    std::vector<PDEProblem> problems;
    if (!only_pde.empty()) {
        for (auto& p : all_problems) {
            if (problem_label(p) == only_pde) {
                problems.push_back(p);
            }
        }
        if (problems.empty()) {
            std::cerr << "[ERROR] PDE '" << only_pde << "' no encontrada.\n";
        }
    } else {
        problems = all_problems;
    }

    std::vector<Stats> all_stats;


    for (auto& prob : problems) {
        std::string lbl = problem_label(prob);

        if (verbose) std::cout << "\n[Run] PISR-NSGA-II en " << lbl << "\n";
        // Referencia de alta precision para EXACTAMENTE este problema (solo se
        // usa para evaluar resultados despues de la busqueda).
        bool ref_ok = true;
        if (prob.is_numerical) {
            auto rep = NumericalSolver::reference(prob);
            ref_ok = rep.ok;
            if (verbose) std::cout << "  [Referencia] " << rep.method << "  n=" << rep.n
                                   << "  residuo discreto=" << rep.newton_residual
                                   << "  dif. Richardson=" << rep.richardson_diff
                                   << (rep.ok ? "" : "  [NO CONVERGIO]") << "\n";
        }
        // Oraculo: si el planteamiento o la maquinaria son incoherentes, no se corre.
        if (!g_skip_oracle) {
            OracleReport orc = run_oracle(prob);
            if (!orc.pass || !ref_ok) {
                print_oracle_report(orc);
                std::cerr << "[ERROR] Oraculo reprobado en " << lbl << " (usar --skip-oracle para forzar). Se omite.\n";
                continue;
            }
            if (verbose && run_id == 0) std::cout << "  [Oraculo] OK\n";
        }
        // Las funciones de exportacion de este archivo usan esta copia de prob.
        prob.priors = probe_priors(prob);

        // Linea de base RK4/FDM: el solver numerico a baja resolucion (50 nodos
        // por eje), medido contra la referencia. Solo problemas estacionarios.
        bool has_numerical = false;
        double num_rt = 0.0, num_mse_dom = 0.0, num_mse_bnd = 0.0;
        {
            auto t_num0 = std::chrono::steady_clock::now();
            auto num_sol = NumericalSolver::solve(prob, 50);
            num_rt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_num0).count();
            if (!num_sol.empty()) {
                has_numerical = true;
                double sum_sq = 0.0; int n = 0;
                int N = (prob.dim == 1) ? (int)num_sol.size() : (int)std::lround(std::sqrt((double)num_sol.size()));
                double h = 1.0 / (N - 1);
                for (int i = 0; i < N; ++i) {
                    if (prob.dim == 1) { sum_sq += std::norm(num_sol[i] - prob.reference(i * h, 0.0)); ++n; continue; }
                    for (int j = 0; j < N; ++j) { sum_sq += std::norm(num_sol[i * N + j] - prob.reference(i * h, j * h)); ++n; }
                }
                num_mse_dom = n > 0 ? sum_sq / n : 0.0;
            }
        }

        {
            auto t0 = std::chrono::steady_clock::now();
            PISolver pi(prob, seed_base + 500u);
            int pop = is_test ? 10 : Config::POP_SIZE;
            int gen = is_test ? 5 : Config::MAX_GEN;
            auto pi_pop = pi.run(pop, gen);
            
            // ─── Post-Evolution Polishing ───
            // Pulimos al campeón del Hall of Fame (ya validado en grilla fija dentro
            // de PISolver, ver update_hall_of_fame) en vez de re-elegir "el mejor" acá
            // escaneando mse de entrenamiento de la población final — ese segundo
            // escaneo podía terminar eligiendo (y puliendo, y exportando) un individuo
            // que sólo sobreajustaba el mini-batch de la última generación, no la
            // solución real. is_physically_complete (chequeado antes de admitir al
            // campeón) ya cubre el filtro de variables que se hacía acá.
            PIIndividual champion;
            bool has_champion = pi.has_champion();
            double champion_sol_mse = -1.0;
            if (has_champion) {
                const PIIndividual& hof = pi.champion();
                champion.mse_domain = hof.mse_domain; champion.mse_boundary = hof.mse_boundary;
                champion.rank = hof.rank; champion.crowding = hof.crowding;
                champion.tree_size = hof.tree_size; champion.root_type = hof.root_type;
                champion.is_feasible = hof.is_feasible; champion.constraint_violation = hof.constraint_violation;
                if (hof.tree) champion.tree = hof.tree->clone(); champion.raw = hof.raw;
                pi.polish_constants(champion);
                // Error contra la referencia: EVALUACION, despues de la busqueda.
                champion_sol_mse = pi.reference_error(champion);
            }

            double pi_rt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            save_pareto_csv(pi_pop, out_dir + "/" + lbl + "_pi_gn_pareto.csv", "PISR-NSGA-II", prob.name(), prob.dim);
            save_pareto_csv(pi.unpolished_final_population(), out_dir + "/" + lbl + "_pi_gn_pareto_unpolished.csv", "PISR-NSGA-II", prob.name(), prob.dim);
            save_population_archive_csv(pi.population_archive(), out_dir + "/" + lbl + "_pi_gn_population_archive.csv", "PISR-NSGA-II", prob.name(), prob.dim);
            save_convergence_csv(pi.history(), out_dir + "/" + lbl + "_pi_gn_convergence.csv");
            if (has_champion) {
                save_best_expression(champion, prob, out_dir + "/expr_" + lbl + "_PISR-EMOAD.tex", prob.dim);
                save_best_grid(champion, prob, out_dir + "/grid_" + lbl + "_PISR-EMOAD.csv");
            }

            Stats ps = compute_stats(pi_pop, prob, "PISR-NSGA-II", lbl, pi_rt, has_champion ? &champion : nullptr, champion_sol_mse);
            if (verbose) print_table(lbl, ps);
            all_stats.push_back(ps);
        }

        if (has_numerical) {
            Stats ns;
            ns.method = "RK4/FDM";
            ns.pde = lbl;
            ns.front_size = 1;
            ns.best_domain = num_mse_dom;
            ns.best_bnd = num_mse_bnd;
            ns.mean_domain = num_mse_dom;
            ns.mean_bnd = num_mse_bnd;
            ns.runtime_s = num_rt;
            ns.hypervolume = 0.0;
            ns.solution_mse = num_mse_dom; // error contra prob.reference(), misma metrica que el campeon
            all_stats.push_back(ns);
        }
    }
    return all_stats;
}

void print_usage(char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --runs N          Number of independent runs (default: 1)\n"
              << "  --test            Fast test mode (small pop/gen)\n"
              << "  --replicable      Use deterministic seeds for reproducibility\n"
              << "  --only NAME       Run only a specific PDE (e.g., Airy_1D)\n"
              << "  --pop N           Population size (default: " << Config::POP_SIZE << ")\n"
              << "  --gen N           Max generations (default: " << Config::MAX_GEN << ")\n"
              << "  --domain N        Number of domain points (default: " << Config::N_DOMAIN << ")\n"
              << "  --boundary N      Number of boundary points (default: " << Config::N_BOUNDARY << ")\n"
              << "  --depth N         Max tree depth (default: " << Config::MAX_TREE_DEPTH << ")\n"
              << "  --sigma F         ERC mutation sigma (default: " << Config::ERC_SIGMA << ")\n"
              << "  --stop F          Convergence threshold (default: " << Config::STOP_THRESHOLD << ")\n"
              << "  --cores N         Number of CPU threads (default: 1)\n"
              << "  --skip-oracle     Run even if the oracle test fails (not recommended)\n"
              << "  --no-hybrid       Only ansatz individuals (no direct-U representation; ablation)\n"
              << "  --help            Show this help\n";
}

int main(int argc, char* argv[]) {
    int n_runs = 1;
    bool is_test = false;
    bool replicable = false;
    std::string only_pde = "";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--runs") == 0 && i+1 < argc) n_runs = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--test") == 0) is_test = true;
        else if (std::strcmp(argv[i], "--replicable") == 0) replicable = true;
        else if (std::strcmp(argv[i], "--only") == 0 && i+1 < argc) only_pde = argv[++i];
        else if (std::strcmp(argv[i], "--pop") == 0 && i+1 < argc) Config::POP_SIZE = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--gen") == 0 && i+1 < argc) Config::MAX_GEN = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--domain") == 0 && i+1 < argc) Config::N_DOMAIN = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--boundary") == 0 && i+1 < argc) Config::N_BOUNDARY = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--depth") == 0 && i+1 < argc) Config::MAX_TREE_DEPTH = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--sigma") == 0 && i+1 < argc) Config::ERC_SIGMA = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--stop") == 0 && i+1 < argc) Config::STOP_THRESHOLD = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--general") == 0) Config::GENERAL_MODE = true;
        else if (std::strcmp(argv[i], "--no-ansatz") == 0) Config::USE_ANSATZ = false;
        else if (std::strcmp(argv[i], "--skip-oracle") == 0) g_skip_oracle = true;
        else if (std::strcmp(argv[i], "--no-hybrid") == 0) Config::HYBRID_REPR = false;
        else if (std::strcmp(argv[i], "--cores") == 0 && i+1 < argc) Config::CORES = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--help") == 0) { print_usage(argv[0]); return 0; }
    }

    // Aplicar configuración de paralelismo
    if (Config::CORES > 0) {
        omp_set_num_threads(Config::CORES);
    } else {
        Config::CORES = omp_get_max_threads();
        omp_set_num_threads(Config::CORES);
    }

    if (is_test) {
        int max_threads = omp_get_max_threads();
        int threads_to_use = std::max(1, max_threads - 2);
        omp_set_num_threads(threads_to_use);
        std::cout << "[INFO] Modo --test activado. Usando " << threads_to_use << " nucleos.\n";
        Config::POP_SIZE = 10;
        Config::MAX_GEN = 5;
    }

    std::cout << "=============================================================\n";
    std::cout << "  PISR-NSGA-II --- Orquestador de Ecuaciones PDE\n";
    std::cout << "  Pop=" << Config::POP_SIZE << "  Gen=" << Config::MAX_GEN << "  Runs=" << n_runs << "\n";
    if (replicable) std::cout << "  [!] Modo --replicable ACTIVADO (Semillas Deterministas)\n";
    std::cout << "=============================================================\n\n";

    // La ablacion sin representacion hibrida (--no-hybrid) escribe aparte, para
    // no pisar los resultados del metodo por defecto; el reporte las compara.
    const std::string base_dir = Config::HYBRID_REPR ? "results" : "results/ablation_no_hybrid";
    if (!Config::HYBRID_REPR) std::cout << "  [!] --no-hybrid: resultados en " << base_dir << "/\n\n";
    fs::create_directories(base_dir);
    bool verbose = true; 
    std::vector<std::vector<Stats>> all_runs;

    auto t0_all = std::chrono::steady_clock::now();
    for (int r = 0; r < n_runs; ++r) {
        if (n_runs > 1) {
            std::cout << "\n\033[1;34m" << "#############################################################" << "\033[0m\n";
            std::cout << "\033[1;34m" << "  INICIANDO RUN " << r + 1 << " / " << n_runs << "\033[0m\n";
            std::cout << "\033[1;34m" << "#############################################################" << "\033[0m\n";
        }
        
        auto t0_run = std::chrono::steady_clock::now();
        std::string out_dir = (n_runs == 1) ? base_dir : base_dir + "/run_" + std::to_string(r);
        if (n_runs > 1) fs::create_directories(out_dir);
        
        auto stats = run_once(r, out_dir, verbose, is_test, only_pde, replicable);
        all_runs.push_back(stats);
        save_summary(stats, out_dir + "/comparison_summary.csv");

        if (!verbose) {
            std::cout << "  Run " << r << " completed.\n";
        }
    }

    // Fusion por EDP (igual que comparison_summary.csv): una corrida con
    // --only reemplaza solo las filas de esa EDP.
    {
        const std::string header = "run,method,pde,pareto_size,best_mse_domain,best_mse_boundary,"
                                   "mean_mse_domain,mean_mse_boundary,hypervolume,runtime_s,solution_mse";
        std::set<std::string> replaced; std::vector<std::string> rows;
        for (int r = 0; r < n_runs; ++r)
            for (auto& s : all_runs[r]) {
                replaced.insert(s.pde);
                rows.push_back(std::to_string(r) + "," + s.method + "," + s.pde + "," + std::to_string(s.front_size) + "," +
                               fmt_num(s.best_domain) + "," + fmt_num(s.best_bnd) + "," + fmt_num(s.mean_domain) + "," +
                               fmt_num(s.mean_bnd) + "," + fmt_num(s.hypervolume) + "," + fmt_num(s.runtime_s) + "," +
                               fmt_num(s.solution_mse));
            }
        merge_csv(base_dir + "/all_runs_summary.csv", header, 2, replaced, rows);
    }

    auto t_total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_all).count();
    std::cout << "\n  Benchmark completado en " << t_total << "s. Resultados en ./results/\n";
    return 0;
}
