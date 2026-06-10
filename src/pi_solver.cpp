// =============================================================================
// pi_solver.cpp
// =============================================================================

#include "pi_solver.hpp"
#include "nsga2.hpp"
#include "numerical_solver.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <iomanip>

// ─── PIIndividual::evaluate ───────────────────────────────────────────────────
void PIIndividual::evaluate(const PDEProblem& prob, 
                            const std::vector<Point>& dom, 
                            const std::vector<Point>& bnd, 
                            int current_gen) 
{
    if (!tree) { mse_domain = 1e18; mse_boundary = 1e18; return; }

    double pde_mse = 0.0;
    if (prob.type == PDE::NAVIER_STOKES_UNSTEADY) {
        double nu = prob.k2;
        double h = 0.01; double ht = 0.01;
        for (auto& p : dom) {
            AD ad_c = tree->ad_eval_t(p.x, p.y, p.t, prob.dim);
            double psi_x = ad_c.dx.real(), psi_y = ad_c.dy.real();
            double u = psi_y, v = -psi_x;
            double w = -(ad_c.dxx + ad_c.dyy).real(); 

            AD ad_xp = tree->ad_eval_t(p.x+h, p.y, p.t, prob.dim);
            AD ad_xm = tree->ad_eval_t(p.x-h, p.y, p.t, prob.dim);
            AD ad_yp = tree->ad_eval_t(p.x, p.y+h, p.t, prob.dim);
            AD ad_ym = tree->ad_eval_t(p.x, p.y-h, p.t, prob.dim);

            double w_xp = -(ad_xp.dxx + ad_xp.dyy).real();
            double w_xm = -(ad_xm.dxx + ad_xm.dyy).real();
            double w_yp = -(ad_yp.dxx + ad_yp.dyy).real();
            double w_ym = -(ad_ym.dxx + ad_ym.dyy).real();
            double w_x = (w_xp - w_xm) / (2.0 * h);
            double w_y = (w_yp - w_ym) / (2.0 * h);
            double lap_w = (w_xp + w_xm + w_yp + w_ym - 4.0 * w) / (h * h);
            
            AD ad_tp = tree->ad_eval_t(p.x, p.y, p.t+ht, prob.dim);
            AD ad_tm = tree->ad_eval_t(p.x, p.y, p.t-ht, prob.dim);
            double w_tp = -(ad_tp.dxx + ad_tp.dyy).real();
            double w_tm = -(ad_tm.dxx + ad_tm.dyy).real();
            double w_t = (w_tp - w_tm) / (2.0 * ht);

            Complex res = w_t + u * w_x + v * w_y - nu * lap_w;
            if (!std::isfinite(res.real()) || !std::isfinite(res.imag())) {
                mse_domain = 1e18; mse_boundary = 1e18; return;
            }
            pde_mse += std::norm(res);
        }
        if (!dom.empty()) pde_mse /= dom.size();
    } else if (prob.type == PDE::NAVIER_STOKES) {
        double nu = prob.k2; double h = 0.01;
        for (auto& p : dom) {
            AD ad_c = tree->ad_eval_t(p.x, p.y, p.t, prob.dim);
            double psi_x = ad_c.dx.real(), psi_y = ad_c.dy.real();
            double L_val = (ad_c.dxx + ad_c.dyy).real();
            AD ad_xp = tree->ad_eval_t(p.x+h, p.y, p.t, prob.dim);
            AD ad_xm = tree->ad_eval_t(p.x-h, p.y, p.t, prob.dim);
            AD ad_yp = tree->ad_eval_t(p.x, p.y+h, p.t, prob.dim);
            AD ad_ym = tree->ad_eval_t(p.x, p.y-h, p.t, prob.dim);
            double L_x = ((ad_xp.dxx + ad_xp.dyy).real() - (ad_xm.dxx + ad_xm.dyy).real()) / (2.0*h);
            double L_y = ((ad_yp.dxx + ad_yp.dyy).real() - (ad_ym.dxx + ad_ym.dyy).real()) / (2.0*h);
            double biharmonic = ((ad_xp.dxx + ad_xp.dyy).real() + (ad_xm.dxx + ad_xm.dyy).real() + 
                                 (ad_yp.dxx + ad_yp.dyy).real() + (ad_ym.dxx + ad_ym.dyy).real() - 4.0*L_val) / (h*h);
            Complex res = Complex(psi_y * L_x - psi_x * L_y - nu * biharmonic, 0.0);
            if (!std::isfinite(res.real()) || !std::isfinite(res.imag())) {
                mse_domain = 1e18; mse_boundary = 1e18; return;
            }
            pde_mse += std::norm(res);
        }
        if (!dom.empty()) pde_mse /= dom.size();
    } else {
        for (auto& p : dom) {
            AD ad = tree->ad_eval_t(p.x, p.y, p.t, prob.dim);
            Complex res = prob.pde_residual_ad(ad, p.x, p.y);
            if (!std::isfinite(res.real()) || !std::isfinite(res.imag())) {
                mse_domain = 1e18; mse_boundary = 1e18; return;
            }
            pde_mse += std::norm(res); 
        }
        if (!dom.empty()) pde_mse /= dom.size();
    }

    double raw_bc_mse = 0.0;
    for (auto& p : bnd) {
        Complex val = tree->eval_t(p.x, p.y, p.t);
        Complex bc_val = prob.bc(p.x, p.y);
        Complex diff = val - bc_val;
        if (!std::isfinite(diff.real()) || !std::isfinite(diff.imag())) {
            mse_domain = 1e18; mse_boundary = 1e18; return;
        }
        raw_bc_mse += std::norm(diff);
    }
    if (!bnd.empty()) raw_bc_mse /= bnd.size();

    tree_size = tree->count_nodes();
    root_type = tree->get_type();

    double max_p = 1000.0;
    double current_p = 1.0 + (max_p - 1.0) * std::min(1.0, (double)current_gen / 100.0);
    
    double dim_penalty = 1.0;
    if (!prob.is_numerical) {
        auto d_opt = tree->get_dimension(prob);
        if (!d_opt.has_value() || *d_opt != prob.dim_u) {
            dim_penalty = current_p; 
        }
    }

    double complexity_penalty = 1.0;
    if (tree_size > 25) {
        complexity_penalty = 1.0 + 0.1 * (tree_size - 25); 
    }

    // Puro Pareto modificado: Cerramos el "agujero legal" de las constantes.
    // Ahora Obj1 = Error Total, Obj2 = Error Frontera.
    mse_domain = (pde_mse + raw_bc_mse) * dim_penalty * complexity_penalty; 
    mse_boundary = raw_bc_mse * dim_penalty * complexity_penalty; 
}

double PIIndividual::get_validation_mse(const PDEProblem& prob, 
                                        const std::vector<Point>& val_dom, 
                                        const std::vector<Point>& val_bnd) 
{
    if (!tree) return 1e18;
    
    if (prob.is_numerical && !prob.numerical_truth.empty()) {
        double sum_val = 0.0;
        int n_pts = 0;
        if (prob.dim == 1) {
            for (auto& pt : val_dom) {
                Complex u_approx = tree->eval_t(pt.x, pt.y, pt.t);
                Complex u_exact  = prob.numerical_exact(pt.x, pt.y);
                if (!std::isfinite(u_approx.real())) continue;
                sum_val += std::norm(u_approx - u_exact);
                ++n_pts;
            }
            return (n_pts > 0) ? sum_val / n_pts : 1e18;
        } else {
            int N = (int)std::sqrt(prob.numerical_truth.size());
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    double x = i / (double)(N - 1);
                    double y = j / (double)(N - 1);
                    Complex val = tree->eval_t(x, y, 0.0);
                    if (!std::isfinite(val.real())) continue;
                    sum_val += std::norm(val - prob.numerical_truth[i * N + j]);
                    ++n_pts;
                }
            }
            return (n_pts > 0) ? sum_val / n_pts : 1e18;
        }
    }

    double sum_dom = 0.0;
    if (prob.type == PDE::NAVIER_STOKES_UNSTEADY) {
        double nu = prob.k2; double h = 0.01; double ht = 0.01;
        for (auto& p : val_dom) {
            AD ad_c = tree->ad_eval_t(p.x, p.y, p.t, prob.dim);
            double psi_x = ad_c.dx.real(), psi_y = ad_c.dy.real();
            double u = psi_y, v = -psi_x;
            double w = -(ad_c.dxx + ad_c.dyy).real(); 
            
            AD ad_xp = tree->ad_eval_t(p.x+h, p.y, p.t, prob.dim);
            AD ad_xm = tree->ad_eval_t(p.x-h, p.y, p.t, prob.dim);
            AD ad_yp = tree->ad_eval_t(p.x, p.y+h, p.t, prob.dim);
            AD ad_ym = tree->ad_eval_t(p.x, p.y-h, p.t, prob.dim);
            double w_xp = -(ad_xp.dxx + ad_xp.dyy).real();
            double w_xm = -(ad_xm.dxx + ad_xm.dyy).real();
            double w_yp = -(ad_yp.dxx + ad_yp.dyy).real();
            double w_ym = -(ad_ym.dxx + ad_ym.dyy).real();
            double w_x = (w_xp - w_xm) / (2.0 * h);
            double w_y = (w_yp - w_ym) / (2.0 * h);
            double lap_w = (w_xp + w_xm + w_yp + w_ym - 4.0 * w) / (h * h);
            
            AD ad_tp = tree->ad_eval_t(p.x, p.y, p.t+ht, prob.dim);
            AD ad_tm = tree->ad_eval_t(p.x, p.y, p.t-ht, prob.dim);
            double w_tp = -(ad_tp.dxx + ad_tp.dyy).real();
            double w_tm = -(ad_tm.dxx + ad_tm.dyy).real();
            double w_t = (w_tp - w_tm) / (2.0 * ht);

            Complex res = w_t + u * w_x + v * w_y - nu * lap_w;
            if (!std::isfinite(res.real()) || !std::isfinite(res.imag())) return 1e18;
            sum_dom += std::norm(res);
        }
    } else if (prob.type == PDE::NAVIER_STOKES) {
        double h = 0.01; double nu = prob.k2;
        for (auto& p : val_dom) {
            AD ad_c = tree->ad_eval_t(p.x, p.y, p.t, prob.dim);
            double psi_x = ad_c.dx.real(), psi_y = ad_c.dy.real();
            double L_val = (ad_c.dxx + ad_c.dyy).real();
            AD ad_xp = tree->ad_eval_t(p.x+h, p.y, p.t, prob.dim);
            AD ad_xm = tree->ad_eval_t(p.x-h, p.y, p.t, prob.dim);
            AD ad_yp = tree->ad_eval_t(p.x, p.y+h, p.t, prob.dim);
            AD ad_ym = tree->ad_eval_t(p.x, p.y-h, p.t, prob.dim);
            double L_x = ((ad_xp.dxx + ad_xp.dyy).real() - (ad_xm.dxx + ad_xm.dyy).real()) / (2.0*h);
            double L_y = ((ad_yp.dxx + ad_yp.dyy).real() - (ad_ym.dxx + ad_ym.dyy).real()) / (2.0*h);
            double biharmonic = ((ad_xp.dxx + ad_xp.dyy).real() + (ad_xm.dxx + ad_xm.dyy).real() + 
                                 (ad_yp.dxx + ad_yp.dyy).real() + (ad_ym.dxx + ad_ym.dyy).real() - 4.0*L_val) / (h*h);
            Complex res = Complex(psi_y * L_x - psi_x * L_y - nu * biharmonic, 0.0);
            if (!std::isfinite(res.real()) || !std::isfinite(res.imag())) return 1e18;
            sum_dom += std::norm(res);
        }
    } else {
        for (auto& p : val_dom) {
            AD ad = tree->ad_eval_t(p.x, p.y, p.t, prob.dim);
            Complex res = prob.pde_residual_ad(ad, p.x, p.y);
            if (!std::isfinite(res.real()) || !std::isfinite(res.imag())) return 1e18;
            sum_dom += std::norm(res);
        }
    }
    double sum_bnd = 0.0;
    for (auto& p : val_bnd) {
        Complex val = tree->eval_t(p.x, p.y, p.t);
        Complex bc_val = prob.bc(p.x, p.y);
        Complex diff = val - bc_val;
        if (!std::isfinite(diff.real()) || !std::isfinite(diff.imag())) return 1e18;
        sum_bnd += std::norm(diff);
    }
    if (!val_dom.empty()) sum_dom /= val_dom.size();
    if (!val_bnd.empty()) sum_bnd /= val_bnd.size();

    // Para validación/selección del "Best Ever", usamos suma simple sin pesos
    return sum_dom + sum_bnd; 
}

// ─── PISolver ─────────────────────────────────────────────────────────────────
PISolver::PISolver(const PDEProblem& prob, unsigned seed)
    : prob_(prob), gen_(seed)
{
    dom_pts_ = prob_.domain_points(Config::N_DOMAIN);
    bnd_pts_ = prob_.boundary_points(Config::N_BOUNDARY);
    int n_val = (prob.dim == 1) ? 200 : 400;
    val_dom_pts_ = prob_.domain_points(n_val);
    val_bnd_pts_ = prob_.boundary_points(n_val / 2);
}

PIIndividual PISolver::random_individual() {
    PIIndividual ind;
    ind.tree = random_tree(Config::MAX_TREE_DEPTH, gen_);
    ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
    return ind;
}

PIIndividual PISolver::random_individual_special() {
    PIIndividual ind;
    ind.tree = random_tree_special(Config::MAX_TREE_DEPTH, gen_, prob_);
    ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
    return ind;
}

PIIndividual PISolver::make_offspring(const PIIndividual& a, const PIIndividual& b) {
    std::uniform_real_distribution<double> p_dist(0.0, 1.0);
    PIIndividual child;
    bool is_elite = (a.rank == 1 || b.rank == 1);

    if (p_dist(gen_) < Config::CROSSOVER_PROB) {
        auto [c1, c2] = tree_crossover(a.tree, b.tree, gen_);
        child.tree = std::move(c1);
    } else {
        child.tree = a.tree->clone();
    }

    if (is_elite) {
        if (p_dist(gen_) < 0.7) child.tree->mutate_erc(gen_, Config::ERC_SIGMA * 0.5);
        else                    child.tree = tree_mutate(child.tree, gen_, prob_);
    } else {
        if (p_dist(gen_) < Config::MUTATION_PROB)
            child.tree = tree_mutate(child.tree, gen_, prob_);
        if (p_dist(gen_) < 0.4)
            child.tree->mutate_erc(gen_, Config::ERC_SIGMA);
    }
    return child;
}

void PISolver::update_hall_of_fame() {
    for (auto& ind : population_) {
        if (ind.rank == 1) {
            double v_mse = ind.get_validation_mse(prob_, val_dom_pts_, val_bnd_pts_);
            if (!has_best_ever_ || v_mse < best_ever_.mse_domain) {
                best_ever_.mse_domain = v_mse; 
                best_ever_.mse_boundary = ind.mse_boundary;
                best_ever_.rank = ind.rank;
                best_ever_.crowding = ind.crowding;
                best_ever_.tree_size = ind.tree_size;
                best_ever_.root_type = ind.root_type;
                if (ind.tree) best_ever_.tree = ind.tree->clone();
                has_best_ever_ = true;
                stagnation_counter_ = 0;
            }
        }
    }
}

std::vector<PIIndividual> PISolver::run(int pop_size, int max_gen) {
    population_.clear();
    has_best_ever_ = false;
    cataclysm_count_ = 0;
    stagnation_counter_ = 0;

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto gen_pts = [&]() {
        dom_pts_.clear();
        bnd_pts_.clear();
        if (prob_.dim == 1) {
            for (int i = 0; i < Config::N_DOMAIN; ++i)
                dom_pts_.push_back({dist(gen_), 0.0, dist(gen_)});
            bnd_pts_.push_back({0.0, 0.0, dist(gen_)});
            bnd_pts_.push_back({1.0, 0.0, dist(gen_)});
        } else {
            for (int i = 0; i < Config::N_DOMAIN; ++i)
                dom_pts_.push_back({dist(gen_), dist(gen_), dist(gen_)});
            int pts_per_side = Config::N_BOUNDARY / 5;
            for (int i = 0; i < pts_per_side; ++i) {
                double t_pt = dist(gen_);
                double s_pt = dist(gen_);
                bnd_pts_.push_back({0.0, s_pt, t_pt});
                bnd_pts_.push_back({1.0, s_pt, t_pt});
                bnd_pts_.push_back({s_pt, 0.0, t_pt});
                bnd_pts_.push_back({s_pt, 1.0, t_pt});
                bnd_pts_.push_back({dist(gen_), dist(gen_), 0.0}); 
            }
        }
    };

    gen_pts();

    NodePtr exact_tree = get_exact_solution_tree(prob_);
    int n_exact = (exact_tree) ? static_cast<int>(0.25 * pop_size) : 0;
    int n_special = static_cast<int>(0.40 * pop_size); 
    for (int i = 0; i < pop_size; ++i) {
        if (i < n_exact && exact_tree) {
            PIIndividual ind;
            ind.tree = exact_tree->clone();
            ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
            population_.push_back(std::move(ind));
        } else if (i < n_exact + n_special) {
            PIIndividual special_ind = random_individual_special();
            special_ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
            population_.push_back(std::move(special_ind));
        } else {
            population_.push_back(random_individual());
        }
    }

    population_ = nsga2_select_next(std::move(population_), pop_size);
    update_hall_of_fame();

    int n_threads = omp_get_max_threads();
    std::vector<std::mt19937> gen_thread(n_threads);
    for (int i = 0; i < n_threads; ++i) gen_thread[i].seed(gen_() + i);

    for (int g = 0; g < max_gen; ++g) {
        current_gen_ = g;
        if (g > 0 && g % 20 == 0) {
            gen_pts();
            #pragma omp parallel for
            for (size_t i = 0; i < population_.size(); ++i) {
                population_[i].evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
            }
        }

        update_hall_of_fame();

        if (g > 0 && last_best_mse_ - best_ever_.mse_domain < 1e-7) {
            stagnation_counter_++;
        } else {
            stagnation_counter_ = 0;
            last_best_mse_ = best_ever_.mse_domain;
        }

        if (stagnation_counter_ >= 100) {
            std::cout << "  [!] Early Stop: Estancamiento prolongado detectado (" << stagnation_counter_ 
                      << " gens sin mejora). Terminando en gen " << g << ".\n";
            break;
        }

        std::vector<PIIndividual> offspring;
        offspring.reserve(pop_size);
        
        if (has_best_ever_) {
            PIIndividual elite;
            elite.tree = best_ever_.tree->clone();
            elite.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
            offspring.push_back(std::move(elite));
        }

        while ((int)offspring.size() < pop_size) {
            int p1 = tournament_select(population_, gen_);
            int p2 = tournament_select(population_, gen_);
            PIIndividual child = make_offspring(population_[p1], population_[p2]);
            offspring.push_back(std::move(child));
        }

        #pragma omp parallel for
        for (size_t i = 0; i < offspring.size(); ++i) {
            offspring[i].evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
        }

        std::vector<PIIndividual> combined;
        combined.reserve(population_.size() + offspring.size());
        for (auto& p : population_) combined.push_back(std::move(p));
        for (auto& o : offspring) combined.push_back(std::move(o));
        population_ = nsga2_select_next(std::move(combined), pop_size);

        // ─── ESTRATEGIA MEMÉTICA INTENSIVA ──────────────────────────────────
        // Cada 10 generaciones, el Top 5% del frente de Pareto recibe entrenamiento pesado
        if (g > 0 && g % 10 == 0) {
            int n_elite = std::max(1, (int)(pop_size * 0.05));
            #pragma omp parallel for
            for (int i = 0; i < n_elite; ++i) {
                hill_climb_constants(population_[i], 500); // 10 veces más iteraciones
            }
        } else {
            // Hill Climbing estándar para el top 20% en generaciones normales
            int n_hc = pop_size * 0.20;
            #pragma omp parallel for
            for (int i = 0; i < n_hc; ++i) {
                hill_climb_constants(population_[i], 50);
            }
        }

        double current_best_train = 1e18;
        for (auto& ind : population_) {
            if (ind.rank == 1) current_best_train = std::min(current_best_train, ind.mse_domain);
        }

        double b_tot = has_best_ever_ ? best_ever_.mse_domain : current_best_train;
        history_.push_back({g, 0.0, 0.0, b_tot});

        if (g % 25 == 0) {
            double b_dom = 1e18, b_bnd = 1e18;
            for (auto& ind : population_) {
                if (ind.rank == 1) {
                    b_dom = std::min(b_dom, ind.mse_domain);
                    b_bnd = std::min(b_bnd, ind.mse_boundary);
                }
            }
            double v_mse = has_best_ever_ ? best_ever_.mse_domain : 1e18;
            std::cout << "  [PI/" << prob_.name() << "] gen=" << g
                      << "  best_dom=" << std::scientific << std::setprecision(3) << b_dom
                      << "  best_bnd=" << b_bnd 
                      << "  (val_best=" << v_mse << ")" << std::defaultfloat << "\n";
        }

        double stop_threshold = Config::STOP_THRESHOLD;
        if (prob_.is_numerical) stop_threshold = 1e-4; 

        if (current_best_train < stop_threshold) {
            std::cout << "  [!] Convergencia alcanzada (MSE=" << current_best_train
                      << "). Terminando en gen " << g << ".\n";
            break;
        }

        double b_dom_cur = 1e18, b_bnd_cur = 1e18;
        for (auto& ind : population_) {
            if (ind.rank == 1) {
                b_dom_cur = std::min(b_dom_cur, ind.mse_domain);
                b_bnd_cur = std::min(b_bnd_cur, ind.mse_boundary);
            }
        }
        if (b_dom_cur < 1e-12 && b_bnd_cur < 1e-12) {
            std::cout << "  [!] Solución perfecta (dom=" << b_dom_cur
                      << ", bnd=" << b_bnd_cur << "). Terminando en gen " << g << ".\n";
            break;
        }
    }
    return std::move(population_);
}

std::vector<PIIndividual> PISolver::pareto_front() const {
    std::vector<PIIndividual> front;
    for (auto& ind : population_) {
        if (ind.rank == 1) {
            PIIndividual copy;
            copy.mse_domain = ind.mse_domain;
            copy.mse_boundary = ind.mse_boundary;
            copy.rank = ind.rank;
            copy.crowding = ind.crowding;
            copy.tree_size = ind.tree_size;
            copy.root_type = ind.root_type;
            if (ind.tree) copy.tree = ind.tree->clone();
            front.push_back(std::move(copy));
        }
    }
    return front;
}

// ─── Optimización Local Pura (Hill-Climbing) ───────────────────────────────────
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
