#include "pi_solver.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <omp.h>

// use_boundary: cuando el ansatz de frontera exacta aplica, mse_boundary es
// EXACTAMENTE 0 para todo individuo — no aporta nada a la dominancia (0<=0
// siempre se cumple, nunca "<" estricto) pero SI contamina crowding distance:
// con todos los valores empatados en 0, calculate_crowding_distance ordena un
// empate arbitrario y le asigna "distancia infinita" (1e15) a dos individuos
// cualesquiera solo por quedar primero/ultimo en ese orden sin sentido,
// inflando su crowding muy por encima de lo que aportan mse_domain/tree_size
// (las unicas dimensiones con informacion real) y distorsionando la seleccion.
// Por eso se excluye el objetivo de frontera por completo (no solo se ignora
// en la comparacion) cuando se sabe que es constante — ver ansatz_applies().
static void fast_non_dominated_sort(std::vector<PIIndividual>& pop, bool use_boundary = true) {
    int n = static_cast<int>(pop.size());
    std::vector<std::vector<int>> S(n);
    std::vector<int> count(n, 0);
    std::vector<int> front;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            bool i_dom_j = false;
            if (pop[i].is_feasible && !pop[j].is_feasible) i_dom_j = true;
            else if (!pop[i].is_feasible && !pop[j].is_feasible) {
                if (pop[i].constraint_violation < pop[j].constraint_violation) i_dom_j = true;
            } else if (pop[i].is_feasible && pop[j].is_feasible) {
                bool bnd_le = !use_boundary || pop[i].mse_boundary <= pop[j].mse_boundary;
                bool bnd_lt = use_boundary && pop[i].mse_boundary < pop[j].mse_boundary;
                if ((pop[i].mse_domain <= pop[j].mse_domain && bnd_le && pop[i].tree_size <= pop[j].tree_size) &&
                    (pop[i].mse_domain < pop[j].mse_domain || bnd_lt || pop[i].tree_size < pop[j].tree_size))
                    i_dom_j = true;
            }
            if (i_dom_j) S[i].push_back(j);
            else {
                bool j_dom_i = false;
                if (pop[j].is_feasible && !pop[i].is_feasible) j_dom_i = true;
                else if (!pop[j].is_feasible && !pop[i].is_feasible) {
                    if (pop[j].constraint_violation < pop[i].constraint_violation) j_dom_i = true;
                } else if (pop[j].is_feasible && pop[i].is_feasible) {
                    bool bnd_le = !use_boundary || pop[j].mse_boundary <= pop[i].mse_boundary;
                    bool bnd_lt = use_boundary && pop[j].mse_boundary < pop[i].mse_boundary;
                    if ((pop[j].mse_domain <= pop[i].mse_domain && bnd_le && pop[j].tree_size <= pop[i].tree_size) &&
                        (pop[j].mse_domain < pop[i].mse_domain || bnd_lt || pop[j].tree_size < pop[i].tree_size))
                        j_dom_i = true;
                }
                if (j_dom_i) count[i]++;
            }
        }
        if (count[i] == 0) { pop[i].rank = 1; front.push_back(i); }
    }
    int curr = 1;
    while (!front.empty()) {
        std::vector<int> next_front;
        for (int i : front) {
            for (int j : S[i]) {
                count[j]--;
                if (count[j] == 0) { pop[j].rank = curr + 1; next_front.push_back(j); }
            }
        }
        front = next_front; curr++;
    }
}

static void calculate_crowding_distance(std::vector<PIIndividual>& pop, const std::vector<int>& front_indices, bool use_boundary = true) {
    if (front_indices.empty()) return;
    int n = static_cast<int>(front_indices.size());
    for (int idx : front_indices) pop[idx].crowding = 0.0;
    auto assign_dist = [&](auto getter) {
        std::vector<int> sorted = front_indices;
        std::sort(sorted.begin(), sorted.end(), [&](int a, int b) { return getter(pop[a]) < getter(pop[b]); });
        pop[sorted[0]].crowding = 1e15; pop[sorted[n-1]].crowding = 1e15;
        double range = getter(pop[sorted[n-1]]) - getter(pop[sorted[0]]);
        if (range < 1e-9) range = 1e-9;
        for (int i = 1; i < n-1; ++i) pop[sorted[i]].crowding += (getter(pop[sorted[i+1]]) - getter(pop[sorted[i-1]])) / range;
    };
    assign_dist([](const PIIndividual& ind) { return ind.mse_domain; });
    if (use_boundary) assign_dist([](const PIIndividual& ind) { return ind.mse_boundary; });
    assign_dist([](const PIIndividual& ind) { return static_cast<double>(ind.tree_size); });
}

// Eliminacion de duplicados (estandar en NSGA-II con representacion discreta):
// individuos con el mismo vector de objetivos no se dominan entre si, asi que
// un atractor trivial (p. ej. N=0, tamaño 1) puede clonarse hasta llenar el
// frente 1 y congelar la busqueda. Las copias repetidas pasan al final y solo
// entran si faltan individuos unicos. No excluye ninguna expresion: la
// primera copia compite normalmente. Devuelve las copias retiradas de `pop`.
static std::vector<PIIndividual> split_duplicates(std::vector<PIIndividual>& pop) {
    auto same = [](double a, double b) { return std::abs(a - b) <= 1e-12 * std::max(1.0, std::abs(a)); };
    std::vector<int> idx(pop.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (pop[a].tree_size != pop[b].tree_size) return pop[a].tree_size < pop[b].tree_size;
        if (pop[a].mse_domain != pop[b].mse_domain) return pop[a].mse_domain < pop[b].mse_domain;
        return pop[a].mse_boundary < pop[b].mse_boundary;
    });
    std::vector<char> dup(pop.size(), 0);
    for (size_t k = 1; k < idx.size(); ++k) {
        const PIIndividual& p = pop[idx[k - 1]];
        const PIIndividual& q = pop[idx[k]];
        if (p.tree_size == q.tree_size && p.is_feasible == q.is_feasible &&
            same(p.mse_domain, q.mse_domain) && same(p.mse_boundary, q.mse_boundary)) dup[idx[k]] = 1;
    }
    std::vector<PIIndividual> uniq, rest;
    for (size_t i = 0; i < pop.size(); ++i) (dup[i] ? rest : uniq).push_back(std::move(pop[i]));
    pop = std::move(uniq);
    return rest;
}

std::vector<PIIndividual> nsga2_select_next(std::vector<PIIndividual> combined, int pop_size, bool use_boundary = true) {
    std::vector<PIIndividual> dups = split_duplicates(combined);
    if (combined.size() < static_cast<size_t>(pop_size)) {
        // Faltan unicos: se completa con copias (quedan al final por rango).
        for (size_t i = 0; combined.size() < static_cast<size_t>(pop_size) && i < dups.size(); ++i)
            combined.push_back(std::move(dups[i]));
    }
    fast_non_dominated_sort(combined, use_boundary);
    std::vector<PIIndividual> next_pop; int rank = 1;
    while (next_pop.size() < static_cast<size_t>(pop_size)) {
        std::vector<int> front;
        for (size_t i = 0; i < combined.size(); ++i) if (combined[i].rank == rank) front.push_back(i);
        if (front.empty()) break;
        calculate_crowding_distance(combined, front, use_boundary);
        if (next_pop.size() + front.size() <= static_cast<size_t>(pop_size)) {
            for (int i : front) next_pop.push_back(std::move(combined[i]));
        } else {
            std::sort(front.begin(), front.end(), [&](int a, int b) {
                if (std::abs(combined[a].crowding - combined[b].crowding) > 1e-6)
                    return combined[a].crowding > combined[b].crowding;
                return combined[a].tree_size < combined[b].tree_size;
            });
            for (size_t i = 0; next_pop.size() < static_cast<size_t>(pop_size); ++i) 
                next_pop.push_back(std::move(combined[front[i]]));
        }
        rank++;
    }
    return next_pop;
}

static int tournament_select(const std::vector<PIIndividual>& pop, std::mt19937& gen) {
    int best = -1;
    for (int i = 0; i < Config::TOURNAMENT_SIZE; ++i) {
        int idx = std::uniform_int_distribution<int>(0, static_cast<int>(pop.size()) - 1)(gen);
        if (best == -1) best = idx;
        else {
            if (pop[idx].rank < pop[best].rank) best = idx;
            else if (pop[idx].rank == pop[best].rank && pop[idx].crowding > pop[best].crowding) best = idx;
        }
    }
    return best;
}

namespace Config {
    int    POP_SIZE       = 500;   
    int    MAX_GEN        = 1000;   
    int    N_DOMAIN       = 500;  
    int    N_BOUNDARY     = 200;   
    double ERC_SIGMA      = 0.20;  
    int    MAX_TREE_DEPTH = 6;
    double CROSSOVER_PROB = 0.80;  
    double MUTATION_PROB  = 0.45;   
    int    TOURNAMENT_SIZE = 4;    
    // Antes 1e-12: demasiado estricto para activarse en la practica — la
    // mayoria de las corridas convergen en el rango 1e-4 a 1e-8 (ver
    // benchmarks de la suite), asi que el early stop casi nunca disparaba.
    // 1e-6 dispara cuando el campeon ya alcanzo una solucion muy buena, sin
    // ser tan laxo como para cortar corridas que todavia podrian mejorar
    // ordenes de magnitud (Laplace/Airy/Lane-Emden llegan a 1e-9..1e-12).
    double STOP_THRESHOLD  = 1e-6;

    int    RAR_INTERVAL       = 25;
    int    RAR_CANDIDATES     = 10000;
    double RAR_ADAPTIVE_RATIO = 0.30;
    int    RAR_ELITE_COUNT    = 4;
    double RAR_RANDOM_RATIO   = 0.40;
    bool   GENERAL_MODE       = false;
    bool   USE_ANSATZ         = true;
    bool   HYBRID_REPR        = true;
    int    CORES              = 1;
    bool   IS_SINGULAR        = false; // Detectado dinamicamente
}

// ─── Ansatz de frontera exacta ────────────────────────────────────────────────
// U = L + B*N, donde N es el arbol y L, B cumplen los datos de borde por
// construccion (B se anula donde hay dato, L lo interpola). Aplicabilidad por
// PROPIEDADES del problema, no por nombre de EDP:
//  - no estacionario: el ansatz es espacial y no conoce t (datos de borde que
//    cambian en el tiempo y la condicion inicial no quedarian impuestos);
//  - orden espacial > 2: solo esta implementado hasta segundas derivadas, y el
//    problema necesita tambien du/dn en la frontera.
// En esos casos la frontera es un objetivo medido (boundary_error) en NSGA-II.
static bool ansatz_applies(const PDEProblem& prob) {
    if (Config::GENERAL_MODE || !Config::USE_ANSATZ) return false;
    if (prob.is_unsteady) return false;
    if (prob.spatial_order > 2) return false;
    return true;
}
// El individuo usa el ansatz si aplica al problema y su arbol no es U directo.
static bool uses_ansatz(const PDEProblem& prob, bool raw) { return !raw && ansatz_applies(prob); }
// Representacion hibrida activa: hay individuos de ambos tipos en la poblacion.
static bool hybrid_active(const PDEProblem& prob) { return Config::HYBRID_REPR && ansatz_applies(prob); }

// Factor del ansatz por N en el valor de U. EXACTAMENTE en la frontera el
// factor es 0 y U vale el dato de borde aunque N no se pueda evaluar ahi
// (p. ej. N = (u - L)/B da 0/0): es el limite que garantiza el ansatz cuando
// N esta acotada cerca del borde, y eso lo verifica ansatz_border_check.
static Complex ansatz_term(double factor, Complex n) { return factor == 0.0 ? Complex(0.0) : factor * n; }

static AD apply_boundary_ansatz(const PDEProblem& prob, const AD& ad, double x, double y) {
    AD U;
    if (prob.dim == 1) {
        if (prob.bc_kind == BCKind::IVP_LEFT) {
            // U = u0 + u0' x + x^2 N: impone u(0)=u0 y u'(0)=u0' a la vez.
            Complex u0 = prob.ivp_u0, du0 = prob.ivp_du0;
            U.v = u0 + du0 * x + ansatz_term(x * x, ad.v);
            U.dx = du0 + 2.0 * x * ad.v + x * x * ad.dx;
            U.dxx = 2.0 * ad.v + 4.0 * x * ad.dx + x * x * ad.dxx;
            return U;
        }
        if (prob.bc_kind == BCKind::ANCHOR_LEFT) {
            // Solo u(0) dado: U = u0 + x N.
            Complex u0 = prob.ivp_u0;
            U.v = u0 + ansatz_term(x, ad.v);
            U.dx = ad.v + x * ad.dx;
            U.dxx = 2.0 * ad.dx + x * ad.dxx;
            return U;
        }
        Complex u0 = prob.bc(0.0, 0.0, 0.0), u1 = prob.bc(1.0, 0.0, 0.0);
        Complex L = u0 + x * (u1 - u0), dL = u1 - u0;
        double B = x * (x - 1.0), dB = 2.0 * x - 1.0, d2B = 2.0;
        U.v = L + ansatz_term(B, ad.v);
        U.dx = dL + dB * ad.v + B * ad.dx;
        U.dxx = d2B * ad.v + 2.0 * dB * ad.dx + B * ad.dxx;
        return U;
    }

    // 2D: interpolacion transfinita (Coons) de los datos en las 4 aristas.
    // Las derivadas de los datos a lo largo de cada arista se toman por
    // diferencias centradas sobre bc() evaluado EN la arista.
    const double hfd = 1e-4;
    Complex g0 = prob.bc(0.0, y), g1 = prob.bc(1.0, y);
    Complex h0 = prob.bc(x, 0.0), h1 = prob.bc(x, 1.0);
    Complex c00 = prob.bc(0.0, 0.0), c10 = prob.bc(1.0, 0.0);
    Complex c01 = prob.bc(0.0, 1.0), c11 = prob.bc(1.0, 1.0);
    Complex g0_yp = prob.bc(0.0, y + hfd), g0_ym = prob.bc(0.0, y - hfd);
    Complex g1_yp = prob.bc(1.0, y + hfd), g1_ym = prob.bc(1.0, y - hfd);
    Complex h0_xp = prob.bc(x + hfd, 0.0), h0_xm = prob.bc(x - hfd, 0.0);
    Complex h1_xp = prob.bc(x + hfd, 1.0), h1_xm = prob.bc(x - hfd, 1.0);
    Complex g0_y = (g0_yp - g0_ym) / (2.0 * hfd), g0_yy = (g0_yp - 2.0 * g0 + g0_ym) / (hfd * hfd);
    Complex g1_y = (g1_yp - g1_ym) / (2.0 * hfd), g1_yy = (g1_yp - 2.0 * g1 + g1_ym) / (hfd * hfd);
    Complex h0_x = (h0_xp - h0_xm) / (2.0 * hfd), h0_xx = (h0_xp - 2.0 * h0 + h0_xm) / (hfd * hfd);
    Complex h1_x = (h1_xp - h1_xm) / (2.0 * hfd), h1_xx = (h1_xp - 2.0 * h1 + h1_xm) / (hfd * hfd);

    Complex L = (1.0 - x) * g0 + x * g1 + (1.0 - y) * h0 + y * h1
              - (1.0 - x) * (1.0 - y) * c00 - x * (1.0 - y) * c10
              - (1.0 - x) * y * c01 - x * y * c11;
    Complex L_x = -g0 + g1 + (1.0 - y) * h0_x + y * h1_x
                + (1.0 - y) * c00 - (1.0 - y) * c10 + y * c01 - y * c11;
    Complex L_xx = (1.0 - y) * h0_xx + y * h1_xx;
    Complex L_y = (1.0 - x) * g0_y + x * g1_y - h0 + h1
                + (1.0 - x) * c00 + x * c10 - (1.0 - x) * c01 - x * c11;
    Complex L_yy = (1.0 - x) * g0_yy + x * g1_yy;

    double B = x * (1.0 - x) * y * (1.0 - y);
    double B_x = (1.0 - 2.0 * x) * y * (1.0 - y), B_xx = -2.0 * y * (1.0 - y);
    double B_y = x * (1.0 - x) * (1.0 - 2.0 * y), B_yy = -2.0 * x * (1.0 - x);

    U.v = L + ansatz_term(B, ad.v);
    U.dx = L_x + B_x * ad.v + B * ad.dx;
    U.dxx = L_xx + B_xx * ad.v + 2.0 * B_x * ad.dx + B * ad.dxx;
    U.dy = L_y + B_y * ad.v + B * ad.dy;
    U.dyy = L_yy + B_yy * ad.v + 2.0 * B_y * ad.dy + B * ad.dyy;
    return U;
}

// ─── Unica cadena de evaluacion de la solucion ────────────────────────────────
// arbol -> jet (orden 4 si el operador lo pide) -> ansatz (si aplica) ->
// parte real (si el campo es real). La usan entrenamiento, validacion, RAR,
// pesos por punto, gradiente de constantes, reporte y oraculo.
static AD tree_jet(const PDEProblem& prob, const Node* tree, double x, double y, double t) {
    return (prob.spatial_order > 2) ? tree->ad_eval_ext_t(x, y, t, prob.dim)
                                    : tree->ad_eval_t(x, y, t, prob.dim);
}

static AD solution_jet(const PDEProblem& prob, const Node* tree, double x, double y, double t, bool use_ansatz) {
    AD a = tree_jet(prob, tree, x, y, t);
    if (use_ansatz) a = apply_boundary_ansatz(prob, a, x, y);
    return prob.complex_field ? a : project_real(a);
}

static Complex residual_at(const PDEProblem& prob, const Node* tree, const Point& p, bool use_ansatz) {
    return prob.pde_residual_ad(solution_jet(prob, tree, p.x, p.y, p.t, use_ansatz), p.x, p.y, p.t);
}

// Error de frontera medido. Con ansatz es 0 por construccion (la verificacion
// de que la construccion realmente funciona es ansatz_border_check).
// Combinacion lineal de jets completos (para extrapolar valor y derivadas).
static AD ad_lincomb3(double a, const AD& A, double b, const AD& B, double c, const AD& C) {
    AD r;
    auto f = [&](Complex AD::*m) { r.*m = a * (A.*m) + b * (B.*m) + c * (C.*m); };
    for (Complex AD::*m : {&AD::v, &AD::dx, &AD::dy, &AD::dt, &AD::dxx, &AD::dyy, &AD::dtt,
                           &AD::dxy, &AD::dxt, &AD::dyt, &AD::dxxx, &AD::dxxy, &AD::dxyy, &AD::dyyy,
                           &AD::dxxt, &AD::dyyt, &AD::dxxxx, &AD::dxxyy, &AD::dyyyy}) f(m);
    return r;
}

// Direccion hacia el interior del dominio desde un punto de frontera.
static Point inward_dir(const PDEProblem& prob, const Point& p) {
    Point n{0.0, 0.0, 0.0};
    if (p.x == 0.0) n.x = 1.0; else if (p.x == 1.0) n.x = -1.0;
    if (prob.dim == 2) { if (p.y == 0.0) n.y = 1.0; else if (p.y == 1.0) n.y = -1.0; }
    if (n.x == 0.0 && n.y == 0.0 && prob.is_unsteady && p.t == 0.0) n.t = 1.0;
    return n;
}

// Sin ansatz (U directo) la condicion de borde se exige como LIMITE lateral,
// no solo en el punto: si no, un pico exp(-(x/eps)^2) con eps -> 0 fija U(0)
// exactamente mientras el interior resuelve otra solucion (u(0+) != dato) y
// ningun punto de colocacion cae dentro del pico. El limite se estima por
// extrapolacion cubica 3U(d)-3U(2d)+U(3d) a dos escalas (1e-3 y 1e-5; entre
// ambas cubren cualquier ancho de pico que el residuo no vea). Sesgo para
// soluciones suaves O(d^3); para la singularidad x^{3/2} de Thomas-Fermi ~1e-5.
static double boundary_error(const PDEProblem& prob, const Node* tree, const std::vector<Point>& bnd, bool use_ansatz) {
    if (use_ansatz || bnd.empty()) return 0.0;
    double s = 0.0;
    for (auto& p : bnd) {
        double e = prob.boundary_mismatch_sq(solution_jet(prob, tree, p.x, p.y, p.t, false), p);
        Point n = inward_dir(prob, p);
        if (n.x != 0.0 || n.y != 0.0 || n.t != 0.0) {
            for (double d : {1e-3, 1e-5}) {
                auto at = [&](double k) {
                    return solution_jet(prob, tree, p.x + k * d * n.x, p.y + k * d * n.y, p.t + k * d * n.t, false);
                };
                AD lim = ad_lincomb3(3.0, at(1), -3.0, at(2), 1.0, at(3));
                double el = prob.boundary_mismatch_sq(lim, p);
                e = std::isfinite(el) ? std::max(e, el) : el;
            }
        }
        s += std::isfinite(e) ? std::min(e, 1e4) : 1e4;
    }
    return s / bnd.size();
}

// Verificacion del ansatz en el borde: U debe tender al dato de borde. La
// garantia de U=L+B*N solo vale si N no diverge cerca del borde; un N ~ C/x
// deja B*N -> C != 0 y rompe la condicion sin que el residuo lo note. Se
// evalua U muy cerca de cada parte de la frontera con dato y se compara con el
// dato EN la frontera.
struct BorderCheck { bool finite = true; double max_gap = 0.0; double sum_sq = 0.0; };

// U debe estar definida EXACTAMENTE en la frontera (dominio cerrado), no solo
// cerca: una singularidad removible (0/x, x log x sin limite evaluable) pasa
// el chequeo a distancia 1e-6 pero deja NaN en x=0 en la solucion exportada.
// Aplica con y sin ansatz.
static bool finite_on_boundary(const PDEProblem& prob, const Node* tree, bool use_ansatz) {
    std::vector<double> ts = prob.is_unsteady ? std::vector<double>{0.0, 0.5, 1.0} : std::vector<double>{0.0};
    auto ok = [&](double x, double y, double t) {
        Complex v = solution_jet(prob, tree, x, y, t, use_ansatz).v;
        return std::isfinite(v.real()) && std::isfinite(v.imag());
    };
    for (double t : ts) {
        if (prob.dim == 1) {
            if (!ok(0.0, 0.0, t) || !ok(1.0, 0.0, t)) return false;
            continue;
        }
        for (double s : {0.0, 0.25, 0.5, 0.75, 1.0})
            if (!ok(0.0, s, t) || !ok(1.0, s, t) || !ok(s, 0.0, t) || !ok(s, 1.0, t)) return false;
    }
    return true;
}
static BorderCheck ansatz_border_check(const PDEProblem& prob, const Node* tree) {
    constexpr double E = 1e-6;
    BorderCheck bc;
    auto add = [&](Complex gap) {
        double g = std::abs(gap);
        if (!std::isfinite(g)) { bc.finite = false; return; }
        bc.max_gap = std::max(bc.max_gap, g);
        bc.sum_sq += g * g;
    };
    if (prob.dim == 1) {
        AD a = solution_jet(prob, tree, E, 0.0, 0.0, true);
        if (prob.bc_kind == BCKind::IVP_LEFT) {
            add(a.v - prob.ivp_u0);
            add(a.dx - prob.ivp_du0);
        } else if (prob.bc_kind == BCKind::ANCHOR_LEFT) {
            add(a.v - prob.ivp_u0);
        } else {
            add(a.v - prob.bc(0.0, 0.0, 0.0));
            AD b = solution_jet(prob, tree, 1.0 - E, 0.0, 0.0, true);
            add(b.v - prob.bc(1.0, 0.0, 0.0));
        }
        return bc;
    }
    for (double s : {0.25, 0.5, 0.75}) {
        add(solution_jet(prob, tree, E, s, 0.0, true).v - prob.bc(0.0, s));
        add(solution_jet(prob, tree, 1.0 - E, s, 0.0, true).v - prob.bc(1.0, s));
        add(solution_jet(prob, tree, s, E, 0.0, true).v - prob.bc(s, 0.0));
        add(solution_jet(prob, tree, s, 1.0 - E, 0.0, true).v - prob.bc(s, 1.0));
    }
    return bc;
}

// Perdidas de entrenamiento (sin chequeos de factibilidad). Unica definicion,
// usada por evaluate() y por el gradiente de constantes.
static void compute_losses(const PDEProblem& prob, const Node* tree,
                           const std::vector<Point>& dom, const std::vector<Point>& bnd,
                           const std::vector<double>* dom_weights, bool use_ansatz,
                           double& mse_dom, double& mse_bnd, bool& finite) {
    finite = true;
    double s = 0.0, wsum = 0.0;
    for (size_t k = 0; k < dom.size(); ++k) {
        Complex res = residual_at(prob, tree, dom[k], use_ansatz);
        if (!std::isfinite(res.real()) || !std::isfinite(res.imag())) { finite = false; mse_dom = mse_bnd = 1e18; return; }
        // Recorte por punto: un outlier finito pero astronomico (p. ej. cerca
        // de una singularidad) no debe dominar el promedio.
        double w = (dom_weights && k < dom_weights->size()) ? (*dom_weights)[k] : 1.0;
        s += w * std::min(std::norm(res), 1e4);
        wsum += w;
    }
    mse_dom = (wsum > 0.0) ? s / wsum : 0.0;
    mse_bnd = Config::GENERAL_MODE ? 0.0 : boundary_error(prob, tree, bnd, use_ansatz);
}

Complex eval_solution(const PDEProblem& prob, const Node* tree, double x, double y, double t, bool raw) {
    return solution_jet(prob, tree, x, y, t, uses_ansatz(prob, raw)).v;
}

bool solution_uses_ansatz(const PDEProblem& prob, bool raw) { return uses_ansatz(prob, raw); }

double physics_loss(const PDEProblem& prob, const Node* tree,
                    const std::vector<Point>& dom, const std::vector<Point>& bnd, bool raw) {
    double d, b; bool fin;
    compute_losses(prob, tree, dom, bnd, nullptr, uses_ansatz(prob, raw), d, b, fin);
    return fin ? d + b : 1e18;
}

// Gradiente de la perdida respecto de cada constante por diferencias
// centradas. Correcto aunque el arbol sea complejo (CONST_I, rotaciones de
// fase), aunque haya proyeccion a parte real o |.| en el operador, y aunque
// la perdida sum|R|^2 no sea holomorfa: se perturba la constante REAL y se
// mide la perdida real. (El paso complejo anterior, Im f(c+ih)/h, solo es
// valido si f es real para c real; con e^{i theta} o i en el arbol producia
// gradientes espurios ~1/h.)
std::vector<double> constant_gradient(const PDEProblem& prob, Node* tree,
                                      const std::vector<Point>& dom, const std::vector<Point>& bnd,
                                      double h_rel, bool raw) {
    std::vector<Complex*> ercs; tree->collect_ercs(ercs);
    std::vector<double> g(ercs.size(), 0.0);
    bool use_ansatz = uses_ansatz(prob, raw);
    for (size_t k = 0; k < ercs.size(); ++k) {
        Complex orig = *ercs[k];
        double h = h_rel * std::max(1.0, std::abs(orig.real()));
        double dp, bp, dm, bm; bool fp, fm;
        *ercs[k] = Complex(orig.real() + h, orig.imag());
        compute_losses(prob, tree, dom, bnd, nullptr, use_ansatz, dp, bp, fp);
        *ercs[k] = Complex(orig.real() - h, orig.imag());
        compute_losses(prob, tree, dom, bnd, nullptr, use_ansatz, dm, bm, fm);
        *ercs[k] = orig;
        g[k] = (fp && fm) ? ((dp + bp) - (dm + bm)) / (2.0 * h) : 0.0;
        if (!std::isfinite(g[k])) g[k] = 0.0;
    }
    return g;
}

thread_local std::vector<std::string>* PIIndividual::violation_log = nullptr;

void PIIndividual::evaluate(const PDEProblem& prob,
                            const std::vector<Point>& dom,
                            const std::vector<Point>& bnd,
                            int /*current_gen*/,
                            const std::vector<double>* dom_weights)
{
    if (!tree) {
        is_feasible = false; constraint_violation = 1.0;
        mse_domain = 1e18; mse_boundary = 1e18; return;
    }

    is_feasible = true;
    constraint_violation = 0.0;
    auto violate = [&](double w, const char* why) {
        is_feasible = false; constraint_violation += w;
        if (violation_log) violation_log->push_back(why);
    };

    // ── Factibilidad: solo validez de la expresion y gramatica del lenguaje
    // de busqueda. Ningun prior del operador (simetria, separabilidad) excluye
    // candidatos: los priors solo sesgan la inicializacion. Tambien se
    // eliminaron la restriccion anti-plano (excluia la solucion exacta de
    // Laplace 1D, u=x), max_grad>15 (hiperparametro dependiente del problema)
    // y la varianza minima: con ansatz o con frontera medida, un plano o una
    // constante que no es la solucion ya falla la frontera o el residuo.
    if (!is_physically_complete(prob)) violate(10.0, "no usa todas las variables independientes");

    if (!(prob.dim_u == Units::None)) {
        // Las ERC son numeros puros: sin una constante que cargue unidades,
        // ningun arbol hecho solo de variables alcanza otra dimension. El
        // chequeo aplica solo a arboles sin ERC.
        auto d_opt = tree->get_dimension(prob);
        bool has_erc = tree->contains_erc();
        if (!has_erc && (!d_opt.has_value() || *d_opt != prob.dim_u)) violate(5.0, "dimension inconsistente");
    }
    if (tree->get_unary_depth() > 3) violate(1.0, "profundidad unaria > 3");
    if (tree->get_depth() > Config::MAX_TREE_DEPTH) violate(1.0, "profundidad > MAX_TREE_DEPTH");
    if (tree->has_nested_trig()) violate(1.0, "trigonometrica anidada");
    if (tree->has_non_affine_trig_arg()) violate(1.0, "argumento trigonometrico no afin");
    if (tree->has_nested_polynomial()) violate(1.0, "polinomio anidado");
    if (tree->has_nested_exp()) violate(1.0, "exponencial anidada");
    if (tree->has_invalid_polynomial_degree()) violate(1.0, "grado de polinomio invalido");

    bool use_ansatz = uses_ansatz(prob, raw);
    if (use_ansatz) {
        BorderCheck b = ansatz_border_check(prob, tree.get());
        if (!b.finite || b.max_gap > 0.05) violate(1.0, "el ansatz no alcanza el dato de borde (N diverge en el borde)");
    }
    if (!finite_on_boundary(prob, tree.get(), use_ansatz)) violate(1.0, "la solucion no esta definida en la frontera (NaN/inf)");

    bool finite;
    compute_losses(prob, tree.get(), dom, bnd, dom_weights, use_ansatz, mse_domain, mse_boundary, finite);
    if (!finite) violate(10.0, "residuo no finito");

    tree_size = tree->count_nodes();
    root_type = tree->get_type();
}

// Puntaje de validacion: SOLO informacion de la EDP (residuo en una grilla fija
// y densa del interior, distinta de los puntos de entrenamiento; frontera
// medida en una grilla fija; y, con ansatz, la verificacion en el borde). Lo
// usan el Hall of Fame y el early-stop. La referencia nunca entra aqui.
double PIIndividual::validation_score(const PDEProblem& prob,
                                      const std::vector<Point>& val_dom,
                                      const std::vector<Point>& val_bnd)
{
    if (!tree) return 1e18;
    bool use_ansatz = uses_ansatz(prob, raw);
    if (!finite_on_boundary(prob, tree.get(), use_ansatz)) return 1e18;
    double d, b; bool fin;
    compute_losses(prob, tree.get(), val_dom, val_bnd, nullptr, use_ansatz, d, b, fin);
    if (!fin) return 1e18;
    double border = 0.0;
    if (use_ansatz) {
        BorderCheck bc = ansatz_border_check(prob, tree.get());
        if (!bc.finite) return 1e18;
        border = bc.sum_sq;
    }
    return d + b + border;
}

// Error contra la referencia (exacta o numerica) en los puntos dados. SOLO
// para evaluar resultados: lo llaman main.cpp (despues de la busqueda), el
// registro de convergencia (solo escritura) y el oraculo. Ninguna decision del
// algoritmo lo lee.
double PIIndividual::reference_error(const PDEProblem& prob, const std::vector<Point>& pts) const
{
    if (!tree || pts.empty()) return 1e18;
    double s = 0.0; int n = 0;
    for (auto& p : pts) {
        Complex u = eval_solution(prob, tree.get(), p.x, p.y, p.t, raw);
        Complex r = prob.reference(p.x, p.y, p.t);
        double e = std::norm(u - r);
        if (!std::isfinite(e)) return 1e18;
        s += e; ++n;
    }
    return s / n;
}

PISolver::PISolver(const PDEProblem& prob, unsigned seed) : prob_(prob), gen_(seed) {
    priors_ = probe_priors(prob_);
    prob_.priors = priors_; // solo sesgo de inicializacion/esqueletos
    // Grillas FIJAS de validacion (densas, deterministas), distintas de los
    // puntos de entrenamiento (muestreados al azar en run()).
    val_dom_pts_ = prob_.validation_points();
    val_bnd_pts_ = prob_.boundary_points(0);

    // --- Detector A-Priori de Singularidad ---
    auto dummy_tree = make_binary(NodeType::ADD, make_var('x'), make_erc(1.0));
    Complex r_center = residual_at(prob_, dummy_tree.get(), {0.5, 0.5, 0.0}, false);
    Complex r_edge = residual_at(prob_, dummy_tree.get(), {1e-5, 1e-5, 0.0}, false);
    if (std::isfinite(r_center.real()) && std::isfinite(r_edge.real())) {
        if (std::norm(r_edge) > std::norm(r_center) * 1000.0) {
            Config::IS_SINGULAR = true;
            std::cout << "[INFO] Singularidad detectada en el PDE. Se usara Gradient Descent (Clipped Mode)." << std::endl;
        } else {
            Config::IS_SINGULAR = false;
        }
    } else {
        Config::IS_SINGULAR = true; // Ante la duda, asume singular
    }
}

PIIndividual PISolver::random_individual() {
    PIIndividual ind;
    ind.tree = make_binary(NodeType::MUL, make_erc(1.0), random_tree(Config::MAX_TREE_DEPTH, gen_, prob_));
    ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
    return ind;
}

PIIndividual PISolver::random_individual_special() {
    PIIndividual ind;
    ind.tree = make_binary(NodeType::MUL, make_erc(1.0), random_tree_special(Config::MAX_TREE_DEPTH, gen_, prob_, priors_));
    ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
    return ind;
}

PIIndividual PISolver::make_offspring(const PIIndividual& a, const PIIndividual& b) {
    std::uniform_real_distribution<double> p_dist(0.0, 1.0);
    PIIndividual child;
    bool is_elite = (a.rank == 1 || b.rank == 1);
    // Enfriamiento (annealing): exploración agresiva al inicio de la corrida,
    // refinamiento más conservador cerca del final para no destruir por azar
    // estructura físicamente correcta ya encontrada. Con piso: si la
    // agresividad cae a 0, la población pierde toda capacidad de escapar de
    // un óptimo local mediocre (visto empíricamente: val_mse quedándose fijo
    // desde generación 0 en Airy/Duffing incluso con cientos de generaciones
    // restantes) — MIN_AGGRESSIVENESS mantiene siempre una vía de escape.
    double progress = max_gen_ > 0 ? std::clamp((double)current_gen_ / max_gen_, 0.0, 1.0) : 0.0;
    constexpr double MIN_AGGRESSIVENESS = 0.3;
    double aggressiveness = MIN_AGGRESSIVENESS + (1.0 - MIN_AGGRESSIVENESS) * (1.0 - progress);
    if (p_dist(gen_) < Config::CROSSOVER_PROB) {
        auto [c1, c2] = tree_crossover(a.tree, b.tree, gen_);
        child.tree = std::move(c1); child.raw = a.raw;
    } else {
        child.tree = a.tree->clone(); child.raw = a.raw;
    }
    if (is_elite) {
        if (p_dist(gen_) < 0.7) child.tree->mutate_erc(gen_, Config::ERC_SIGMA * 0.5);
        else child.tree = tree_mutate(child.tree, gen_, prob_, aggressiveness);
    } else {
        if (p_dist(gen_) < Config::MUTATION_PROB) child.tree = tree_mutate(child.tree, gen_, prob_, aggressiveness);
        if (p_dist(gen_) < 0.4) child.tree->mutate_erc(gen_, Config::ERC_SIGMA);
    }
    // Cambio de representacion (N <-> U directo): mismo arbol, otro significado.
    if (hybrid_active(prob_) && p_dist(gen_) < 0.05) child.raw = !child.raw;
    return child;
}

void PISolver::update_hall_of_fame() {
    // Paso 1 (barato): entre los candidatos de rank 1 de ESTA generacion, quedarnos
    // con el mejor segun el mini-batch de entrenamiento ya evaluado — no cuesta
    // evaluaciones extra, solo un min() sobre lo que ya se calculo.
    PIIndividual* best_candidate = nullptr;
    double best_train_err = 1e18;
    for (auto& ind : population_) {
        if (ind.rank == 1 && ind.tree_size >= 1 && ind.is_feasible && ind.is_physically_complete(prob_)) {
            double train_err = ind.mse_domain + ind.mse_boundary;
            if (train_err < best_train_err) { best_train_err = train_err; best_candidate = &ind; }
        }
    }
    if (!best_candidate) return;

    // Paso 2: decidir si reemplaza al campeon comparando AMBOS con el puntaje de
    // validacion en las grillas fijas (no el mini-batch, distinto en cada
    // generacion). El puntaje es solo fisica (residuo + frontera + borde del
    // ansatz): antes esta comparacion era contra la solucion de referencia, lo
    // que metia la referencia en la seleccion del campeon.
    double candidate_val = best_candidate->validation_score(prob_, val_dom_pts_, val_bnd_pts_);
    double incumbent_val = has_best_ever_ ? best_ever_.validation_score(prob_, val_dom_pts_, val_bnd_pts_) : 1e18;

    if (!has_best_ever_ || candidate_val < incumbent_val) {
        best_ever_.mse_domain = best_candidate->mse_domain; best_ever_.mse_boundary = best_candidate->mse_boundary;
        best_ever_.rank = best_candidate->rank; best_ever_.crowding = best_candidate->crowding;
        best_ever_.tree_size = best_candidate->tree_size; best_ever_.root_type = best_candidate->root_type;
        if (best_candidate->tree) best_ever_.tree = best_candidate->tree->clone(); best_ever_.raw = best_candidate->raw;
        has_best_ever_ = true;
    }
}

Point PISolver::generate_random_point() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double t_val = (prob_.is_unsteady) ? dist(gen_) : 0.0;
    if (prob_.dim == 1) {
        double u = dist(gen_);
        double x_cheb = 0.5 * (1.0 - std::cos(M_PI * u));
        return {x_cheb, 0.0, t_val};
    }
    return {dist(gen_), dist(gen_), t_val};
}

// Peso adaptativo por punto, inspirado en el analisis NTK de PINNs (Wang et
// al. 2022, "When and why PINNs fail to train"): en vez de promediar el
// residuo parejo entre todos los puntos del batch (lo que "aplana" la señal
// justo donde mas importa — ver el caso de Thomas-Fermi, donde el promedio
// uniforme escondia el mal ajuste cerca de la singularidad), los puntos
// donde el residuo del comite de elites es consistentemente mas alto reciben
// mas peso en la suma. Complementa RAR (que decide QUE puntos entran al
// batch via resampling): esto decide CUANTO pesa cada uno dentro del batch
// actual. Rango [1,3]: nunca por debajo de 1 (no se pierde señal en los
// puntos faciles), hasta 3x en los mas dificiles.
void PISolver::update_point_weights(const std::vector<Point>& pts) {
    point_weights_.assign(pts.size(), 1.0);
    if (pts.empty()) return;
    std::vector<PIIndividual*> committee;
    for (auto& ind : population_) {
        if (ind.rank == 1 && ind.tree) committee.push_back(&ind);
        if (committee.size() >= 6) break;
    }
    if (committee.empty()) return;
    std::vector<double> raw(pts.size(), 0.0);
    // Residuo de U (con ansatz y proyeccion), no del arbol crudo N: antes los
    // pesos se calculaban sobre N, una cantidad sin significado fisico cuando
    // el ansatz aplica.
    for (size_t i = 0; i < pts.size(); ++i) {
        double sum_err = 0.0; int n = 0;
        for (auto* ind : committee) {
            Complex res = residual_at(prob_, ind->tree.get(), pts[i], uses_ansatz(prob_, ind->raw));
            double e = std::norm(res);
            if (std::isfinite(e)) { sum_err += std::min(e, 1e4); n++; }
        }
        raw[i] = (n > 0) ? sum_err / n : 0.0;
    }
    double max_r = *std::max_element(raw.begin(), raw.end());
    if (max_r < 1e-12) return; // todo converge parejo, se dejan los pesos en 1.0
    for (size_t i = 0; i < pts.size(); ++i) point_weights_[i] = 1.0 + 2.0 * (raw[i] / max_r);
}

void PISolver::apply_committee_rar() {
    if (population_.empty()) return;
    std::vector<PIIndividual*> committee;
    std::vector<PIIndividual*> elites;
    for (auto& ind : population_) if (ind.rank == 1) elites.push_back(&ind);
    std::sort(elites.begin(), elites.end(), [](PIIndividual* a, PIIndividual* b){ return a->mse_domain < b->mse_domain; });
    for (int i = 0; i < std::min((int)elites.size(), Config::RAR_ELITE_COUNT); ++i) committee.push_back(elites[i]);
    int n_random = (int)(population_.size() * Config::RAR_RANDOM_RATIO);
    std::vector<int> indices(population_.size()); std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen_);
    for (int i = 0; i < std::min(n_random, (int)indices.size()); ++i) committee.push_back(&population_[indices[i]]);
    if (committee.empty()) return;
    int pool_size = (prob_.dim == 1) ? Config::RAR_CANDIDATES : 2000;
    struct ScoredPoint { Point p; double score; };
    std::vector<ScoredPoint> scored(pool_size);
    unsigned int seed_base = gen_();
    #pragma omp parallel
    {
        std::mt19937 thread_gen(seed_base + omp_get_thread_num());
        std::uniform_real_distribution<double> ud(0, 1);
        #pragma omp for schedule(static)
        for (int i = 0; i < pool_size; ++i) {
            double tx = ud(thread_gen), ty = (prob_.dim == 2) ? ud(thread_gen) : 0.0;
            double tt = (prob_.is_unsteady) ? ud(thread_gen) : 0.0;
            Point p = {tx, ty, tt};
            double sum_error = 0.0; int valid_count = 0;
            for (auto* ind : committee) {
                if (!ind->tree) continue;
                Complex res = residual_at(prob_, ind->tree.get(), p, uses_ansatz(prob_, ind->raw));
                double err = std::norm(res);
                if (std::isfinite(err)) { sum_error += err; valid_count++; }
            }
            double consensus_score = (valid_count > 0) ? (sum_error / valid_count) : 0.0;
            scored[i] = {p, std::isfinite(consensus_score) ? consensus_score : 0.0};
        }
    }
    std::sort(scored.begin(), scored.end(), [](const ScoredPoint& a, const ScoredPoint& b){ return a.score > b.score; });
    int n_total = (prob_.dim == 1) ? Config::N_DOMAIN : 800;
    int n_adaptive = (int)(n_total * Config::RAR_ADAPTIVE_RATIO);
    std::vector<Point> next_pts;
    int n_keep = n_total - n_adaptive;
    if (dom_pts_.size() > (size_t)n_keep) {
        std::shuffle(dom_pts_.begin(), dom_pts_.end(), gen_);
        for (int i = 0; i < n_keep; ++i) next_pts.push_back(dom_pts_[i]);
    } else { next_pts = dom_pts_; }
    for (int i = 0; i < std::min(n_adaptive, (int)scored.size()); ++i) next_pts.push_back(scored[i].p);
    dom_pts_ = std::move(next_pts);
}

std::vector<PIIndividual> PISolver::run(int pop_size, int max_gen) {
    max_gen_ = max_gen;
    population_.clear(); has_best_ever_ = false; population_archive_.clear();
    dom_pts_.clear(); bnd_pts_.clear();
    for (int i = 0; i < Config::N_DOMAIN; ++i) dom_pts_.push_back(generate_random_point());
    // Datos de borde de entrenamiento: muestras al azar del MISMO conjunto que
    // describe la validacion (prob_.boundary_points), incluida la condicion
    // inicial si el problema es no estacionario. Antes se armaban aca a mano
    // y no incluian ni la condicion inicial ni la frontera para t>0 en 1D.
    for (int i = 0; i < Config::N_BOUNDARY; ++i) bnd_pts_.push_back(prob_.sample_boundary_point(gen_));
    // Con representacion hibrida conviven individuos con frontera exacta (0) y
    // con frontera medida: el objetivo de frontera vuelve a tener informacion.
    use_boundary_objective_ = !ansatz_applies(prob_) || hybrid_active(prob_);
    // Priors como SESGO de inicializacion, nunca como restriccion: hasta un
    // tercio de la poblacion arranca con las estructuras que sugiere el
    // operador (separable, suma modal, simetrica), repartido entre los priors
    // activos; el resto es la mezcla habitual de esqueletos y arboles al azar.
    // Antes, si el operador era separable, TODA la poblacion arrancaba
    // separable y ademas una restriccion de factibilidad descartaba cualquier
    // candidato no separable, aunque la solucion no lo fuera.
    enum class Bias { SEPARABLE, MODAL, MIRROR, EXP_MODES };
    std::vector<Bias> biases;
    if (priors_.exponential_modes) biases.push_back(Bias::EXP_MODES);
    if (priors_.triple_separable || priors_.additive_separable || priors_.multiplicative_separable) biases.push_back(Bias::SEPARABLE);
    if (priors_.modal_sum_separable) biases.push_back(Bias::MODAL);
    if (priors_.mirror_symmetric_x || priors_.mirror_symmetric_y) biases.push_back(Bias::MIRROR);
    const int n_biased = biases.empty() ? 0 : pop_size / 3;
    for (int i = 0; i < pop_size; ++i) {
        PIIndividual ind;
        if (i < n_biased) {
            switch (biases[i % biases.size()]) {
                case Bias::SEPARABLE:
                    ind.tree = priors_.triple_separable
                        ? random_triple_separable_tree(Config::MAX_TREE_DEPTH, gen_, prob_)
                        : random_separable_tree(Config::MAX_TREE_DEPTH, gen_, prob_, priors_.multiplicative_separable);
                    break;
                case Bias::MODAL:
                    ind.tree = random_modal_sum_tree(Config::MAX_TREE_DEPTH, gen_, prob_);
                    break;
                case Bias::MIRROR: {
                    // N(x(1-x)) es simetrico ante x <-> 1-x (idem en y). Con
                    // datos de borde simetricos, U = L + B*N tambien lo es.
                    NodePtr t = random_tree_special(Config::MAX_TREE_DEPTH - 1, gen_, prob_, priors_);
                    if (priors_.mirror_symmetric_x) {
                        auto sx = make_binary(NodeType::MUL, make_var('x'), make_binary(NodeType::SUB, make_erc(1.0), make_var('x')));
                        t = substitute_variable(t.get(), NodeType::VAR_X, sx.get());
                    }
                    if (prob_.dim == 2 && priors_.mirror_symmetric_y) {
                        auto sy = make_binary(NodeType::MUL, make_var('y'), make_binary(NodeType::SUB, make_erc(1.0), make_var('y')));
                        t = substitute_variable(t.get(), NodeType::VAR_Y, sy.get());
                    }
                    ind.tree = std::move(t);
                    break;
                }
                case Bias::EXP_MODES:
                    // Los modos describen U, no N: van como U directo si hay
                    // representacion hibrida.
                    ind.tree = random_exponential_modes_tree(Config::MAX_TREE_DEPTH, gen_, prob_);
                    ind.raw = hybrid_active(prob_);
                    break;
            }
        } else if (i < n_biased + (pop_size - n_biased) / 2) {
            // random_tree_special incluye los esqueletos sesgados por los
            // priors restantes (rotacion, amortiguamiento, polo, escala).
            ind.tree = random_tree_special(Config::MAX_TREE_DEPTH, gen_, prob_, priors_);
        } else {
            ind.tree = random_tree(Config::MAX_TREE_DEPTH, gen_, prob_);
        }
        // Un tercio de los no sesgados arranca como U directo (hibrido).
        if (i >= n_biased && hybrid_active(prob_) && (i - n_biased) % 3 == 2) ind.raw = true;
        if (ind.tree) ind.tree = ind.tree->simplify();
        ind.evaluate(prob_, dom_pts_, bnd_pts_, 0);
        population_.push_back(std::move(ind));
    }
    fast_non_dominated_sort(population_, use_boundary_objective_);
    update_hall_of_fame();
    // Poblacion inicial (gen=-1): primer punto del archivo historico, ver
    // population_archive() en el header.
    population_archive_.reserve(pop_size * (max_gen + 1));
    for (auto& ind : population_) {
        population_archive_.push_back({-1, ind.mse_domain, ind.mse_boundary, ind.tree_size, ind.rank});
    }
    int n_threads = omp_get_max_threads();

    // Viven fuera del loop: el batch ahora persiste entre generaciones (se
    // resamplea cada RESAMPLE_INTERVAL, no en cada una — ver comentario abajo).
    std::vector<Point> curr_dom = dom_pts_;
    std::vector<Point> curr_bnd = bnd_pts_;

    for (int g = 0; g < max_gen; ++g) {
        current_gen_ = g;

        // Stochastic Mini-Batching
        // Batch ~80% del pool disponible (antes 8-16%, luego 30-40%). Diagnostico
        // empirico (ver update_hall_of_fame): con batches chicos, una funcion casi
        // trivial (ej. N(x)~=0 en el ansatz L(x)+B(x)*N(x)) puede lograr residuo de
        // entrenamiento carisimamente bajo por pura coincidencia de muestreo, sin
        // parecerse en nada a la solucion real — se vio directamente en Airy_1D:
        // "0.074*x" con train_err=0.00095 pero val_mse=0.82 (55x peor que el
        // campeon real). Un batch mucho mas grande hace estadisticamente mucho mas
        // dificil que una funcion degenerada tenga residuo bajo en TANTOS puntos
        // por casualidad. Con la build en Release (-O3 -march=native) hay margen
        // de sobra de velocidad para pagar el costo extra.
        //
        // Ademas, el batch se resamplea cada RESAMPLE_INTERVAL generaciones, no en
        // cada una. Antes se resampleaba siempre: la seleccion (offspring vs.
        // resto de la poblacion) comparaba contra un blanco que cambiaba a cada
        // paso, asi que una "mejora" de una generacion a la siguiente podia ser pura casualidad
        // de que tocaron puntos mas faciles, no progreso real. Con un blanco
        // estable durante varias generaciones, la presion de seleccion tiene
        // tiempo de premiar mejoras genuinas en vez de ruido de muestreo.
        constexpr int RESAMPLE_INTERVAL = 8;
        auto resample_batch = [&]() {
            // max(1, .): con pocos puntos el truncado a int podia dejar el
            // batch vacio (con 1 punto de frontera, 0.8 -> 0).
            int n_dom = std::max(1, (int)(dom_pts_.size() * 0.8));
            int n_bnd = std::max(1, (int)(bnd_pts_.size() * 0.8));
            curr_dom.clear(); curr_bnd.clear();
            std::sample(dom_pts_.begin(), dom_pts_.end(), std::back_inserter(curr_dom), n_dom, gen_);
            std::sample(bnd_pts_.begin(), bnd_pts_.end(), std::back_inserter(curr_bnd), n_bnd, gen_);
            update_point_weights(curr_dom);
        };

        if (g < max_gen - 50) {
            if (g % RESAMPLE_INTERVAL == 0) resample_batch();
            #pragma omp parallel for
            for (size_t i = 0; i < population_.size(); ++i) {
                population_[i].evaluate(prob_, curr_dom, curr_bnd, current_gen_, &point_weights_);
            }
        } else if (g == max_gen - 50) {
            // Re-evaluate on full grid when switching
            curr_dom = dom_pts_;
            curr_bnd = bnd_pts_;
            update_point_weights(curr_dom);
            #pragma omp parallel for
            for (size_t i = 0; i < population_.size(); ++i) {
                population_[i].evaluate(prob_, curr_dom, curr_bnd, current_gen_, &point_weights_);
            }
        }

        if (g > 0 && g % Config::RAR_INTERVAL == 0) {
            apply_committee_rar();
            // dom_pts_/bnd_pts_ cambiaron (RAR inyecta puntos de residuo alto):
            // reflejarlo en el batch YA, sin esperar al proximo multiplo de
            // RESAMPLE_INTERVAL.
            if (g >= max_gen - 50) {
                curr_dom = dom_pts_;
                curr_bnd = bnd_pts_;
                update_point_weights(curr_dom);
            } else {
                resample_batch();
            }
            #pragma omp parallel for
            for (size_t i = 0; i < population_.size(); ++i) population_[i].evaluate(prob_, curr_dom, curr_bnd, current_gen_, &point_weights_);
        }

        // Re-sort and update hall of fame based on current batch evaluation
        fast_non_dominated_sort(population_, use_boundary_objective_);
        update_hall_of_fame();

        if (has_best_ever_) {
            best_ever_.evaluate(prob_, curr_dom, curr_bnd, current_gen_, &point_weights_);
            double train_err = Config::GENERAL_MODE ? best_ever_.mse_domain : (best_ever_.mse_domain + best_ever_.mse_boundary);
            double threshold = Config::STOP_THRESHOLD;
            
            // Clona (no mueve) al campeon: best_ever_ es un miembro de la
            // instancia (usado despues por polish_constants/solution_mse en
            // main.cpp via pi.champion()) — moverlo lo dejaba con el arbol
            // nulo mientras has_best_ever_ seguia en true, asi que main.cpp
            // creia tener un campeon valido y todo colapsaba a sentinelas
            // (1e18/-1). No se manifestaba antes porque el umbral (1e-12)
            // nunca se alcanzaba en la practica.
            auto clone_champion = [&]() {
                PIIndividual c;
                c.mse_domain = best_ever_.mse_domain; c.mse_boundary = best_ever_.mse_boundary;
                c.rank = best_ever_.rank; c.crowding = best_ever_.crowding; c.tree_size = best_ever_.tree_size;
                c.is_feasible = best_ever_.is_feasible; c.constraint_violation = best_ever_.constraint_violation;
                if (best_ever_.tree) c.tree = best_ever_.tree->clone(); c.raw = best_ever_.raw;
                return c;
            };
            auto clone_ind_local = [](const PIIndividual& ind) {
                PIIndividual c;
                c.mse_domain = ind.mse_domain; c.mse_boundary = ind.mse_boundary;
                c.rank = ind.rank; c.crowding = ind.crowding; c.tree_size = ind.tree_size;
                c.is_feasible = ind.is_feasible; c.constraint_violation = ind.constraint_violation;
                if (ind.tree) c.tree = ind.tree->clone(); c.raw = ind.raw;
                return c;
            };
            if (train_err < threshold) {
                // En modo general, aceptamos la solución por su residuo dinámico
                if (Config::GENERAL_MODE) {
                    std::cout << "[INFO] Early stopping en gen=" << g << " (Solucion General Descubierta, Res < " << threshold << ")" << std::endl;
                    unpolished_snapshot_.clear();
                    for (auto& ind : population_) unpolished_snapshot_.push_back(clone_ind_local(ind));
                    std::vector<PIIndividual> final_pop; final_pop.push_back(clone_champion());
                    for (auto& ind : population_) final_pop.push_back(std::move(ind));
                    fast_non_dominated_sort(final_pop, use_boundary_objective_);
                    return final_pop;
                }

                // Confirmacion en la grilla fija de validacion, con informacion
                // de la EDP solamente (antes: error contra la referencia).
                double val_err = best_ever_.validation_score(prob_, val_dom_pts_, val_bnd_pts_);
                if (val_err < threshold * 10.0) {
                    std::cout << "[INFO] Early stopping en gen=" << g << " (residuo + frontera de validacion < " << threshold * 10.0 << ")" << std::endl;
                    unpolished_snapshot_.clear();
                    for (auto& ind : population_) unpolished_snapshot_.push_back(clone_ind_local(ind));
                    std::vector<PIIndividual> final_pop; final_pop.push_back(clone_champion());
                    for (auto& ind : population_) final_pop.push_back(std::move(ind));
                    fast_non_dominated_sort(final_pop, use_boundary_objective_);
                    return final_pop;
                }
            }
        }

        auto clone_ind = [](const PIIndividual& ind) {
            PIIndividual c;
            c.mse_domain = ind.mse_domain; c.mse_boundary = ind.mse_boundary;
            c.rank = ind.rank; c.crowding = ind.crowding; c.tree_size = ind.tree_size;
            c.is_feasible = ind.is_feasible; c.constraint_violation = ind.constraint_violation;
            if (ind.tree) c.tree = ind.tree->clone(); c.raw = ind.raw;
            return c;
        };

        // ─── NSGA-II: torneo binario/4-ario + combinar padres+hijos + orden
        // no-dominado + crowding distance (con bono de diversidad estructural
        // por tipo de nodo raíz en calculate_crowding_distance).
        std::vector<PIIndividual> offsprings(pop_size);
        for (int i = 0; i < pop_size; ++i) {
            int r1 = tournament_select(population_, gen_);
            int r2 = tournament_select(population_, gen_);
            offsprings[i] = make_offspring(population_[r1], population_[r2]);
            if (offsprings[i].tree) offsprings[i].tree = offsprings[i].tree->simplify();
        }

        #pragma omp parallel for schedule(dynamic) num_threads(Config::CORES > 0 ? Config::CORES : omp_get_max_threads())
        for (int i = 0; i < pop_size; ++i) {
            offsprings[i].evaluate(prob_, curr_dom, curr_bnd, current_gen_, &point_weights_);
        }

        std::vector<PIIndividual> combined;
        combined.reserve(pop_size * 2);
        for (const auto& ind : population_) combined.push_back(clone_ind(ind));
        for (auto& ind : offsprings) combined.push_back(std::move(ind));

        population_ = nsga2_select_next(std::move(combined), pop_size, use_boundary_objective_);
        fast_non_dominated_sort(population_, use_boundary_objective_);

        // Archivo historico (ver population_archive() en el header): solo 4
        // numeros por individuo, sin clonar el arbol, asi que se puede
        // acumular en CADA generacion (no solo la ultima) a costo
        // despreciable — a diferencia de unpolished_snapshot_ (que si clona
        // arboles completos y por eso se limita a la ultima generacion).
        population_archive_.reserve(population_archive_.size() + population_.size());
        for (auto& ind : population_) {
            population_archive_.push_back({g, ind.mse_domain, ind.mse_boundary, ind.tree_size, ind.rank});
        }

        // Snapshot del frente SIN pulir de la ultima generacion (ver
        // unpolished_final_population() en el header): el pulido de abajo
        // corre sobre TODA la poblacion cada generacion y converge a los
        // individuos de un mismo tree_size al mismo optimo local, aplanando
        // la diversidad visual del frente. Se guarda solo en la ultima
        // generacion para no pagar el costo de clonar arboles en cada paso.
        if (g == max_gen - 1) {
            unpolished_snapshot_.clear();
            unpolished_snapshot_.reserve(population_.size());
            for (auto& ind : population_) unpolished_snapshot_.push_back(clone_ind(ind));
        }

        // Pulido de constantes para TODA la población, no sólo rank==1. Antes una
        // estructura genuinamente buena con constantes sin afinar podía perder la
        // comparación de Pareto contra estructuras peores pero con mejor suerte de
        // inicialización, y quedaba descartada antes de tener oportunidad de
        // demostrar su valor. Prioridad no es velocidad sino calidad del resultado
        // final; ya tenemos margen de sobra desde el build en Release.
        unsigned int g_seed = gen_();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)population_.size(); ++i) {
            std::mt19937 local_gen(g_seed + i);
            gradient_descent_constants(population_[i], (g % 10 == 0 ? 30 : 10));
        }
        {
            double b_dom = 1e18, b_bnd = 1e18;
            for (auto& ind : population_) if (ind.rank == 1) { b_dom = std::min(b_dom, ind.mse_domain); b_bnd = std::min(b_bnd, ind.mse_boundary); }
            // Puntaje de validacion del campeon (fisica, grilla fija): comparable
            // generacion a generacion, a diferencia de b_dom/b_bnd (mini-batch).
            double val_score = has_best_ever_ ? best_ever_.validation_score(prob_, val_dom_pts_, val_bnd_pts_) : 1e18;
            // Error contra la referencia: SOLO REGISTRO para analizar la
            // convergencia despues. Se escribe en history_ y nada lo lee
            // dentro de la busqueda.
            double ref_err = has_best_ever_ ? best_ever_.reference_error(prob_, val_dom_pts_) : 1e18;
            history_.push_back({g, b_dom, b_bnd, b_dom + b_bnd, val_score, ref_err});
            // Antes imprimia cada 25 generaciones: a pop=500 (pulido de
            // constantes de TODA la poblacion cada gen, ver comentario mas
            // arriba) eso son minutos de silencio entre lineas -- se ve
            // identico a un cuelgue aunque la CPU siga al 100% trabajando
            // (confirmado: sin este cambio, un corrida real se reporto como
            // "bugueada" estando solo lenta). Cada 5 generaciones + flush
            // explicito (antes el buffer de cout no se vaciaba solo al
            // redirigir a archivo/tee, agravando la apariencia de cuelgue).
            if (g % 5 == 0) {
                int n_infeasible = 0, front1 = 0;
                for (auto& ind : population_) { if (!ind.is_feasible) n_infeasible++; if (ind.rank == 1) front1++; }
                std::cout << "  [PI/" << prob_.name() << "] gen=" << g
                          << "  best_dom(batch)=" << std::scientific << b_dom
                          << "  best_bnd(batch)=" << b_bnd
                          << "  val_score(hof)=" << val_score
                          << "  ref_err(solo registro)=" << ref_err << std::defaultfloat
                          << "  infeasible=" << n_infeasible << "/" << population_.size()
                          << "  front1=" << front1 << std::endl;
            }
        }
    }
    if (has_best_ever_) {
        PIIndividual champ;
        champ.mse_domain = best_ever_.mse_domain; champ.mse_boundary = best_ever_.mse_boundary;
        champ.rank = 1; champ.is_feasible = true;
        if (best_ever_.tree) champ.tree = best_ever_.tree->clone(); champ.raw = best_ever_.raw;
        population_.push_back(std::move(champ));
        fast_non_dominated_sort(population_, use_boundary_objective_);
    }
    return std::move(population_);
}

// Descenso por gradiente sobre las constantes del arbol. El gradiente es el de
// la perdida fisica de entrenamiento (residuo + frontera medida) por
// diferencias centradas (constant_gradient), valido para cualquier arbol,
// incluidos los complejos; mismo metodo para todas las EDPs.
void PISolver::gradient_descent_constants(PIIndividual& ind, int iterations,
                                           const std::vector<Point>* dom_arg,
                                           const std::vector<Point>* bnd_arg) {
    if (!ind.tree) return;
    std::vector<Complex*> ercs; ind.tree->collect_ercs(ercs);
    if (ercs.empty()) return;

    // Por defecto opera sobre los puntos de entrenamiento (uso in-loop, pulido
    // de toda la poblacion cada generacion). polish_constants() del campeon
    // final pasa las grillas de validacion explicitamente.
    const std::vector<Point>& dom = dom_arg ? *dom_arg : dom_pts_;
    const std::vector<Point>& bnd = bnd_arg ? *bnd_arg : bnd_pts_;

    double lr = 0.05;
    for (int iter = 0; iter < iterations; ++iter) {
        ind.evaluate(prob_, dom, bnd, current_gen_);
        if (!ind.is_feasible) return;
        double current_loss = ind.mse_domain + ind.mse_boundary;

        std::vector<double> grads = constant_gradient(prob_, ind.tree.get(), dom, bnd, 1e-6, ind.raw);

        // Recorte del gradiente en EDPs singulares
        if (Config::IS_SINGULAR) {
            for (auto& g : grads) g = std::clamp(g, -50.0, 50.0);
        }

        for (size_t i = 0; i < ercs.size(); ++i) *ercs[i] -= Complex(lr * grads[i], 0);

        ind.evaluate(prob_, dom, bnd, current_gen_);
        if (!ind.is_feasible || (ind.mse_domain + ind.mse_boundary >= current_loss)) {
            // Revertir y reducir el paso
            for (size_t i = 0; i < ercs.size(); ++i) *ercs[i] += Complex(lr * grads[i], 0);
            lr *= 0.5;
            ind.evaluate(prob_, dom, bnd, current_gen_);
        } else {
            lr *= 1.1;
        }
        if (lr < 1e-7) break;
    }
}

void PISolver::hill_climb_constants(PIIndividual& ind, int iterations, std::mt19937& thread_gen) {
    if (!ind.tree) return;
    std::vector<Complex*> ercs; ind.tree->collect_ercs(ercs);
    if (ercs.empty()) return;

    ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
    double best_err = ind.mse_domain + ind.mse_boundary;
    std::vector<Complex> best_vals(ercs.size());
    for(size_t i=0; i<ercs.size(); ++i) best_vals[i] = *ercs[i];
    
    for (int i = 0; i < iterations; ++i) {
        ind.tree->mutate_erc(thread_gen, Config::ERC_SIGMA);
        ind.evaluate(prob_, dom_pts_, bnd_pts_, current_gen_);
        double err = ind.mse_domain + ind.mse_boundary;
        if (ind.is_feasible && err < best_err) {
            best_err = err;
            for(size_t j=0; j<ercs.size(); ++j) best_vals[j] = *ercs[j];
        } else {
            for(size_t j=0; j<ercs.size(); ++j) *ercs[j] = best_vals[j]; // revertir
        }
    }
}

void PISolver::polish_constants(PIIndividual& ind) {
    if (!ind.tree) return;
    std::vector<Complex*> ercs; ind.tree->collect_ercs(ercs);
    if (ercs.empty()) return;

    if (Config::IS_SINGULAR) {
        std::cout << "  [Optimizer] Polishing " << ercs.size() << " constants via Gradient Descent (Clipped Mode)...\n";
    } else {
        std::cout << "  [Optimizer] Polishing " << ercs.size() << " constants via Adaptive Gradient Descent...\n";
    }
    // Se pule contra la grilla FIJA de validacion (val_dom_pts_/val_bnd_pts_),
    // no el mini-batch de entrenamiento — este metodo se llama una sola vez
    // sobre el campeon del Hall of Fame, ya elegido por su val_mse; pulir
    // contra el mini-batch (que puede ser un subconjunto mas facil o mas
    // dificil que el promedio) puede sobreajustar y empeorar el resultado
    // real reportado, deshaciendo la proteccion que da el Hall of Fame.
    gradient_descent_constants(ind, 200, &val_dom_pts_, &val_bnd_pts_);
    ind.evaluate(prob_, val_dom_pts_, val_bnd_pts_, current_gen_);
}

std::vector<PIIndividual> PISolver::pareto_front() const {
    std::vector<PIIndividual> front;
    for (auto& ind : population_) {
        if (ind.rank == 1 && ind.is_feasible && ind.is_physically_complete(prob_)) {
            PIIndividual copy; copy.mse_domain = ind.mse_domain; copy.mse_boundary = ind.mse_boundary;
            copy.rank = ind.rank; copy.tree_size = ind.tree_size;
            if (ind.tree) copy.tree = ind.tree->clone(); copy.raw = ind.raw;
            front.push_back(std::move(copy));
        }
    }
    return front;
}

bool PIIndividual::is_physically_complete(const PDEProblem& prob) const {
    if (!tree) return false;
    if (!tree->uses_variable(NodeType::VAR_X)) return false;
    if (prob.dim >= 2 && !tree->uses_variable(NodeType::VAR_Y)) return false;
    if (prob.is_unsteady && !tree->uses_variable(NodeType::VAR_T)) return false;
    return true;
}
