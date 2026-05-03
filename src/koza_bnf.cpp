// =============================================================================
// koza_bnf.cpp  —  Método Koza con gramática BNF y diferencias finitas
// =============================================================================

#include "koza_bnf.hpp"
#include "nsga2.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>

// ─── Gramática BNF ────────────────────────────────────────────────────────────
//  <expr>  ::= <expr> <op> <expr>  | <unary>(<expr>) | <var> | <erc>
//  <op>    ::= + | - | *
//  <unary> ::= sin | cos | exp
//  <var>   ::= x | y
//  <erc>   ::= 1..9
// Se usa "wrapping" circular cuando se agotan los codones.
NodePtr bnf_to_tree(const std::vector<int>& codons,
                    int& idx, int depth, int max_depth)
{
    if (codons.empty()) return make_erc(1.0);

    auto next_codon = [&]() -> int {
        int c = codons[idx % codons.size()];
        idx++;
        return c;
    };

    if (depth >= max_depth) {
        // Forzar terminal
        int c = next_codon();
        if (c % 3 == 0) return make_var('x');
        if (c % 3 == 1) return make_var('y');
        return make_erc((double)((next_codon() % 9) + 1));
    }

    int rule = next_codon() % 4;

    if (rule == 0) {
        // <expr> <op> <expr>
        NodePtr L = bnf_to_tree(codons, idx, depth + 1, max_depth);
        int op_code = next_codon() % 3;
        NodePtr R = bnf_to_tree(codons, idx, depth + 1, max_depth);
        NodeType op = (op_code == 0) ? NodeType::ADD
                    : (op_code == 1) ? NodeType::SUB
                    :                  NodeType::MUL;
        return make_binary(op, L, R);
    }
    else if (rule == 1) {
        // <unary>(<expr>)
        int u_code = next_codon() % 3;
        NodeType op = (u_code == 0) ? NodeType::SIN
                    : (u_code == 1) ? NodeType::COS
                    :                 NodeType::EXP;
        NodePtr child = bnf_to_tree(codons, idx, depth + 1, max_depth);
        return make_unary(op, child);
    }
    else if (rule == 2) {
        // <var>
        return (next_codon() % 2 == 0) ? make_var('x') : make_var('y');
    }
    else {
        // <erc>  — constante entera rígida (característica de Koza)
        return make_erc((double)((next_codon() % 9) + 1));
    }
}

// ─── KozaIndividual::decode ───────────────────────────────────────────────────
void KozaIndividual::decode() {
    int idx = 0;
    tree = bnf_to_tree(codons, idx, 0, Config::MAX_TREE_DEPTH);
}

// ─── KozaIndividual::evaluate ─────────────────────────────────────────────────
void KozaIndividual::evaluate(const PDEProblem& prob,
                              const std::vector<Point>& dom,
                              const std::vector<Point>& bnd)
{
    if (!tree) decode();

    // MSE dominio — Laplaciano por diferencias finitas
    double sum_dom = 0.0;
    for (auto& p : dom) {
        double lap = fd_laplacian(tree, p.x, p.y);
        double u   = tree->eval(p.x, p.y);
        double res = lap + prob.k2 * u - prob.source(p.x, p.y);
        // Sanitizar NaN/Inf
        if (!std::isfinite(res)) { mse_domain = 1e10; mse_boundary = 1e10; return; }
        sum_dom += res * res;
    }
    mse_domain = sum_dom / (double)dom.size();

    // MSE frontera — diferencia con condición de Dirichlet
    double sum_bnd = 0.0;
    for (auto& p : bnd) {
        double u    = tree->eval(p.x, p.y);
        double diff = u - prob.bc(p.x, p.y);
        if (!std::isfinite(diff)) { mse_boundary = 1e10; return; }
        sum_bnd += diff * diff;
    }
    mse_boundary = sum_bnd / (double)bnd.size();
}

// ─── KozaSolver ───────────────────────────────────────────────────────────────
KozaSolver::KozaSolver(const PDEProblem& prob, unsigned seed)
    : prob_(prob), gen_(seed)
{
    dom_pts_ = prob_.domain_points(Config::N_DOMAIN);
    bnd_pts_ = prob_.boundary_points(Config::N_BOUNDARY);
}

KozaIndividual KozaSolver::random_individual() {
    KozaIndividual ind;
    ind.codons.resize(Config::CODON_LENGTH);
    std::uniform_int_distribution<int> d(0, 255);
    for (auto& c : ind.codons) c = d(gen_);
    ind.decode();
    ind.evaluate(prob_, dom_pts_, bnd_pts_);
    return ind;
}

KozaIndividual KozaSolver::crossover(const KozaIndividual& a, const KozaIndividual& b) {
    KozaIndividual child;
    child.codons.resize(Config::CODON_LENGTH);
    // Cruce en un punto (codon-level)
    std::uniform_int_distribution<int> cut(1, Config::CODON_LENGTH - 1);
    int c = cut(gen_);
    for (int i = 0; i < Config::CODON_LENGTH; ++i)
        child.codons[i] = (i < c) ? a.codons[i] : b.codons[i];
    child.decode();
    return child;
}

void KozaSolver::mutate(KozaIndividual& ind) {
    std::uniform_int_distribution<int> pos(0, Config::CODON_LENGTH - 1);
    std::uniform_int_distribution<int> val(0, 255);
    // Mutar ~10% de los codones
    int n_mut = std::max(1, Config::CODON_LENGTH / 10);
    for (int i = 0; i < n_mut; ++i)
        ind.codons[pos(gen_)] = val(gen_);
    ind.decode();
}

std::vector<KozaIndividual> KozaSolver::run(int pop_size, int max_gen) {
    // Inicialización
    population_.clear();
    population_.reserve(pop_size);
    for (int i = 0; i < pop_size; ++i)
        population_.push_back(random_individual());

    // Bucle generacional NSGA-II
    for (int gen = 0; gen < max_gen; ++gen) {
        // Generar hijos
        std::vector<KozaIndividual> offspring;
        offspring.reserve(pop_size);
        std::uniform_real_distribution<double> prob(0.0, 1.0);

        while ((int)offspring.size() < pop_size) {
            int p1 = tournament_select(population_, gen_);
            int p2 = tournament_select(population_, gen_);
            KozaIndividual child;
            if (prob(gen_) < Config::CROSSOVER_PROB)
                child = crossover(population_[p1], population_[p2]);
            else
                child = population_[p1]; // copia
            if (prob(gen_) < Config::MUTATION_PROB)
                mutate(child);
            child.evaluate(prob_, dom_pts_, bnd_pts_);
            offspring.push_back(std::move(child));
        }

        // Combinar y seleccionar
        std::vector<KozaIndividual> combined;
        combined.reserve(pop_size * 2);
        for (auto& x : population_) combined.push_back(x);
        for (auto& x : offspring)   combined.push_back(std::move(x));
        population_ = nsga2_select_next(std::move(combined), pop_size);

        // Progreso (cada 25 generaciones)
        if (gen % 25 == 0) {
            auto& best = population_.front();
            std::cout << "  [Koza/" << prob_.name() << "] gen=" << gen
                      << "  dom=" << best.mse_domain
                      << "  bnd=" << best.mse_boundary << "\n";
        }
    }
    return population_;
}

std::vector<KozaIndividual> KozaSolver::pareto_front() const {
    std::vector<KozaIndividual> front;
    for (auto& ind : population_)
        if (ind.rank == 1) front.push_back(ind);
    return front;
}
