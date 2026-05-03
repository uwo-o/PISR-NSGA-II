#pragma once
// =============================================================================
// pi_solver.hpp  —  Método Propuesto: PI-NSGA-II con árboles simbólicos
//   Genotipo = árbol de expresión
//   Derivadas = Diferenciación Automática exacta (sin FD)
//   Constantes = ERC con mutación Gaussiana paramétrica
// =============================================================================

#include "common.hpp"
#include "tree_node.hpp"
#include "pde_problems.hpp"
#include <vector>
#include <random>

// ─── Individuo PI ─────────────────────────────────────────────────────────────
struct PIIndividual : public Individual {
    NodePtr tree;

    PIIndividual() = default;
    PIIndividual(const PIIndividual& other) : Individual(other) {
        if (other.tree) tree = other.tree->clone();
    }
    PIIndividual(PIIndividual&&) noexcept = default;
    PIIndividual& operator=(const PIIndividual& other) {
        if (this != &other) {
            Individual::operator=(other);
            tree = other.tree ? other.tree->clone() : nullptr;
        }
        return *this;
    }
    PIIndividual& operator=(PIIndividual&&) noexcept = default;

    // Evalúa MSE dominio (AD simbólico) + MSE frontera
    void evaluate(const PDEProblem& prob,
                  const std::vector<Point>& dom,
                  const std::vector<Point>& bnd);
};

// ─── Clase Algoritmo PI-NSGA-II ───────────────────────────────────────────────
class PISolver {
public:
    explicit PISolver(const PDEProblem& prob, unsigned seed = 42);

    std::vector<PIIndividual> run(int pop_size = Config::POP_SIZE,
                                  int max_gen  = Config::MAX_GEN);

    std::vector<PIIndividual> pareto_front() const;

private:
    PDEProblem           prob_;
    std::mt19937         gen_;
    std::vector<Point>   dom_pts_;
    std::vector<Point>   bnd_pts_;
    std::vector<PIIndividual> population_;

    PIIndividual random_individual();
    PIIndividual make_offspring(const PIIndividual& a, const PIIndividual& b);
};
