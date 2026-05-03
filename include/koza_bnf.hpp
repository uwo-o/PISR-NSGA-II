#pragma once
// =============================================================================
// koza_bnf.hpp  —  Método Koza con codificación BNF (Gramática Evolutiva)
//   Genotipo: vector de enteros (codones)
//   Fenotipo: árbol de expresión (NodePtr) construido por la gramática BNF
//   Derivadas: diferencias finitas (NO simbólicas)
// =============================================================================

#include "common.hpp"
#include "tree_node.hpp"
#include "pde_problems.hpp"
#include <vector>
#include <random>

// ─── Individuo Koza ───────────────────────────────────────────────────────────
struct KozaIndividual : public Individual {
    std::vector<int> codons;    // genotipo: secuencia de enteros
    NodePtr          tree;      // fenotipo: árbol generado por BNF

    // Mapea codones → árbol usando la gramática BNF
    void decode();

    // Evalúa MSE dominio + MSE frontera con diferencias finitas
    void evaluate(const PDEProblem& prob,
                  const std::vector<Point>& dom,
                  const std::vector<Point>& bnd);
};

// ─── Clase Algoritmo Koza+NSGA-II ─────────────────────────────────────────────
class KozaSolver {
public:
    explicit KozaSolver(const PDEProblem& prob, unsigned seed = 42);

    // Corre MAX_GEN generaciones de NSGA-II con representación BNF
    // Devuelve la población final ordenada por rango
    std::vector<KozaIndividual> run(int pop_size  = Config::POP_SIZE,
                                    int max_gen   = Config::MAX_GEN);

    // Retorna el frente de Pareto (rank == 1) de la última ejecución
    std::vector<KozaIndividual> pareto_front() const;

private:
    PDEProblem             prob_;
    std::mt19937           gen_;
    std::vector<Point>     dom_pts_;
    std::vector<Point>     bnd_pts_;
    std::vector<KozaIndividual> population_;

    KozaIndividual random_individual();
    KozaIndividual crossover(const KozaIndividual& a, const KozaIndividual& b);
    void           mutate(KozaIndividual& ind);
};

// ─── Función libre: gramática BNF → árbol ─────────────────────────────────────
// Construye un árbol de expresión siguiendo la gramática BNF de Koza.
// Los codones se consumen circularmente (wrapping).
NodePtr bnf_to_tree(const std::vector<int>& codons,
                    int& idx, int depth, int max_depth);
