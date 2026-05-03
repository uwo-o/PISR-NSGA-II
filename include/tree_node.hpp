#pragma once
// =============================================================================
// tree_node.hpp  —  Árbol de expresión simbólica con AD exacto
//   Usado por AMBOS métodos (PI propuesto y Koza).
//   El método propuesto usa derivadas simbólicas (AD).
//   El método Koza usa diferencias finitas externas.
// =============================================================================

#include "common.hpp"
#include <memory>
#include <random>
#include <ostream>
#include <vector>
#include <functional>

// ─── Tipos de nodo ────────────────────────────────────────────────────────────
enum class NodeType {
    VAR_X, VAR_Y, ERC,        // terminales
    ADD, SUB, MUL,             // operadores binarios
    SIN, COS, EXP              // operadores unarios
};

inline bool is_binary(NodeType t) { return t==NodeType::ADD || t==NodeType::SUB || t==NodeType::MUL; }
inline bool is_unary(NodeType t)  { return t==NodeType::SIN || t==NodeType::COS || t==NodeType::EXP; }
inline bool is_terminal(NodeType t){ return t==NodeType::VAR_X || t==NodeType::VAR_Y || t==NodeType::ERC; }

// ─── Nodo del árbol ───────────────────────────────────────────────────────────
struct Node {
    NodeType type;
    double   erc_val = 0.0;                         // solo para ERC
    std::vector<std::shared_ptr<Node>> children;    // 0,1 ó 2 hijos

    // ── Evaluación con Diferenciación Automática (método propuesto) ───────────
    AD ad_eval(double x, double y) const;

    // ── Evaluación escalar simple (usada por FD de Koza) ─────────────────────
    double eval(double x, double y) const;

    // ── Mutación paramétrica Gaussiana en nodos ERC ───────────────────────────
    void mutate_erc(std::mt19937& gen, double sigma = Config::ERC_SIGMA);

    // ── Copia profunda ────────────────────────────────────────────────────────
    std::shared_ptr<Node> clone() const;

    // ── Tamaño del subárbol ───────────────────────────────────────────────────
    int size() const;

    // ── Impresión ─────────────────────────────────────────────────────────────
    void print(std::ostream& os) const;

    // ── Acceso al k-ésimo nodo (DFS, in-order); counter decrementado ──────────
    std::shared_ptr<Node>& get_node_ref(int& counter);
};

using NodePtr = std::shared_ptr<Node>;

// ─── Constructores de nodos ───────────────────────────────────────────────────
NodePtr make_var(char v);                    // 'x' o 'y'
NodePtr make_erc(double val);
NodePtr make_binary(NodeType op, NodePtr l, NodePtr r);
NodePtr make_unary(NodeType op, NodePtr child);

// ─── Generación aleatoria de árbol (ramped half-and-half) ────────────────────
NodePtr random_tree(int max_depth, std::mt19937& gen, bool force_terminal = false);

// ─── Operadores evolutivos ────────────────────────────────────────────────────
// Crossover de subárbol: devuelve dos hijos clonados con subárboles intercambiados
std::pair<NodePtr, NodePtr> tree_crossover(const NodePtr& p1, const NodePtr& p2, std::mt19937& gen);

// Mutación: reemplaza un subárbol aleatorio con uno nuevo
NodePtr tree_mutate(const NodePtr& tree, std::mt19937& gen);

// ─── Laplaciano via diferencias finitas (para método Koza) ───────────────────
double fd_laplacian(const NodePtr& tree, double x, double y, double h = 1e-5);
