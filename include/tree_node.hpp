#pragma once
// =============================================================================
// tree_node.hpp  —  Árbol de expresión simbólica con AD exacto (Polimórfico)
// =============================================================================

#include "common.hpp"
#include "dimensions.hpp"
#include "pde_problems.hpp"
#include <memory>
#include <random>
#include <ostream>
#include <optional>

inline bool is_binary(NodeType t) {
    return t == NodeType::ADD || t == NodeType::SUB || t == NodeType::MUL || t == NodeType::DIV || 
           t == NodeType::LEGENDRE || t == NodeType::HERMITE || t == NodeType::CHEBYSHEV || t == NodeType::LAGUERRE;
}
inline bool is_unary(NodeType t) {
    return t == NodeType::SIN  || t == NodeType::COS  ||
           t == NodeType::SINH || t == NodeType::COSH ||
           t == NodeType::EXP  || t == NodeType::SQR  ||
           t == NodeType::BESSEL_J || t == NodeType::GAMMA || t == NodeType::GAUSSIAN;
}
inline bool is_terminal(NodeType t) {
    return t == NodeType::VAR_X || t == NodeType::VAR_Y || t == NodeType::VAR_T || t == NodeType::ERC || t == NodeType::CONST_I;
}

class PDEProblem;
class Node;
using NodePtr = std::unique_ptr<Node>;

class Node {
public:
    virtual ~Node() = default;

    virtual NodeType get_type() const = 0;
    virtual AD ad_eval(double x, double y, int dim = 2) const = 0;
    virtual AD ad_eval_t(double x, double y, double t, int dim = 2) const = 0;
    virtual Complex eval(double x, double y) const = 0;
    virtual Complex eval_t(double x, double y, double t) const = 0;
    
    // Análisis Dimensional
    virtual std::optional<Dimension> get_dimension(const PDEProblem& prob) const = 0;
    virtual bool contains_variables() const = 0;
    
    bool is_consistent(const PDEProblem& prob) const {
        return get_dimension(prob).has_value();
    }

    virtual NodePtr clone() const = 0;
    virtual int count_nodes() const = 0;
    
    virtual void mutate_erc(std::mt19937& gen, double sigma = Config::ERC_SIGMA) {}

    virtual void print(std::ostream& os) const = 0;
    virtual void print_latex(std::ostream& os) const = 0;
    virtual NodePtr simplify() const = 0;
    virtual NodePtr prune_recursive(const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd, double original_mse, double tolerance) = 0;
    
    virtual void collect_ercs(std::vector<Complex*>& ptrs) = 0;
    virtual bool uses_variable(NodeType var_type) const = 0;
};

// ─── Clases derivadas ─────────────────────────────────────────────────────────

class TerminalNode final : public Node {
public:
    NodeType type;
    Complex erc_val;

    TerminalNode(NodeType t, Complex val = 0.0) : type(t), erc_val(val) {}
    NodeType get_type() const override { return type; }
    AD ad_eval(double x, double y, int dim = 2) const override;
    AD ad_eval_t(double x, double y, double t, int dim = 2) const override;
    Complex eval(double x, double y) const override;
    Complex eval_t(double x, double y, double t) const override;
    std::optional<Dimension> get_dimension(const PDEProblem& prob) const override;
    bool contains_variables() const override;
    NodePtr clone() const override { return std::make_unique<TerminalNode>(type, erc_val); }
    int count_nodes() const override { return 1; }
    void mutate_erc(std::mt19937& gen, double sigma = Config::ERC_SIGMA) override;
    void print(std::ostream& os) const override;
    void print_latex(std::ostream& os) const override;
    NodePtr simplify() const override;
    NodePtr prune_recursive(const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd, double original_mse, double tolerance) override;
    void collect_ercs(std::vector<Complex*>& ptrs) override;
    bool uses_variable(NodeType var_type) const override;
};

class UnaryNode final : public Node {
public:
    NodeType type;
    NodePtr child;

    UnaryNode(NodeType t, NodePtr c) : type(t), child(std::move(c)) {}
    NodeType get_type() const override { return type; }
    AD ad_eval(double x, double y, int dim = 2) const override;
    AD ad_eval_t(double x, double y, double t, int dim = 2) const override;
    Complex eval(double x, double y) const override;
    Complex eval_t(double x, double y, double t) const override;
    std::optional<Dimension> get_dimension(const PDEProblem& prob) const override;
    bool contains_variables() const override;
    NodePtr clone() const override { return std::make_unique<UnaryNode>(type, child->clone()); }
    int count_nodes() const override { return 1 + child->count_nodes(); }
    void mutate_erc(std::mt19937& gen, double sigma = Config::ERC_SIGMA) override { child->mutate_erc(gen, sigma); }
    void print(std::ostream& os) const override;
    void print_latex(std::ostream& os) const override;
    NodePtr simplify() const override;
    NodePtr prune_recursive(const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd, double original_mse, double tolerance) override;
    void collect_ercs(std::vector<Complex*>& ptrs) override;
    bool uses_variable(NodeType var_type) const override;
};

class BinaryNode final : public Node {
public:
    NodeType type;
    NodePtr left;
    NodePtr right;

    BinaryNode(NodeType t, NodePtr l, NodePtr r) : type(t), left(std::move(l)), right(std::move(r)) {}
    NodeType get_type() const override { return type; }
    AD ad_eval(double x, double y, int dim = 2) const override;
    AD ad_eval_t(double x, double y, double t, int dim = 2) const override;
    Complex eval(double x, double y) const override;
    Complex eval_t(double x, double y, double t) const override;
    std::optional<Dimension> get_dimension(const PDEProblem& prob) const override;
    bool contains_variables() const override;
    NodePtr clone() const override { return std::make_unique<BinaryNode>(type, left->clone(), right->clone()); }
    int count_nodes() const override { return 1 + left->count_nodes() + right->count_nodes(); }
    void mutate_erc(std::mt19937& gen, double sigma = Config::ERC_SIGMA) override {
        left->mutate_erc(gen, sigma);
        right->mutate_erc(gen, sigma);
    }
    void print(std::ostream& os) const override;
    void print_latex(std::ostream& os) const override;
    NodePtr simplify() const override;
    NodePtr prune_recursive(const PDEProblem& prob, const std::vector<Point>& dom, const std::vector<Point>& bnd, double original_mse, double tolerance) override;
    void collect_ercs(std::vector<Complex*>& ptrs) override;
    bool uses_variable(NodeType var_type) const override;
};

// ─── Constructores de nodos ───────────────────────────────────────────────────
NodePtr make_var(char v);
NodePtr make_erc(Complex val);
NodePtr make_const_i();
NodePtr make_binary(NodeType op, NodePtr l, NodePtr r);
NodePtr make_unary(NodeType op, NodePtr child);

// ─── Generación aleatoria de árbol ────────────────────────────────────────────
NodePtr random_tree(int max_depth, std::mt19937& gen, bool force_terminal = false);
NodePtr random_tree_special(int max_depth, std::mt19937& gen, const PDEProblem& prob);
NodePtr get_exact_solution_tree(const PDEProblem& prob);
NodePtr remove_nested_polynomials(NodePtr node, bool inside_poly = false);

// ─── Operadores evolutivos ────────────────────────────────────────────────────
std::pair<NodePtr, NodePtr> tree_crossover(const NodePtr& p1, const NodePtr& p2, std::mt19937& gen);
NodePtr tree_mutate(const NodePtr& tree, std::mt19937& gen, const PDEProblem& prob);
void replace_node_at(NodePtr& current, int& target_idx, NodePtr& replacement);

// ─── Laplaciano via diferencias finitas (para método Koza) ───────────────────
Complex fd_laplacian(const NodePtr& tree, double x, double y, int dim, double h = 1e-5);
