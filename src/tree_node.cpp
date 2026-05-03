// =============================================================================
// tree_node.cpp  —  Implementación del árbol de expresión con AD exacto
// =============================================================================

#include "tree_node.hpp"
#include <cmath>
#include <stdexcept>
#include <sstream>

// ─── ad_eval ──────────────────────────────────────────────────────────────────
AD Node::ad_eval(double x, double y) const {
    AD r;
    switch (type) {
        case NodeType::VAR_X:
            r.v = x; r.dx = 1.0; r.dy = 0.0; r.dxx = 0.0; r.dyy = 0.0;
            break;
        case NodeType::VAR_Y:
            r.v = y; r.dx = 0.0; r.dy = 1.0; r.dxx = 0.0; r.dyy = 0.0;
            break;
        case NodeType::ERC:
            r.v = erc_val;
            break;
        case NodeType::ADD: {
            AD L = children[0]->ad_eval(x, y);
            AD R = children[1]->ad_eval(x, y);
            r.v   = L.v   + R.v;
            r.dx  = L.dx  + R.dx;
            r.dy  = L.dy  + R.dy;
            r.dxx = L.dxx + R.dxx;
            r.dyy = L.dyy + R.dyy;
            break;
        }
        case NodeType::SUB: {
            AD L = children[0]->ad_eval(x, y);
            AD R = children[1]->ad_eval(x, y);
            r.v   = L.v   - R.v;
            r.dx  = L.dx  - R.dx;
            r.dy  = L.dy  - R.dy;
            r.dxx = L.dxx - R.dxx;
            r.dyy = L.dyy - R.dyy;
            break;
        }
        case NodeType::MUL: {
            AD L = children[0]->ad_eval(x, y);
            AD R = children[1]->ad_eval(x, y);
            r.v   = L.v * R.v;
            // Regla del producto
            r.dx  = L.dx*R.v  + L.v*R.dx;
            r.dy  = L.dy*R.v  + L.v*R.dy;
            // Segunda derivada: (uv)'' = u''v + 2u'v' + uv''
            r.dxx = L.dxx*R.v + 2.0*L.dx*R.dx + L.v*R.dxx;
            r.dyy = L.dyy*R.v + 2.0*L.dy*R.dy + L.v*R.dyy;
            break;
        }
        case NodeType::SIN: {
            AD C = children[0]->ad_eval(x, y);
            double s = std::sin(C.v), c = std::cos(C.v);
            r.v   = s;
            r.dx  = c * C.dx;
            r.dy  = c * C.dy;
            // (sin f)'' = cos(f)*f'' - sin(f)*(f')²
            r.dxx = c*C.dxx - s*C.dx*C.dx;
            r.dyy = c*C.dyy - s*C.dy*C.dy;
            break;
        }
        case NodeType::COS: {
            AD C = children[0]->ad_eval(x, y);
            double s = std::sin(C.v), c = std::cos(C.v);
            r.v   = c;
            r.dx  = -s * C.dx;
            r.dy  = -s * C.dy;
            r.dxx = -c*C.dxx - s*(-C.dx*C.dx);
            // Wait: (cos f)'' = -sin(f)*f'' - cos(f)*(f')²
            r.dxx = -s*C.dxx - c*C.dx*C.dx;
            r.dyy = -s*C.dyy - c*C.dy*C.dy;
            break;
        }
        case NodeType::EXP: {
            AD C = children[0]->ad_eval(x, y);
            // Clamp to avoid overflow
            double ev = std::exp(std::min(C.v, 20.0));
            r.v   = ev;
            r.dx  = ev * C.dx;
            r.dy  = ev * C.dy;
            // (exp f)'' = exp(f)*(f'' + (f')²)
            r.dxx = ev * (C.dxx + C.dx*C.dx);
            r.dyy = ev * (C.dyy + C.dy*C.dy);
            break;
        }
    }
    return r;
}

// ─── eval simple ──────────────────────────────────────────────────────────────
double Node::eval(double x, double y) const {
    return ad_eval(x, y).v;
}

// ─── mutate_erc ───────────────────────────────────────────────────────────────
void Node::mutate_erc(std::mt19937& gen, double sigma) {
    if (type == NodeType::ERC) {
        std::normal_distribution<double> nd(0.0, sigma);
        erc_val += nd(gen);
    }
    for (auto& c : children) c->mutate_erc(gen, sigma);
}

// ─── clone ────────────────────────────────────────────────────────────────────
NodePtr Node::clone() const {
    auto n = std::make_shared<Node>();
    n->type    = type;
    n->erc_val = erc_val;
    n->children.reserve(children.size());
    for (auto& c : children) n->children.push_back(c->clone());
    return n;
}

// ─── size ─────────────────────────────────────────────────────────────────────
int Node::size() const {
    int s = 1;
    for (auto& c : children) s += c->size();
    return s;
}

// ─── print ────────────────────────────────────────────────────────────────────
void Node::print(std::ostream& os) const {
    switch (type) {
        case NodeType::VAR_X: os << "x"; break;
        case NodeType::VAR_Y: os << "y"; break;
        case NodeType::ERC:   os << erc_val; break;
        case NodeType::ADD:
            os << "("; children[0]->print(os); os << "+"; children[1]->print(os); os << ")"; break;
        case NodeType::SUB:
            os << "("; children[0]->print(os); os << "-"; children[1]->print(os); os << ")"; break;
        case NodeType::MUL:
            os << "("; children[0]->print(os); os << "*"; children[1]->print(os); os << ")"; break;
        case NodeType::SIN:
            os << "sin("; children[0]->print(os); os << ")"; break;
        case NodeType::COS:
            os << "cos("; children[0]->print(os); os << ")"; break;
        case NodeType::EXP:
            os << "exp("; children[0]->print(os); os << ")"; break;
    }
}

// ─── get_node_ref ─────────────────────────────────────────────────────────────
NodePtr& Node::get_node_ref(int& counter) {
    // DFS: cuando counter llega a 0 en un hijo, devuelve referencia al hijo
    for (auto& c : children) {
        if (counter == 0) return c;
        --counter;
        NodePtr& found = c->get_node_ref(counter);
        if (counter < 0) return found;
    }
    static NodePtr dummy = nullptr;
    return dummy;
}

// ─── Constructores de nodos ───────────────────────────────────────────────────
NodePtr make_var(char v) {
    auto n = std::make_shared<Node>();
    n->type = (v == 'x') ? NodeType::VAR_X : NodeType::VAR_Y;
    return n;
}
NodePtr make_erc(double val) {
    auto n = std::make_shared<Node>();
    n->type    = NodeType::ERC;
    n->erc_val = val;
    return n;
}
NodePtr make_binary(NodeType op, NodePtr l, NodePtr r) {
    auto n = std::make_shared<Node>();
    n->type = op;
    n->children = {l, r};
    return n;
}
NodePtr make_unary(NodeType op, NodePtr child) {
    auto n = std::make_shared<Node>();
    n->type = op;
    n->children = {child};
    return n;
}

// ─── Generación aleatoria de árbol ────────────────────────────────────────────
NodePtr random_tree(int max_depth, std::mt19937& gen, bool force_terminal) {
    static const std::vector<NodeType> binaries = {NodeType::ADD, NodeType::SUB, NodeType::MUL};
    static const std::vector<NodeType> unaries  = {NodeType::SIN, NodeType::COS, NodeType::EXP};
    static const std::vector<NodeType> terminals= {NodeType::VAR_X, NodeType::VAR_Y, NodeType::ERC};

    std::uniform_real_distribution<double> ud(0.0, 1.0);
    std::uniform_real_distribution<double> erc_dist(-3.0, 3.0);

    bool terminal = force_terminal || (max_depth <= 0) || (ud(gen) < 0.35);
    if (terminal) {
        std::uniform_int_distribution<int> ti(0, 2);
        int choice = ti(gen);
        if (choice == 0) return make_var('x');
        if (choice == 1) return make_var('y');
        return make_erc(erc_dist(gen));
    }

    // Probabilidad: 60% binario, 40% unario
    if (ud(gen) < 0.6) {
        std::uniform_int_distribution<int> bi(0, 2);
        NodeType op = binaries[bi(gen)];
        return make_binary(op,
            random_tree(max_depth - 1, gen),
            random_tree(max_depth - 1, gen));
    } else {
        std::uniform_int_distribution<int> ui(0, 2);
        NodeType op = unaries[ui(gen)];
        return make_unary(op, random_tree(max_depth - 1, gen));
    }
}

// ─── Crossover de subárbol ────────────────────────────────────────────────────
std::pair<NodePtr, NodePtr> tree_crossover(const NodePtr& p1,
                                            const NodePtr& p2,
                                            std::mt19937& gen) {
    NodePtr c1 = p1->clone();
    NodePtr c2 = p2->clone();
    int s1 = c1->size(), s2 = c2->size();
    if (s1 <= 1 || s2 <= 1) return {c1, c2};

    // Elegimos un nodo de cruce (excluyendo la raíz → [1, size-1])
    std::uniform_int_distribution<int> d1(1, s1 - 1);
    std::uniform_int_distribution<int> d2(1, s2 - 1);
    int k1 = d1(gen), k2 = d2(gen);

    // Obtenemos referencias a los hijos en la posición indicada
    int cnt1 = k1, cnt2 = k2;
    NodePtr& ref1 = c1->get_node_ref(cnt1);
    NodePtr& ref2 = c2->get_node_ref(cnt2);
    if (ref1 && ref2) std::swap(ref1, ref2);
    return {c1, c2};
}

// ─── Mutación de subárbol ─────────────────────────────────────────────────────
NodePtr tree_mutate(const NodePtr& tree, std::mt19937& gen) {
    NodePtr c = tree->clone();
    int sz = c->size();
    if (sz <= 1) {
        // Reemplaza la raíz completa
        return random_tree(Config::MAX_TREE_DEPTH, gen);
    }
    std::uniform_int_distribution<int> d(0, sz - 1);
    int k = d(gen);
    if (k == 0) return random_tree(Config::MAX_TREE_DEPTH, gen);
    int cnt = k;
    NodePtr& ref = c->get_node_ref(cnt);
    if (ref) ref = random_tree(Config::MAX_TREE_DEPTH - 2, gen, false);
    return c;
}

// ─── Laplaciano por diferencias finitas (Koza) ────────────────────────────────
double fd_laplacian(const NodePtr& tree, double x, double y, double h) {
    double v    = tree->eval(x,   y);
    double vxp  = tree->eval(x+h, y);
    double vxm  = tree->eval(x-h, y);
    double vyp  = tree->eval(x,   y+h);
    double vym  = tree->eval(x,   y-h);
    double dxx  = (vxp - 2.0*v + vxm) / (h*h);
    double dyy  = (vyp - 2.0*v + vym) / (h*h);
    return dxx + dyy;
}
