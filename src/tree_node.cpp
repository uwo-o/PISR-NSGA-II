// =============================================================================
// tree_node.cpp  —  Árbol de expresión con AD exacto para todos los operadores
//
// Reglas de diferenciación de segundo orden (regla de la cadena):
//   Si f = g(u), u=u(x):
//     f'  = g'(u) · u'
//     f'' = g''(u) · (u')² + g'(u) · u''
//
// Para funciones en 2D, se aplica independientemente en x e y.
// =============================================================================

#include "tree_node.hpp"
#include <cmath>
#include <stdexcept>
#include <sstream>

static constexpr double LOG_EPS  = 1e-8;   // protección log(|·|+ε)
static constexpr double SQRT_EPS = 1e-8;   // protección sqrt(·+ε)
static constexpr double EXP_CLAMP = 30.0;  // clamp exp para evitar overflow

// ─── ad_eval ──────────────────────────────────────────────────────────────────
AD Node::ad_eval(double x, double y) const {
    AD r;
    switch (type) {

    // ── Terminales ─────────────────────────────────────────────────────────────
    case NodeType::VAR_X:
        r.v = x; r.dx = 1.0; r.dy = 0.0; r.dxx = 0.0; r.dyy = 0.0;
        break;
    case NodeType::VAR_Y:
        r.v = y; r.dx = 0.0; r.dy = 1.0; r.dxx = 0.0; r.dyy = 0.0;
        break;
    case NodeType::ERC:
        r.v = erc_val; // dx=dy=dxx=dyy=0 por defecto
        break;

    // ── Operadores binarios ────────────────────────────────────────────────────
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
        // (uv)' = u'v + uv'
        r.dx  = L.dx*R.v  + L.v*R.dx;
        r.dy  = L.dy*R.v  + L.v*R.dy;
        // (uv)'' = u''v + 2u'v' + uv''
        r.dxx = L.dxx*R.v + 2.0*L.dx*R.dx + L.v*R.dxx;
        r.dyy = L.dyy*R.v + 2.0*L.dy*R.dy + L.v*R.dyy;
        break;
    }

    // ── Trigonométricas ────────────────────────────────────────────────────────
    case NodeType::SIN: {
        AD C = children[0]->ad_eval(x, y);
        double s = std::sin(C.v), c = std::cos(C.v);
        r.v   = s;
        r.dx  = c * C.dx;
        r.dy  = c * C.dy;
        // (sin u)'' = cos(u)·u'' - sin(u)·(u')²
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
        // (cos u)'' = -sin(u)·u'' - cos(u)·(u')²
        r.dxx = -s*C.dxx - c*C.dx*C.dx;
        r.dyy = -s*C.dyy - c*C.dy*C.dy;
        break;
    }

    // ── Hiperbólicas (clave para ecuaciones de Laplace, difusión, onda) ────────
    case NodeType::SINH: {
        AD C = children[0]->ad_eval(x, y);
        double sh = std::sinh(C.v), ch = std::cosh(C.v);
        r.v   = sh;
        r.dx  = ch * C.dx;
        r.dy  = ch * C.dy;
        // (sinh u)'' = cosh(u)·u'' + sinh(u)·(u')²
        r.dxx = ch*C.dxx + sh*C.dx*C.dx;
        r.dyy = ch*C.dyy + sh*C.dy*C.dy;
        break;
    }
    case NodeType::COSH: {
        AD C = children[0]->ad_eval(x, y);
        double sh = std::sinh(C.v), ch = std::cosh(C.v);
        r.v   = ch;
        r.dx  = sh * C.dx;
        r.dy  = sh * C.dy;
        // (cosh u)'' = sinh(u)·u'' + cosh(u)·(u')²
        r.dxx = sh*C.dxx + ch*C.dx*C.dx;
        r.dyy = sh*C.dyy + ch*C.dy*C.dy;
        break;
    }
    case NodeType::TANH: {
        AD C = children[0]->ad_eval(x, y);
        double th = std::tanh(C.v);
        double sech2 = 1.0 - th*th;          // sech²(u)
        r.v   = th;
        r.dx  = sech2 * C.dx;
        r.dy  = sech2 * C.dy;
        // (tanh u)'' = sech²(u)·u'' - 2·tanh(u)·sech²(u)·(u')²
        r.dxx = sech2*C.dxx - 2.0*th*sech2*C.dx*C.dx;
        r.dyy = sech2*C.dyy - 2.0*th*sech2*C.dy*C.dy;
        break;
    }

    // ── Exponencial ────────────────────────────────────────────────────────────
    case NodeType::EXP: {
        AD C = children[0]->ad_eval(x, y);
        double ev = std::exp(std::min(C.v, EXP_CLAMP));
        r.v   = ev;
        r.dx  = ev * C.dx;
        r.dy  = ev * C.dy;
        // (exp u)'' = exp(u)·(u'' + (u')²)
        r.dxx = ev * (C.dxx + C.dx*C.dx);
        r.dyy = ev * (C.dyy + C.dy*C.dy);
        break;
    }

    // ── Raíz cuadrada — aparece en ecuaciones de Bessel y potencial ────────────
    case NodeType::SQRT: {
        AD C = children[0]->ad_eval(x, y);
        double arg = std::abs(C.v) + SQRT_EPS;
        double sq  = std::sqrt(arg);
        double sign_v = (C.v >= 0.0) ? 1.0 : -1.0; // preserva signo
        // g(u) = sqrt(|u|+ε),  g'(u) = sign(u)/(2·sqrt(|u|+ε))
        double gp  =  sign_v / (2.0 * sq);
        // g''(u) = -1/(4·(|u|+ε)^{3/2})
        double gpp = -1.0    / (4.0 * sq * arg);
        r.v   = sq;
        r.dx  = gp  * C.dx;
        r.dy  = gp  * C.dy;
        r.dxx = gpp*C.dx*C.dx + gp*C.dxx;
        r.dyy = gpp*C.dy*C.dy + gp*C.dyy;
        break;
    }

    // ── Logaritmo — fundamental en Laplace polar, función de Green ────────────
    case NodeType::LOG: {
        AD C = children[0]->ad_eval(x, y);
        double arg = std::abs(C.v) + LOG_EPS;
        double sign_v = (C.v >= 0.0) ? 1.0 : -1.0;
        // g(u) = log(|u|+ε),  g'(u) = sign(u)/(|u|+ε)
        double gp  =  sign_v / arg;
        // g''(u) = -1/(|u|+ε)²
        double gpp = -1.0 / (arg * arg);
        r.v   = std::log(arg);
        r.dx  = gp  * C.dx;
        r.dy  = gp  * C.dy;
        r.dxx = gpp*C.dx*C.dx + gp*C.dxx;
        r.dyy = gpp*C.dy*C.dy + gp*C.dyy;
        break;
    }

    // ── Arcotangente — soluciones Laplace en polares ────────────────────────────
    case NodeType::ATAN: {
        AD C = children[0]->ad_eval(x, y);
        double denom  = 1.0 + C.v*C.v;         // 1+u²
        double gp     = 1.0 / denom;            // 1/(1+u²)
        double gpp    = -2.0*C.v / (denom*denom); // -2u/(1+u²)²
        r.v   = std::atan(C.v);
        r.dx  = gp  * C.dx;
        r.dy  = gp  * C.dy;
        r.dxx = gpp*C.dx*C.dx + gp*C.dxx;
        r.dyy = gpp*C.dy*C.dy + gp*C.dyy;
        break;
    }

    } // end switch
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
        case NodeType::SIN:  os << "sin(";  children[0]->print(os); os << ")"; break;
        case NodeType::COS:  os << "cos(";  children[0]->print(os); os << ")"; break;
        case NodeType::SINH: os << "sinh("; children[0]->print(os); os << ")"; break;
        case NodeType::COSH: os << "cosh("; children[0]->print(os); os << ")"; break;
        case NodeType::TANH: os << "tanh("; children[0]->print(os); os << ")"; break;
        case NodeType::EXP:  os << "exp(";  children[0]->print(os); os << ")"; break;
        case NodeType::SQRT: os << "sqrt("; children[0]->print(os); os << ")"; break;
        case NodeType::LOG:  os << "log(";  children[0]->print(os); os << ")"; break;
        case NodeType::ATAN: os << "atan("; children[0]->print(os); os << ")"; break;
    }
}

// ─── print_latex ──────────────────────────────────────────────────────────────
void Node::print_latex(std::ostream& os) const {
    switch (type) {
        case NodeType::VAR_X: os << "x"; break;
        case NodeType::VAR_Y: os << "y"; break;
        case NodeType::ERC:   os << erc_val; break;
        case NodeType::ADD:
            os << "("; children[0]->print_latex(os); os << " + "; children[1]->print_latex(os); os << ")"; break;
        case NodeType::SUB:
            os << "("; children[0]->print_latex(os); os << " - "; children[1]->print_latex(os); os << ")"; break;
        case NodeType::MUL:
            os << "("; children[0]->print_latex(os); os << " \\cdot "; children[1]->print_latex(os); os << ")"; break;
        case NodeType::SIN:  os << "\\sin(";  children[0]->print_latex(os); os << ")"; break;
        case NodeType::COS:  os << "\\cos(";  children[0]->print_latex(os); os << ")"; break;
        case NodeType::SINH: os << "\\sinh("; children[0]->print_latex(os); os << ")"; break;
        case NodeType::COSH: os << "\\cosh("; children[0]->print_latex(os); os << ")"; break;
        case NodeType::TANH: os << "\\tanh("; children[0]->print_latex(os); os << ")"; break;
        case NodeType::EXP:  os << "\\exp(";  children[0]->print_latex(os); os << ")"; break;
        case NodeType::SQRT: os << "\\sqrt{"; children[0]->print_latex(os); os << "}"; break;
        case NodeType::LOG:  os << "\\log(";  children[0]->print_latex(os); os << ")"; break;
        case NodeType::ATAN: os << "\\arctan("; children[0]->print_latex(os); os << ")"; break;
    }
}

// ─── get_node_ref ─────────────────────────────────────────────────────────────
NodePtr& Node::get_node_ref(int& counter) {
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

// ─── Generación aleatoria de árbol (ramped half-and-half) ────────────────────
// Conjunto completo de operadores para PDE/ODE:
//   Trigonométricas: SIN, COS
//   Hiperbólicas:    SINH, COSH, TANH
//   Transcendentes:  EXP, LOG, SQRT, ATAN
//   Binarias:        ADD, SUB, MUL
NodePtr random_tree(int max_depth, std::mt19937& gen, bool force_terminal) {
    static const std::vector<NodeType> binaries = {
        NodeType::ADD, NodeType::SUB, NodeType::MUL
    };
    static const std::vector<NodeType> unaries = {
        NodeType::SIN, NodeType::COS,
        NodeType::SINH, NodeType::COSH, NodeType::TANH,
        NodeType::EXP, NodeType::SQRT, NodeType::LOG, NodeType::ATAN
    };
    static const std::vector<NodeType> terminals = {
        NodeType::VAR_X, NodeType::VAR_Y, NodeType::ERC
    };

    std::uniform_real_distribution<double> ud(0.0, 1.0);
    std::uniform_real_distribution<double> erc_dist(-3.0, 3.0);

    bool terminal = force_terminal || (max_depth <= 0) || (ud(gen) < 0.30);
    if (terminal) {
        std::uniform_int_distribution<int> ti(0, 2);
        int choice = ti(gen);
        if (choice == 0) return make_var('x');
        if (choice == 1) return make_var('y');
        return make_erc(erc_dist(gen));
    }

    // Probabilidad: 55% binario, 45% unario
    if (ud(gen) < 0.55) {
        std::uniform_int_distribution<int> bi(0, (int)binaries.size() - 1);
        NodeType op = binaries[bi(gen)];
        return make_binary(op,
            random_tree(max_depth - 1, gen),
            random_tree(max_depth - 1, gen));
    } else {
        std::uniform_int_distribution<int> ui(0, (int)unaries.size() - 1);
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

    std::uniform_int_distribution<int> d1(1, s1 - 1);
    std::uniform_int_distribution<int> d2(1, s2 - 1);
    int k1 = d1(gen), k2 = d2(gen);

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
    double v   = tree->eval(x,   y);
    double vxp = tree->eval(x+h, y);
    double vxm = tree->eval(x-h, y);
    double vyp = tree->eval(x,   y+h);
    double vym = tree->eval(x,   y-h);
    double dxx = (vxp - 2.0*v + vxm) / (h*h);
    double dyy = (vyp - 2.0*v + vym) / (h*h);
    return dxx + dyy;
}
