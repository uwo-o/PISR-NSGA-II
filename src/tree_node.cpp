// =============================================================================
// tree_node.cpp  —  Motor de Regresión Simbólica (Versión Física Estándar)
// =============================================================================

#include "tree_node.hpp"
#include "pde_problems.hpp"
#include <cmath>
#include <algorithm>
#include <iomanip>

static const double PI_VAL = std::acos(-1.0);

// Helper para polinomios ortogonales y sus derivadas de 2do orden
static void eval_poly_all(NodeType type, int n, double x, double& v, double& dv, double& dvv) {
    if (n <= 0) { v = 1.0; dv = 0.0; dvv = 0.0; return; }
    if (n == 1) {
        if (type == NodeType::HERMITE) { v = 2.0*x; dv = 2.0; dvv = 0.0; }
        else { v = x; dv = 1.0; dvv = 0.0; }
        return;
    }
    double p0 = 1.0, p1 = (type == NodeType::HERMITE) ? 2.0*x : x;
    double dp0 = 0.0, dp1 = (type == NodeType::HERMITE) ? 2.0 : 1.0;
    double ddp0 = 0.0, ddp1 = 0.0;
    for (int i = 2; i <= n; ++i) {
        double cur_v, cur_dv, cur_dvv;
        if (type == NodeType::LEGENDRE) {
            cur_v = ((2.0*i-1.0)*x*p1 - (i-1.0)*p0) / (double)i;
            cur_dv = ((2.0*i-1.0)*(p1 + x*dp1) - (i-1.0)*dp0) / (double)i;
            cur_dvv = ((2.0*i-1.0)*(2.0*dp1 + x*ddp1) - (i-1.0)*ddp0) / (double)i;
        } else if (type == NodeType::HERMITE) {
            cur_v = 2.0*x*p1 - 2.0*(i-1.0)*p0;
            cur_dv = 2.0*(p1 + x*dp1) - 2.0*(i-1.0)*dp0;
            cur_dvv = 2.0*(2.0*dp1 + x*ddp1) - 2.0*(i-1.0)*ddp0;
        } else { // Chebyshev
            cur_v = 2.0*x*p1 - p0;
            cur_dv = 2.0*(p1 + x*dp1) - dp0;
            cur_dvv = 2.0*(2.0*dp1 + x*ddp1) - ddp0;
        }
        p0 = p1; p1 = cur_v; dp0 = dp1; dp1 = cur_dv; ddp0 = ddp1; ddp1 = cur_dvv;
    }
    v = p1; dv = dp1; dvv = ddp1;
}

// ─── TerminalNode ────────────────────────────────────────────────────────────
AD TerminalNode::ad_eval(double x, double y, int dim) const { return ad_eval_t(x, y, 0.0, dim); }
AD TerminalNode::ad_eval_t(double x, double y, double t, int dim) const {
    AD r(eval_t(x, y, t));
    if (type == NodeType::VAR_X) r.dx = 1.0;
    else if (type == NodeType::VAR_Y && dim > 1) r.dy = 1.0;
    else if (type == NodeType::VAR_T) r.dt = 1.0;
    return r;
}
Complex TerminalNode::eval(double x, double y) const { return eval_t(x, y, 0.0); }
Complex TerminalNode::eval_t(double x, double y, double t) const {
    if (type == NodeType::VAR_X) return x; if (type == NodeType::VAR_Y) return y;
    if (type == NodeType::VAR_T) return t; if (type == NodeType::ERC) return erc_val;
    if (type == NodeType::CONST_I) return I_COMPLEX;
    if (type == NodeType::CONST_PI) return PI_VAL;
    if (type == NodeType::CONST_E) return std::exp(1.0);
    return 0.0;
}
void TerminalNode::mutate_erc(std::mt19937& gen, double sigma) {
    if (type == NodeType::ERC) { erc_val += std::normal_distribution<double>(0.0, sigma)(gen); }
}
void TerminalNode::print(std::ostream& os) const {
    if (type == NodeType::VAR_X) os << "x"; else if (type == NodeType::VAR_Y) os << "y";
    else if (type == NodeType::VAR_T) os << "t"; else if (type == NodeType::ERC) os << std::fixed << std::setprecision(3) << erc_val.real();
    else if (type == NodeType::CONST_I) os << "i";
    else if (type == NodeType::CONST_PI) os << "pi";
    else if (type == NodeType::CONST_E) os << "e";
}
void TerminalNode::print_latex(std::ostream& os) const {
    if (type == NodeType::VAR_X) os << "x"; else if (type == NodeType::VAR_Y) os << "y";
    else if (type == NodeType::VAR_T) os << "t"; else if (type == NodeType::ERC) os << std::fixed << std::setprecision(3) << erc_val.real();
    else if (type == NodeType::CONST_I) os << "i";
    else if (type == NodeType::CONST_PI) os << "\\pi";
    else if (type == NodeType::CONST_E) os << "e";
}
void TerminalNode::collect_ercs(std::vector<Complex*>& ptrs) { if (type == NodeType::ERC) ptrs.push_back(&erc_val); }
bool TerminalNode::uses_variable(NodeType vt) const { return type == vt; }
bool TerminalNode::contains_variables() const { return type == NodeType::VAR_X || type == NodeType::VAR_Y || type == NodeType::VAR_T; }
std::optional<Dimension> TerminalNode::get_dimension(const PDEProblem& p) const {
    if (type == NodeType::VAR_X) return p.dim_x; if (type == NodeType::VAR_Y) return p.dim_y;
    if (type == NodeType::VAR_T) return p.dim_t; return Units::None; 
}
NodePtr TerminalNode::simplify() const { return clone(); }
NodePtr TerminalNode::prune_recursive(const PDEProblem& p, const std::vector<Point>& d, const std::vector<Point>& b, double o, double t) { return clone(); }

// ─── UnaryNode ───────────────────────────────────────────────────────────────
AD UnaryNode::ad_eval(double x, double y, int dim) const { return ad_eval_t(x, y, 0.0, dim); }
AD UnaryNode::ad_eval_t(double x, double y, double t, int dim) const {
    AD r; if (!child) return r; AD C = child->ad_eval_t(x, y, t, dim);
    if (type == NodeType::SIN) {
        Complex s = std::sin(C.v), c = std::cos(C.v); r.v = s; r.dx = c*C.dx; r.dy = c*C.dy; r.dt = c*C.dt;
        r.dxx = c*C.dxx - s*C.dx*C.dx; r.dyy = c*C.dyy - s*C.dy*C.dy; r.dtt = c*C.dtt - s*C.dt*C.dt;
    } else if (type == NodeType::COS) {
        Complex s = std::sin(C.v), c = std::cos(C.v); r.v = c; r.dx = -s*C.dx; r.dy = -s*C.dy; r.dt = -s*C.dt;
        r.dxx = -s*C.dxx - c*C.dx*C.dx; r.dyy = -s*C.dyy - c*C.dy*C.dy; r.dtt = -s*C.dtt - c*C.dt*C.dt;
    } else if (type == NodeType::EXP) {
        Complex ev = std::exp(C.v); r.v = ev; r.dx = ev*C.dx; r.dy = ev*C.dy; r.dt = ev*C.dt;
        r.dxx = ev*(C.dxx + C.dx*C.dx); r.dyy = ev*(C.dyy + C.dy*C.dy); r.dtt = ev*(C.dtt + C.dt*C.dt);
    } else if (type == NodeType::SQR) {
        r.v = C.v*C.v; r.dx = 2.0*C.v*C.dx; r.dy = 2.0*C.v*C.dy; r.dt = 2.0*C.v*C.dt;
        r.dxx = 2.0*(C.dx*C.dx + C.v*C.dxx); r.dyy = 2.0*(C.dy*C.dy + C.v*C.dyy); r.dtt = 2.0*(C.dt*C.dt + C.v*C.dtt);
    } else if (type == NodeType::GAUSSIAN) {
        // DEFINICIÓN FÍSICA: exp(-0.5 * v^2)
        Complex g = std::exp(-0.5 * C.v * C.v); r.v = g;
        Complex dG = -C.v * g; Complex ddG = g * (C.v * C.v - 1.0);
        r.dx = dG*C.dx; r.dy = dG*C.dy; r.dt = dG*C.dt;
        r.dxx = ddG*C.dx*C.dx + dG*C.dxx; 
        r.dyy = ddG*C.dy*C.dy + dG*C.dyy;
        r.dtt = ddG*C.dt*C.dt + dG*C.dtt;
    } else if (type == NodeType::TANH) {
        Complex th = std::tanh(C.v); Complex sech2 = 1.0 - th*th;
        r.v = th; r.dx = sech2*C.dx; r.dy = sech2*C.dy; r.dt = sech2*C.dt;
        r.dxx = sech2*C.dxx - 2.0*th*sech2*C.dx*C.dx; 
        r.dyy = sech2*C.dyy - 2.0*th*sech2*C.dy*C.dy;
        r.dtt = sech2*C.dtt - 2.0*th*sech2*C.dt*C.dt;
    } else if (type == NodeType::LOG) {
        Complex val = (std::abs(C.v) < 1e-9) ? Complex(1e-9, 0.0) : C.v;
        r.v = std::log(val); r.dx = C.dx / val; r.dy = C.dy / val; r.dt = C.dt / val;
        r.dxx = (C.dxx*val - C.dx*C.dx) / (val*val);
        r.dyy = (C.dyy*val - C.dy*C.dy) / (val*val);
        r.dtt = (C.dtt*val - C.dt*C.dt) / (val*val);
    } else if (type == NodeType::SINH) {
        Complex sh = std::sinh(C.v), ch = std::cosh(C.v);
        r.v = sh; r.dx = ch*C.dx; r.dy = ch*C.dy; r.dt = ch*C.dt;
        r.dxx = sh*C.dx*C.dx + ch*C.dxx; r.dyy = sh*C.dy*C.dy + ch*C.dyy; r.dtt = sh*C.dt*C.dt + ch*C.dtt;
    } else if (type == NodeType::COSH) {
        Complex sh = std::sinh(C.v), ch = std::cosh(C.v);
        r.v = ch; r.dx = sh*C.dx; r.dy = sh*C.dy; r.dt = sh*C.dt;
        r.dxx = ch*C.dx*C.dx + sh*C.dxx; r.dyy = ch*C.dy*C.dy + sh*C.dyy; r.dtt = ch*C.dt*C.dt + sh*C.dtt;
    }
    return r;
}
Complex UnaryNode::eval(double x, double y) const { return eval_t(x, y, 0.0); }
Complex UnaryNode::eval_t(double x, double y, double t) const {
    if (!child) return 0.0; Complex v = child->eval_t(x, y, t);
    if (type == NodeType::SIN) return std::sin(v); if (type == NodeType::COS) return std::cos(v);
    if (type == NodeType::EXP) return std::exp(v); if (type == NodeType::SQR) return v*v;
    if (type == NodeType::GAUSSIAN) return std::exp(-0.5*v*v); if (type == NodeType::TANH) return std::tanh(v);
    if (type == NodeType::LOG) return std::log(std::abs(v) < 1e-9 ? 1e-9 : v);
    if (type == NodeType::SINH) return std::sinh(v);
    if (type == NodeType::COSH) return std::cosh(v);
    return 0.0;
}
void UnaryNode::print(std::ostream& os) const {
    if (type == NodeType::SIN) os << "sin("; 
    else if (type == NodeType::COS) os << "cos(";
    else if (type == NodeType::EXP) os << "exp(";
    else if (type == NodeType::SQR) os << "sqr(";
    else if (type == NodeType::GAUSSIAN) os << "G(";
    else if (type == NodeType::TANH) os << "tanh(";
    else if (type == NodeType::LOG) os << "ln(";
    else if (type == NodeType::SINH) os << "sinh(";
    else if (type == NodeType::COSH) os << "cosh(";
    else os << "u(";
    if (child) child->print(os); os << ")";
}
void UnaryNode::print_latex(std::ostream& os) const {
    if (type == NodeType::EXP) os << "e^{";
    else if (type == NodeType::GAUSSIAN) os << "\\mathcal{G}(";
    else if (type == NodeType::SQR) os << "(";
    else if (type == NodeType::SIN) os << "\\sin(";
    else if (type == NodeType::COS) os << "\\cos(";
    else if (type == NodeType::SINH) os << "\\sinh(";
    else if (type == NodeType::COSH) os << "\\cosh(";
    else if (type == NodeType::TANH) os << "\\tanh(";
    else if (type == NodeType::LOG) os << "\\ln(";
    else os << "\\text{u}(";
    if (child) child->print_latex(os);
    if (type == NodeType::EXP) os << "}";
    else if (type == NodeType::SQR) os << ")^2";
    else os << ")";
}
void UnaryNode::collect_ercs(std::vector<Complex*>& ptrs) { if (child) child->collect_ercs(ptrs); }
bool UnaryNode::uses_variable(NodeType vt) const { return child ? child->uses_variable(vt) : false; }
bool UnaryNode::contains_variables() const { return child ? child->contains_variables() : false; }
std::optional<Dimension> UnaryNode::get_dimension(const PDEProblem& p) const {
    if (!child) return std::nullopt; auto d = child->get_dimension(p); if (!d) return std::nullopt;
    if (type == NodeType::SQR) return *d + *d;
    return (d->is_adimensional() || !child->contains_variables()) ? std::optional<Dimension>(Units::None) : std::nullopt;
}
NodePtr UnaryNode::simplify() const { return clone(); }
NodePtr UnaryNode::prune_recursive(const PDEProblem& p, const std::vector<Point>& d, const std::vector<Point>& b, double o, double t) { return clone(); }

// ─── BinaryNode ──────────────────────────────────────────────────────────────
AD BinaryNode::ad_eval(double x, double y, int dim) const { return ad_eval_t(x, y, 0.0, dim); }
AD BinaryNode::ad_eval_t(double x, double y, double t, int dim) const {
    AD r; if (!left || !right) return r; AD L = left->ad_eval_t(x, y, t, dim); AD R = right->ad_eval_t(x, y, t, dim);
    if (type == NodeType::ADD) {
        r.v = L.v+R.v; r.dx = L.dx+R.dx; r.dy = L.dy+R.dy; r.dt = L.dt+R.dt; r.dxx = L.dxx+R.dxx; r.dyy = L.dyy+R.dyy;
    } else if (type == NodeType::SUB) {
        r.v = L.v-R.v; r.dx = L.dx-R.dx; r.dy = L.dy-R.dy; r.dt = L.dt-R.dt; r.dxx = L.dxx-R.dxx; r.dyy = L.dyy-R.dyy;
    } else if (type == NodeType::MUL) {
        r.v = L.v * R.v; r.dx = L.dx*R.v+L.v*R.dx; r.dy = L.dy*R.v+L.v*R.dy; r.dt = L.dt*R.v+L.v*R.dt;
        r.dxx = L.dxx*R.v + 2.0*L.dx*R.dx + L.v*R.dxx; r.dyy = L.dyy*R.v + 2.0*L.dy*R.dy + L.v*R.dyy;
    } else if (type == NodeType::DIV) {
        Complex rv = (std::abs(R.v) < 1e-9) ? Complex(1e-9, 0.0) : R.v;
        r.v = L.v / rv;
        r.dx = (L.dx*rv - L.v*R.dx)/(rv*rv); 
        r.dy = (L.dy*rv - L.v*R.dy)/(rv*rv);
        r.dt = (L.dt*rv - L.v*R.dt)/(rv*rv);
        r.dxx = L.dxx/rv - 2.0*L.dx*R.dx/(rv*rv) - L.v*R.dxx/(rv*rv) + 2.0*L.v*R.dx*R.dx/(rv*rv*rv);
        r.dyy = L.dyy/rv - 2.0*L.dy*R.dy/(rv*rv) - L.v*R.dyy/(rv*rv) + 2.0*L.v*R.dy*R.dy/(rv*rv*rv);
        r.dtt = L.dtt/rv - 2.0*L.dt*R.dt/(rv*rv) - L.v*R.dtt/(rv*rv) + 2.0*L.v*R.dt*R.dt/(rv*rv*rv);
    } else { // Polinomios Ortogonales
        int n = std::clamp((int)std::round(R.v.real()), 0, 5); double pv, pdv, pdvv; eval_poly_all(type, n, L.v.real(), pv, pdv, pdvv);
        r.v = pv; r.dx = pdv*L.dx; r.dy = pdv*L.dy; r.dt = pdv*L.dt; r.dxx = pdvv*L.dx*L.dx+pdv*L.dxx; r.dyy = pdvv*L.dy*L.dy+pdv*L.dyy;
    }
    return r;
}
Complex BinaryNode::eval(double x, double y) const { return eval_t(x, y, 0.0); }
Complex BinaryNode::eval_t(double x, double y, double t) const {
    if (!left || !right) return 0.0; Complex lv = left->eval_t(x, y, t), rv = right->eval_t(x, y, t);
    if (type == NodeType::ADD) return lv+rv; if (type == NodeType::SUB) return lv-rv;
    if (type == NodeType::MUL) return lv*rv; if (type == NodeType::DIV) return (std::abs(rv)<1e-9)?lv/1e-9:lv/rv;
    double pv, pdv, pdvv; eval_poly_all(type, (int)std::round(rv.real()), lv.real(), pv, pdv, pdvv); return pv;
}
void BinaryNode::print(std::ostream& os) const {
    if (type == NodeType::DIV) { os << "("; if(left) left->print(os); os << "/"; if(right) right->print(os); os << ")"; return; }
    if (type == NodeType::ADD) { os << "("; if(left) left->print(os); os << " + "; if(right) right->print(os); os << ")"; return; }
    if (type == NodeType::SUB) { os << "("; if(left) left->print(os); os << " - "; if(right) right->print(os); os << ")"; return; }
    if (type == NodeType::MUL) { os << "("; if(left) left->print(os); os << " * "; if(right) right->print(os); os << ")"; return; }
    
    std::string p_name = "P";
    if (type == NodeType::HERMITE) p_name = "H";
    else if (type == NodeType::CHEBYSHEV) p_name = "T";
    else if (type == NodeType::LAGUERRE) p_name = "L";
    os << p_name << "_{"; if (right) right->print(os); os << "}("; if (left) left->print(os); os << ")";
}
void BinaryNode::print_latex(std::ostream& os) const {
    if (type == NodeType::DIV) {
        os << "\\frac{"; if (left) left->print_latex(os); os << "}{";
        if (right) right->print_latex(os); os << "}";
        return;
    }
    if (type != NodeType::ADD && type != NodeType::SUB && type != NodeType::MUL && type != NodeType::DIV) {
        std::string p_name = "P";
        if (type == NodeType::HERMITE) p_name = "H";
        else if (type == NodeType::CHEBYSHEV) p_name = "T";
        else if (type == NodeType::LAGUERRE) p_name = "L";
        os << p_name << "_{"; if (right) right->print_latex(os); os << "}("; if (left) left->print_latex(os); os << ")";
        return;
    }
    os << "("; if (left) left->print_latex(os);
    if (type == NodeType::ADD) os << " + "; 
    else if (type == NodeType::SUB) os << " - ";
    else if (type == NodeType::MUL) os << " \\cdot ";
    if (right) right->print_latex(os); os << ")";
}
void BinaryNode::collect_ercs(std::vector<Complex*>& ptrs) { if (left) left->collect_ercs(ptrs); if (right) right->collect_ercs(ptrs); }
bool BinaryNode::uses_variable(NodeType vt) const { return (left && left->uses_variable(vt)) || (right && right->uses_variable(vt)); }
bool BinaryNode::contains_variables() const { return (left && left->contains_variables()) || (right && right->contains_variables()); }
std::optional<Dimension> BinaryNode::get_dimension(const PDEProblem& p) const {
    if (!left || !right) return std::nullopt; auto dl = left->get_dimension(p), dr = right->get_dimension(p);
    if (!dl || !dr) return std::nullopt; bool lv = left->contains_variables(), rv = right->contains_variables();
    if (type == NodeType::ADD || type == NodeType::SUB) {
        if (lv && rv) return (*dl == *dr) ? std::optional<Dimension>(*dl) : std::nullopt; return lv ? dl : dr;
    }
    if (type == NodeType::MUL) return *dl + *dr; if (type == NodeType::DIV) return *dl - *dr; return Units::None;
}
NodePtr BinaryNode::simplify() const { return clone(); }
NodePtr BinaryNode::prune_recursive(const PDEProblem& p, const std::vector<Point>& d, const std::vector<Point>& b, double o, double t) { return clone(); }

// ─── Fabricación y Evolución ────────────────────────────────────────────────
NodePtr make_var(char v) { 
    if (v == 'x') return std::make_unique<TerminalNode>(NodeType::VAR_X);
    if (v == 'y') return std::make_unique<TerminalNode>(NodeType::VAR_Y);
    return std::make_unique<TerminalNode>(NodeType::VAR_T);
}
NodePtr make_erc(Complex v) { return std::make_unique<TerminalNode>(NodeType::ERC, v); }
NodePtr make_const_i() { return std::make_unique<TerminalNode>(NodeType::CONST_I); }
NodePtr make_const_pi() { return std::make_unique<TerminalNode>(NodeType::CONST_PI); }
NodePtr make_const_e() { return std::make_unique<TerminalNode>(NodeType::CONST_E); }
NodePtr make_binary(NodeType op, NodePtr l, NodePtr r) { return std::make_unique<BinaryNode>(op, std::move(l), std::move(r)); }
NodePtr make_unary(NodeType op, NodePtr c) { return std::make_unique<UnaryNode>(op, std::move(c)); }

NodePtr random_tree(int depth, std::mt19937& gen, bool force_t) {
    std::uniform_real_distribution<double> ud(0, 1);
    if (depth <= 0 || ud(gen) < 0.3) {
        int v = std::uniform_int_distribution<int>(0, 5)(gen);
        if (v == 0) return make_var('x'); if (v == 1) return make_var('y');
        if (v == 2) return make_var('t');
        if (v == 3) return make_const_pi();
        if (v == 4) return make_const_e();
        return make_erc(std::uniform_real_distribution<double>(-2.0, 2.0)(gen));
    }
    double p = ud(gen);
    if (p < 0.3) {
        NodeType op = (ud(gen) < 0.5) ? NodeType::ADD : (ud(gen) < 0.5 ? NodeType::MUL : NodeType::SUB);
        return make_binary(op, random_tree(depth-1, gen), random_tree(depth-1, gen));
    }
    if (p < 0.6) {
        NodeType op = (ud(gen) < 0.5) ? NodeType::SIN : (ud(gen) < 0.5 ? NodeType::EXP : NodeType::GAUSSIAN);
        return make_unary(op, random_tree(depth-1, gen));
    }
    // Para simplificar, a veces genera polinomios directamente
    if (p < 0.8) {
        NodeType op = (ud(gen) < 0.5) ? NodeType::LEGENDRE : NodeType::HERMITE;
        return make_binary(op, random_tree(depth-1, gen), make_erc(std::uniform_int_distribution<int>(1, 3)(gen)));
    }
    return make_unary(NodeType::TANH, random_tree(depth-1, gen));
}
NodePtr random_tree_special(int depth, std::mt19937& gen, const PDEProblem& prob) { return random_tree(1, gen); }
void replace_node_at(NodePtr& cur, int& idx, NodePtr& rep) {
    if (!cur || !rep) return; if (idx == 0) { cur = std::move(rep); idx = -1; return; }
    idx--; if (auto* un = dynamic_cast<UnaryNode*>(cur.get())) { if (idx >= 0) replace_node_at(un->child, idx, rep); }
    else if (auto* bn = dynamic_cast<BinaryNode*>(cur.get())) { if (idx >= 0) replace_node_at(bn->left, idx, rep); if (idx >= 0) replace_node_at(bn->right, idx, rep); }
}
NodePtr tree_mutate(const NodePtr& t, std::mt19937& gen, const PDEProblem& p) {
    if (!t) return random_tree(2, gen); NodePtr res = t->clone();
    int sz = res->count_nodes(); int target = std::uniform_int_distribution<int>(0, sz-1)(gen);
    NodePtr sub = random_tree(1, gen); replace_node_at(res, target, sub); return res;
}
std::pair<NodePtr, NodePtr> tree_crossover(const NodePtr& p1, const NodePtr& p2, std::mt19937& gen) { return {p1->clone(), p2->clone()}; }

// ─── EL DICCIONARIO FÍSICO COMPLETO (13 ESQUELETOS) ──────────────────────────
NodePtr get_exact_solution_tree(const PDEProblem& prob) {
    if (prob.dim == 1) {
        if (prob.type == PDE::LAPLACE) return make_var('x');
        if (prob.type == PDE::POISSON || prob.type == PDE::HELMHOLTZ || prob.type == PDE::SINE_GORDON) 
            return make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('x')));
        if (prob.type == PDE::SCHRODINGER)
            return make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_const_i(), make_binary(NodeType::MUL, make_const_pi(), make_var('x'))));
        if (prob.type == PDE::HARMONIC_OSCILLATOR)
            return make_unary(NodeType::GAUSSIAN, make_var('x')); 
        if (prob.type == PDE::AIRY) 
            return make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-0.5), make_var('x')));
        if (prob.type == PDE::FISHER || prob.type == PDE::DUFFING)
            return make_unary(NodeType::TANH, make_var('x'));
        if (prob.type == PDE::THOMAS_FERMI)
            return make_binary(NodeType::DIV, make_erc(1.0), make_binary(NodeType::ADD, make_var('x'), make_erc(1.0)));
        if (prob.type == PDE::NONLINEAR_POISSON || prob.type == PDE::LIOUVILLE)
            return make_binary(NodeType::DIV, make_erc(1.0), make_binary(NodeType::ADD, make_erc(1.0), make_unary(NodeType::SQR, make_var('x'))));
    } else {
        if (prob.type == PDE::LAPLACE) {
            auto sx = make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('x')));
            auto sy = make_unary(NodeType::SINH, make_binary(NodeType::MUL, make_const_pi(), make_var('y')));
            auto res = make_binary(NodeType::MUL, std::move(sx), std::move(sy));
            return make_binary(NodeType::DIV, std::move(res), make_erc(std::sinh(PI_VAL)));
        }
        if (prob.type == PDE::POISSON || prob.type == PDE::HELMHOLTZ || prob.type == PDE::SINE_GORDON) {
            auto sx = make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('x')));
            auto sy = make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('y')));
            return make_binary(NodeType::MUL, std::move(sx), std::move(sy));
        }
        if (prob.type == PDE::HARMONIC_OSCILLATOR) {
            auto gx = make_unary(NodeType::GAUSSIAN, make_var('x'));
            auto gy = make_unary(NodeType::GAUSSIAN, make_var('y'));
            return make_binary(NodeType::MUL, std::move(gx), std::move(gy));
        }
        if (prob.type == PDE::AIRY) {
            auto phase = make_binary(NodeType::ADD, make_var('x'), make_var('y'));
            return make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-0.5), std::move(phase)));
        }
        if (prob.type == PDE::FISHER || prob.type == PDE::DUFFING) {
            auto phase = make_binary(NodeType::ADD, make_var('x'), make_var('y'));
            return make_unary(NodeType::TANH, std::move(phase));
        }
        if (prob.type == PDE::NAVIER_STOKES) {
            double nu = prob.k2; double Re = 1.0/nu;
            double lambda = Re/2.0 - std::sqrt(Re*Re/4.0 + 4.0*PI_VAL*PI_VAL);
            auto ex = make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(lambda), make_var('x')));
            auto sy = make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_erc(2.0*PI_VAL), make_var('y')));
            return make_binary(NodeType::SUB, make_var('y'), make_binary(NodeType::DIV, make_binary(NodeType::MUL, std::move(ex), std::move(sy)), make_erc(2.0*PI_VAL*Re)));
        }
        if (prob.type == PDE::NAVIER_STOKES_UNSTEADY) {
            auto sx = make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('x')));
            auto sy = make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('y')));
            auto et = make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_erc(-0.1), make_var('t')));
            return make_binary(NodeType::MUL, std::move(sx), make_binary(NodeType::MUL, std::move(sy), std::move(et)));
        }
        if (prob.type == PDE::NONLINEAR_POISSON || prob.type == PDE::LIOUVILLE || prob.type == PDE::THOMAS_FERMI)
            return make_binary(NodeType::DIV, make_erc(1.0), make_binary(NodeType::ADD, make_erc(1.0), make_binary(NodeType::ADD, make_unary(NodeType::SQR, make_var('x')), make_unary(NodeType::SQR, make_var('y')))));
        if (prob.type == PDE::SCHRODINGER) {
            auto phase = make_binary(NodeType::ADD, make_var('x'), make_var('y'));
            return make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_const_i(), make_binary(NodeType::MUL, make_const_pi(), std::move(phase))));
        }
    }
    return nullptr;
}

NodePtr remove_nested_polynomials(NodePtr node, bool inside_poly) { return node; }
Complex fd_laplacian(const NodePtr& tree, double x, double y, int dim, double h) { return 0.0; }
