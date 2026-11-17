// =============================================================================
// oracle.cpp — ver oracle.hpp
// =============================================================================
#include "oracle.hpp"
#include "pi_solver.hpp"
#include "numerical_solver.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>

namespace {

using UFun = std::function<Complex(double, double, double)>;

std::string sci(double v) {
    std::ostringstream s; s << std::scientific << std::setprecision(2) << v; return s.str();
}

// Jet por diferencias finitas centradas de u(x,y,t): orden 4 para las
// derivadas de primer y segundo orden; orden 2 (paso he) para las de orden 3-4
// que solo necesitan los operadores de orden espacial 4.
AD fd_jet(const UFun& u, const PDEProblem& prob, double x, double y, double t, double H, double he) {
    AD j;
    auto d1 = [&](auto f) { return (-f(2) + 8.0 * f(1) - 8.0 * f(-1) + f(-2)) / (12.0 * H); };
    auto d2 = [&](auto f) { return (-f(2) + 16.0 * f(1) - 30.0 * f(0) + 16.0 * f(-1) - f(-2)) / (12.0 * H * H); };
    j.v = u(x, y, t);
    j.dx = d1([&](int k) { return u(x + k * H, y, t); });
    j.dxx = d2([&](int k) { return u(x + k * H, y, t); });
    if (prob.dim == 2) {
        j.dy = d1([&](int k) { return u(x, y + k * H, t); });
        j.dyy = d2([&](int k) { return u(x, y + k * H, t); });
        j.dxy = (u(x + H, y + H, t) - u(x + H, y - H, t) - u(x - H, y + H, t) + u(x - H, y - H, t)) / (4.0 * H * H);
    }
    if (prob.is_unsteady) j.dt = d1([&](int k) { return u(x, y, t + k * H); });
    if (prob.spatial_order > 2) {
        double h = he;
        auto uxx = [&](double xx, double yy, double tt) { return (u(xx + h, yy, tt) - 2.0 * u(xx, yy, tt) + u(xx - h, yy, tt)) / (h * h); };
        auto uyy = [&](double xx, double yy, double tt) { return (u(xx, yy + h, tt) - 2.0 * u(xx, yy, tt) + u(xx, yy - h, tt)) / (h * h); };
        j.dxxx = (u(x + 2*h, y, t) - 2.0 * u(x + h, y, t) + 2.0 * u(x - h, y, t) - u(x - 2*h, y, t)) / (2.0 * h * h * h);
        j.dyyy = (u(x, y + 2*h, t) - 2.0 * u(x, y + h, t) + 2.0 * u(x, y - h, t) - u(x, y - 2*h, t)) / (2.0 * h * h * h);
        j.dxxxx = (u(x + 2*h, y, t) - 4.0 * u(x + h, y, t) + 6.0 * u(x, y, t) - 4.0 * u(x - h, y, t) + u(x - 2*h, y, t)) / (h * h * h * h);
        j.dyyyy = (u(x, y + 2*h, t) - 4.0 * u(x, y + h, t) + 6.0 * u(x, y, t) - 4.0 * u(x, y - h, t) + u(x, y - 2*h, t)) / (h * h * h * h);
        j.dxxy = (uxx(x, y + h, t) - uxx(x, y - h, t)) / (2.0 * h);
        j.dxyy = (uyy(x + h, y, t) - uyy(x - h, y, t)) / (2.0 * h);
        j.dxxyy = (uxx(x, y + h, t) - 2.0 * uxx(x, y, t) + uxx(x, y - h, t)) / (h * h);
        if (prob.is_unsteady) {
            j.dxxt = (uxx(x, y, t + h) - uxx(x, y, t - h)) / (2.0 * h);
            j.dyyt = (uyy(x, y, t + h) - uyy(x, y, t - h)) / (2.0 * h);
        }
    }
    return j;
}

bool inside(const PDEProblem& prob, const Point& p, double m) {
    if (p.x < m || p.x > 1.0 - m) return false;
    if (prob.dim == 2 && (p.y < m || p.y > 1.0 - m)) return false;
    if (prob.is_unsteady && (p.t < m || p.t > 1.0 - m)) return false;
    return true;
}

const std::set<std::string> kGrammar = {
    "profundidad unaria > 3", "profundidad > MAX_TREE_DEPTH", "trigonometrica anidada",
    "argumento trigonometrico no afin", "polinomio anidado", "exponencial anidada",
    "grado de polinomio invalido"};

std::string join(const std::vector<std::string>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "; " : "") + v[i];
    return s;
}

} // namespace

// ─── Arboles de las soluciones en forma cerrada ──────────────────────────────
NodePtr exact_solution_tree(const PDEProblem& prob) {
    auto x = [] { return make_var('x'); };
    auto y = [] { return make_var('y'); };
    auto t = [] { return make_var('t'); };
    auto c = [](Complex v) { return make_erc(v); };
    auto mul = [](NodePtr a, NodePtr b) { return make_binary(NodeType::MUL, std::move(a), std::move(b)); };
    auto add = [](NodePtr a, NodePtr b) { return make_binary(NodeType::ADD, std::move(a), std::move(b)); };
    auto sub = [](NodePtr a, NodePtr b) { return make_binary(NodeType::SUB, std::move(a), std::move(b)); };
    auto un = [](NodeType op, NodePtr a) { return make_unary(op, std::move(a)); };
    const double PI = M_PI;
    bool d2 = (prob.dim == 2);
    switch (prob.type) {
        case PDE::LAPLACE:
            if (!d2) return x();
            return mul(mul(un(NodeType::SIN, mul(make_const_pi(), x())), un(NodeType::SINH, mul(make_const_pi(), y()))),
                       c(1.0 / std::sinh(PI)));
        case PDE::POISSON:
        case PDE::HELMHOLTZ:
            if (!d2) return un(NodeType::SIN, mul(make_const_pi(), x()));
            return mul(un(NodeType::SIN, mul(make_const_pi(), x())), un(NodeType::SIN, mul(make_const_pi(), y())));
        case PDE::SCHRODINGER: {
            NodePtr arg = d2 ? add(x(), y()) : x();
            return un(NodeType::EXP, mul(mul(make_const_i(), c(std::sqrt(prob.k2))), std::move(arg)));
        }
        case PDE::SCHRODINGER_UNSTEADY: {
            // 1D: e^{i pi (x - pi t)}; 2D: e^{i pi (x + y - 2 pi t)}
            NodePtr space = d2 ? add(x(), y()) : x();
            NodePtr arg = sub(std::move(space), mul(c(d2 ? 2.0 * PI : PI), t()));
            return un(NodeType::EXP, mul(mul(make_const_i(), make_const_pi()), std::move(arg)));
        }
        case PDE::HARMONIC_OSCILLATOR: {
            NodePtr r2 = d2 ? add(un(NodeType::SQR, x()), un(NodeType::SQR, y())) : un(NodeType::SQR, x());
            return un(NodeType::EXP, mul(c(-0.5), std::move(r2)));
        }
        case PDE::GINZBURG_LANDAU:
            return un(NodeType::TANH, mul(c(1.0 / std::sqrt(2.0)), x()));
        case PDE::NONLINEAR_POISSON:
        case PDE::LIOUVILLE:
            return make_binary(NodeType::DIV, c(1.0), add(c(1.0), add(un(NodeType::SQR, x()), un(NodeType::SQR, y()))));
        case PDE::NAVIER_STOKES: {
            double Re = 1.0 / prob.k2;
            double lambda = Re / 2.0 - std::sqrt(Re * Re / 4.0 + 4.0 * PI * PI);
            NodePtr wave = mul(un(NodeType::EXP, mul(c(lambda), x())), un(NodeType::SIN, mul(c(2.0 * PI), y())));
            return sub(y(), mul(c(1.0 / (2.0 * PI)), std::move(wave)));
        }
        case PDE::NAVIER_STOKES_UNSTEADY: {
            double lambda = 2.0 * PI * PI * prob.k2;
            return mul(mul(un(NodeType::SIN, mul(make_const_pi(), x())), un(NodeType::SIN, mul(make_const_pi(), y()))),
                       un(NodeType::EXP, mul(c(-lambda), t())));
        }
        default:
            return nullptr; // Airy (Ai no elemental) y problemas con referencia numerica
    }
}

// ─── Oraculo ─────────────────────────────────────────────────────────────────
OracleReport run_oracle(const PDEProblem& prob_in) {
    OracleReport rep;
    PDEProblem prob = prob_in;
    prob.priors = probe_priors(prob);
    rep.problem = problem_label(prob);
    auto add_check = [&](OracleCheck c) {
        if (!c.informative && !c.pass) rep.pass = false;
        rep.checks.push_back(std::move(c));
    };

    if (prob.is_numerical && prob.numerical_truth.empty()) {
        add_check({"referencia numerica disponible", false, false, "falta calcular NumericalSolver::reference"});
        return rep;
    }
    if (prob.is_numerical) {
        size_t bad = 0;
        for (const Complex& v : prob.numerical_truth)
            if (!std::isfinite(v.real()) || !std::isfinite(v.imag())) ++bad;
        if (bad) {
            add_check({"referencia numerica finita", false, false,
                       std::to_string(bad) + " de " + std::to_string(prob.numerical_truth.size()) + " nodos no finitos"});
            return rep;
        }
    }
    UFun uref = [&](double x, double y, double t) { return prob.reference(x, y, t); };
    std::vector<Point> val = prob.validation_points();

    // 1. La referencia satisface la EDP
    {
        double H = 1e-3, he = 2e-3, tol = 1e-5;
        if (prob.is_numerical) {
            double hg = 1.0 / (prob.truth_n - 1);
            H = (prob.dim == 1) ? 5.0 * hg : 2.0 * hg;
            he = H;
            tol = (prob.dim == 1) ? 1e-3 : 1e-2;
        }
        double margin = std::max(0.05, 2.5 * 2.0 * std::max(H, he));
        auto rms_with = [&](double h1, double h2, double& mx_out) {
            double s = 0.0, mx = 0.0; int n = 0;
            for (auto& p : val) {
                if (!inside(prob, p, margin)) continue;
                double r = std::abs(prob.pde_residual_ad(fd_jet(uref, prob, p.x, p.y, p.t, h1, h2), p.x, p.y, p.t));
                if (!std::isfinite(r)) { mx_out = INFINITY; return (double)INFINITY; }
                s += r * r; mx = std::max(mx, r); ++n;
            }
            mx_out = mx;
            return n ? std::sqrt(s / n) : (double)INFINITY;
        };
        double mx = 0.0, mx2 = 0.0;
        double rms = rms_with(H, he, mx);
        bool pass = rms < tol;
        std::string note;
        if (!pass && !prob.is_numerical) {
            // Un umbral absoluto no sirve para derivadas de orden 4 cuyo tamano
            // escala con la amplitud de la solucion. Si el residuo es solo error
            // de truncacion de la DF (orden 2), duplicar el paso lo multiplica
            // por ~4; una referencia que NO resuelve la EDP deja un residuo que
            // no depende del paso.
            double rms2 = rms_with(2.0 * H, 2.0 * he, mx2);
            double ratio = rms2 / rms;
            pass = ratio > 3.0 && ratio < 5.0;
            note = "; con paso doble RMS=" + sci(rms2) + " (razon " + sci(ratio) + ", truncacion O(h^2) => ~4)";
        }
        add_check({"referencia satisface la EDP", pass, false,
                   "RMS|R|=" + sci(rms) + " max|R|=" + sci(mx) + " (tol RMS " + sci(tol) + ", DF sobre la referencia)" + note});
    }

    // 2. La referencia satisface los datos de borde / iniciales
    {
        double worst = 0.0;
        for (auto& p : prob.boundary_points(0)) {
            AD j;
            if (!prob.is_numerical) {
                j = fd_jet(uref, prob, p.x, p.y, p.t, 1e-4, 2e-3);
            } else {
                j.v = uref(p.x, p.y, p.t);
                if (prob.bc_kind == BCKind::IVP_LEFT) {
                    double h = 1.0 / (prob.truth_n - 1);
                    j.dx = (-3.0 * uref(0.0, 0.0, 0.0) + 4.0 * uref(h, 0.0, 0.0) - uref(2.0 * h, 0.0, 0.0)) / (2.0 * h);
                }
            }
            worst = std::max(worst, std::sqrt(prob.boundary_mismatch_sq(j, p)));
        }
        double tol = prob.is_numerical ? 1e-5 : 1e-6;
        add_check({"referencia satisface frontera/condicion inicial", worst < tol, false,
                   "max desajuste=" + sci(worst) + " (tol " + sci(tol) + ")"});
    }

    // 3. El solver, solo con ecuacion + datos de borde, reproduce la referencia
    //    exacta (problema bien planteado: solucion unica alcanzable).
    if (!prob.is_numerical && !prob.is_unsteady && prob.spatial_order <= 2) {
        PDEProblem q = prob; q.is_numerical = true;
        auto sr = NumericalSolver::reference(q);
        double mx = 0.0;
        for (auto& p : val) mx = std::max(mx, std::abs(q.numerical_exact(p.x, p.y, p.t) - prob.exact(p.x, p.y, p.t)));
        double tol = (prob.dim == 1) ? 1e-6 : 1e-4;
        add_check({"bien planteado: el solver reproduce la solucion exacta", sr.ok && mx < tol, false,
                   "max|u_num - u_exacta|=" + sci(mx) + " (tol " + sci(tol) + "), " + sr.method + (sr.ok ? "" : " [NO CONVERGIO]")});
    } else if (prob.is_numerical) {
        add_check({"bien planteado: unicidad de la referencia numerica", true, true,
                   "la referencia ES la solucion numerica del mismo problema (ver [Referencia] en el log)"});
    } else {
        add_check({"bien planteado", true, true,
                   prob.is_unsteady ? "no estacionario: frontera para todo t + condicion inicial (sin solver de verificacion)"
                                    : "orden 4: psi y d(psi)/dn en la frontera (sin solver de verificacion)"});
    }

    // 4. El arbol de la solucion conocida es factible y casi exacto en el algoritmo
    NodePtr tree = exact_solution_tree(prob);
    if (!tree) {
        add_check({"arbol exacto factible", true, true, "sin forma cerrada elemental: se omite"});
    } else {
        std::vector<Point> vbnd = prob.boundary_points(0);
        bool saved = Config::USE_ANSATZ;
        // (a) sin ansatz: el arbol es U
        {
            Config::USE_ANSATZ = false;
            std::vector<std::string> why;
            PIIndividual ind; ind.tree = tree->clone();
            PIIndividual::violation_log = &why;
            ind.evaluate(prob, val, vbnd);
            PIIndividual::violation_log = nullptr;
            double vs = ind.validation_score(prob, val, vbnd);
            Config::USE_ANSATZ = saved;
            bool ok = ind.is_feasible && ind.mse_domain < 1e-10 && ind.mse_boundary < 1e-10;
            add_check({"arbol exacto (sin ansatz) factible y exacto", ok, false,
                       std::string(ind.is_feasible ? "factible" : "INFACTIBLE: " + join(why)) +
                       "  mse_dom=" + sci(ind.mse_domain) + " mse_bnd=" + sci(ind.mse_boundary) + " val_score=" + sci(vs)});
        }
        // (a') representacion hibrida: ansatz activo pero el individuo es U
        //      directo (raw); debe comportarse igual que (a).
        if (saved && Config::HYBRID_REPR && solution_uses_ansatz(prob)) {
            std::vector<std::string> why;
            PIIndividual ind; ind.tree = tree->clone(); ind.raw = true;
            PIIndividual::violation_log = &why;
            ind.evaluate(prob, val, vbnd);
            PIIndividual::violation_log = nullptr;
            double vs = ind.validation_score(prob, val, vbnd);
            bool ok = ind.is_feasible && ind.mse_domain < 1e-10 && ind.mse_boundary < 1e-10 && vs < 1e-10;
            add_check({"arbol exacto (hibrido, U directo) factible y exacto", ok, false,
                       std::string(ind.is_feasible ? "factible" : "INFACTIBLE: " + join(why)) +
                       "  mse_dom=" + sci(ind.mse_domain) + " mse_bnd=" + sci(ind.mse_boundary) + " val_score=" + sci(vs)});
        }
        // (b) con ansatz (1D estacionario): N = (u - L)/B segun el tipo de borde
        if (saved && prob.dim == 1 && !prob.is_unsteady && prob.spatial_order <= 2) {
            NodePtr N;
            if (prob.bc_kind == BCKind::IVP_LEFT) {
                NodePtr lin = make_binary(NodeType::ADD, make_erc(prob.ivp_u0), make_binary(NodeType::MUL, make_erc(prob.ivp_du0), make_var('x')));
                N = make_binary(NodeType::DIV, make_binary(NodeType::SUB, tree->clone(), std::move(lin)),
                                make_binary(NodeType::MUL, make_var('x'), make_var('x')));
            } else {
                Complex u0 = prob.bc(0.0, 0.0, 0.0), u1 = prob.bc(1.0, 0.0, 0.0);
                NodePtr L = make_binary(NodeType::ADD, make_erc(u0), make_binary(NodeType::MUL, make_var('x'), make_erc(u1 - u0)));
                NodePtr B = make_binary(NodeType::MUL, make_var('x'), make_binary(NodeType::SUB, make_var('x'), make_erc(1.0)));
                N = make_binary(NodeType::DIV, make_binary(NodeType::SUB, tree->clone(), std::move(L)), std::move(B));
            }
            std::vector<std::string> why;
            PIIndividual ind; ind.tree = std::move(N);
            PIIndividual::violation_log = &why;
            ind.evaluate(prob, val, vbnd);
            PIIndividual::violation_log = nullptr;
            double vs = ind.validation_score(prob, val, vbnd);
            std::vector<std::string> real_viol;
            for (auto& w : why) if (!kGrammar.count(w)) real_viol.push_back(w);
            bool ok = real_viol.empty() && ind.mse_domain < 1e-8 && vs < 1e-8;
            std::string note = why.empty() ? "factible"
                             : (real_viol.empty() ? "solo gramatica (profundidad de la construccion N=(u-L)/B): " + join(why)
                                                  : "INFACTIBLE: " + join(real_viol));
            add_check({"arbol exacto (con ansatz) factible y exacto", ok, false,
                       note + "  mse_dom=" + sci(ind.mse_domain) + " val_score=" + sci(vs)});
        }
    }

    // 5. Priors: orden detectado vs declarado (debe coincidir) e informe de
    //    que priors detectados en el OPERADOR se cumplen en la SOLUCION.
    {
        const PDEPriors& pr = prob.priors;
        add_check({"orden de derivacion detectado = declarado", pr.max_deriv_order == prob.spatial_order, false,
                   "detectado " + std::to_string(pr.max_deriv_order) + ", declarado " + std::to_string(prob.spatial_order)});

        std::vector<std::string> det;
        auto flag = [&](bool b, const char* n) { if (b) det.push_back(n); };
        flag(pr.autonomous_x, "autonomo_x"); flag(pr.autonomous_y, "autonomo_y"); flag(pr.autonomous_t, "autonomo_t");
        flag(pr.pole_at_origin, "polo_origen"); flag(pr.even_parity_x, "paridad_x"); flag(pr.scale_invariant, "escala");
        flag(pr.rotation_invariant, "rotacion"); flag(pr.damped_oscillator, "amortiguado"); flag(pr.exponential_modes, "modos_exp");
        add_check({"priors detectados (operador)", true, true, det.empty() ? "ninguno" : join(det)});

        auto sym_err = [&](bool in_x) {
            double mx = 0.0;
            for (auto& p : val) {
                Complex a = uref(p.x, p.y, p.t);
                Complex b = in_x ? uref(1.0 - p.x, p.y, p.t) : uref(p.x, 1.0 - p.y, p.t);
                mx = std::max(mx, std::abs(a - b));
            }
            return mx;
        };
        if (pr.mirror_symmetric_x)
            add_check({"simetria x<->1-x (operador+borde) -> solucion?", true, true, "max|u(x)-u(1-x)|=" + sci(sym_err(true))});
        if (pr.mirror_symmetric_y)
            add_check({"simetria y<->1-y (operador+borde) -> solucion?", true, true, "max|u(y)-u(1-y)|=" + sci(sym_err(false))});
        if (prob.dim == 2 && (pr.additive_separable || pr.multiplicative_separable || pr.modal_sum_separable)) {
            double add_e = 0.0, mul_e = 0.0;
            for (double a : {0.2, 0.35}) for (double b : {0.6, 0.8}) {
                double t = prob.is_unsteady ? 0.5 : 0.0;
                Complex u11 = uref(a, a, t), u22 = uref(b, b, t), u12 = uref(a, b, t), u21 = uref(b, a, t);
                add_e = std::max(add_e, std::abs(u11 + u22 - u12 - u21));
                mul_e = std::max(mul_e, std::abs(u11 * u22 - u12 * u21));
            }
            std::string which = pr.additive_separable ? "aditiva" : pr.multiplicative_separable ? "multiplicativa" : "suma modal";
            add_check({"separabilidad del operador (" + which + ") -> solucion?", true, true,
                       "identidad aditiva=" + sci(add_e) + " multiplicativa=" + sci(mul_e) + " (0 => la solucion es separable)"});
        }
    }
    return rep;
}

void print_oracle_report(const OracleReport& r) {
    std::cout << (r.pass ? "[ORACULO OK]    " : "[ORACULO FALLA] ") << r.problem << "\n";
    for (auto& c : r.checks) {
        const char* tag = c.informative ? "   info " : (c.pass ? "   ok   " : "   FALLA");
        std::cout << tag << "  " << c.name << ": " << c.detail << "\n";
    }
}
