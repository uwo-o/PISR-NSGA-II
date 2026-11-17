// =============================================================================
// tests/gradient_test.cpp — gradiente de las constantes (constant_gradient)
//
// 1) Campo REAL con representacion compleja: Poisson 1D, arbol
//    U = c sin(pi x) e^{i theta}  ->  Re(U) = c cos(theta) sin(pi x).
//    Derivadas analiticas de la perdida (sin ansatz).
// 2) Campo COMPLEJO: Schrodinger 1D (k^2=4), U = c e^{i theta x}. Derivadas
//    analiticas de residuo + frontera.
// 3) Con ansatz de valor inicial (Lane-Emden) y factor de fase: consistencia
//    con la extrapolacion de Richardson de las diferencias centradas.
// 4) Regresion: el paso complejo anterior (Im f(c+ih)/h) aplicado al caso 1
//    da un gradiente erroneo; se imprime para documentar el bug.
// Sale con codigo != 0 si algun chequeo falla.
// =============================================================================
#include "pi_solver.hpp"
#include <cmath>
#include <cstdio>

int last_special_choice = -1;

static int failures = 0;
static void check(const char* what, double got, double want, double rtol) {
    double err = std::abs(got - want) / std::max(1.0, std::abs(want));
    bool ok = err < rtol;
    if (!ok) ++failures;
    std::printf("  %-48s calculado=% .10e  esperado=% .10e  err_rel=%.1e  %s\n",
                what, got, want, err, ok ? "ok" : "FALLA");
}

int main() {
    const double PI = M_PI;
    bool saved = Config::USE_ANSATZ;

    // ── 1) Campo real, arbol con e^{i theta} ─────────────────────────────────
    {
        std::printf("[1] Poisson 1D, campo real, U = c sin(pi x) e^{i theta} (sin ansatz)\n");
        Config::USE_ANSATZ = false;
        PDEProblem prob = make_poisson(1);
        auto dom = prob.validation_points();
        auto bnd = prob.boundary_points(0);
        double c = 0.7, th = 0.4;
        NodePtr tree = make_binary(NodeType::MUL,
            make_binary(NodeType::MUL, make_erc(c), make_unary(NodeType::SIN, make_binary(NodeType::MUL, make_const_pi(), make_var('x')))),
            make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_const_i(), make_erc(th))));
        // L = pi^4 S (1 - c cos th)^2 + frontera(~0), S = media de sin^2(pi x_i)
        double S = 0.0;
        for (auto& p : dom) S += std::pow(std::sin(PI * p.x), 2);
        S /= dom.size();
        double k = 1.0 - c * std::cos(th);
        double dLdc = -2.0 * std::pow(PI, 4) * S * std::cos(th) * k;
        double dLdth = 2.0 * std::pow(PI, 4) * S * c * std::sin(th) * k;
        auto g = constant_gradient(prob, tree.get(), dom, bnd, 1e-6);
        check("dL/dc", g[0], dLdc, 1e-6);
        check("dL/dtheta", g[1], dLdth, 1e-6);

        // 4) Regresion: formula del paso complejo anterior sobre el mismo arbol
        //    (sin proyeccion a parte real, como era antes).
        std::vector<Complex*> ercs; tree->collect_ercs(ercs);
        double h = 1e-8, grad_old = 0.0;
        for (auto& p : dom) {
            AD a = tree->ad_eval_t(p.x, 0.0, 0.0, 1);
            double r = prob.pde_residual_ad(a, p.x, 0.0, 0.0).real();
            Complex orig = *ercs[0];
            *ercs[0] = orig + Complex(0.0, h);
            AD b = tree->ad_eval_t(p.x, 0.0, 0.0, 1);
            double rim = prob.pde_residual_ad(b, p.x, 0.0, 0.0).imag();
            *ercs[0] = orig;
            grad_old += r * rim / h;
        }
        grad_old = 2.0 * grad_old / dom.size();
        std::printf("  [regresion] paso complejo anterior: dL/dc = % .3e (analitico % .3e) -> %s\n",
                    grad_old, dLdc, std::abs(grad_old - dLdc) > 1e-2 * std::abs(dLdc) ? "erroneo, como se esperaba" : "coincide (?)");
    }

    // ── 2) Campo complejo: Schrodinger 1D ────────────────────────────────────
    {
        std::printf("[2] Schrodinger 1D (campo complejo, k^2=4), U = c e^{i theta x} (sin ansatz)\n");
        Config::USE_ANSATZ = false;
        PDEProblem prob = make_schrodinger(1);
        auto dom = prob.validation_points();
        auto bnd = prob.boundary_points(0); // x=0 y x=1
        double c = 0.9, th = 1.7;
        NodePtr tree = make_binary(NodeType::MUL, make_erc(c),
            make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_const_i(), make_binary(NodeType::MUL, make_erc(th), make_var('x')))));
        // dominio: |R|^2 = c^2 (4 - th^2)^2
        // frontera: 1/2 (|c - 1|^2 + |c e^{i th} - e^{2i}|^2) = 1/2((c-1)^2 + c^2 + 1 - 2c cos(th-2))
        double dLdc = 2.0 * c * std::pow(4.0 - th * th, 2) + (c - 1.0) + c - std::cos(th - 2.0);
        double dLdth = -4.0 * c * c * th * (4.0 - th * th) + c * std::sin(th - 2.0);
        auto g = constant_gradient(prob, tree.get(), dom, bnd, 1e-6);
        check("dL/dc", g[0], dLdc, 1e-6);
        check("dL/dtheta", g[1], dLdth, 1e-6);
    }

    // ── 3) Ansatz de valor inicial + factor de fase: Richardson ──────────────
    {
        std::printf("[3] Lane-Emden (IVP, ansatz U = 1 + x^2 N), N con factor e^{i theta}\n");
        Config::USE_ANSATZ = true;
        PDEProblem prob = make_lane_emden();
        auto dom = prob.validation_points();
        auto bnd = prob.boundary_points(0);
        NodePtr tree = make_binary(NodeType::MUL,
            make_binary(NodeType::MUL, make_erc(-0.15), make_unary(NodeType::COS, make_binary(NodeType::MUL, make_erc(0.8), make_var('x')))),
            make_unary(NodeType::EXP, make_binary(NodeType::MUL, make_const_i(), make_erc(0.3))));
        auto g1 = constant_gradient(prob, tree.get(), dom, bnd, 1e-5);
        auto g2 = constant_gradient(prob, tree.get(), dom, bnd, 5e-6);
        for (size_t k = 0; k < g1.size(); ++k) {
            double rich = (4.0 * g2[k] - g1[k]) / 3.0;
            char name[64]; std::snprintf(name, sizeof name, "dL/dc_%zu vs Richardson", k);
            check(name, g2[k], rich, 1e-6);
        }
    }

    Config::USE_ANSATZ = saved;
    std::printf(failures ? "\nGRADIENTE: %d FALLAS\n" : "\nGRADIENTE: todo ok\n", failures);
    return failures ? 1 : 0;
}
