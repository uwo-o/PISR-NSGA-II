#include "numerical_solver.hpp"
#include <algorithm>
#include <cmath>

namespace NumericalSolver {
namespace {

using CVec = std::vector<Complex>;

double max_abs(const CVec& v) {
    double m = 0.0;
    for (auto& z : v) m = std::max(m, std::abs(z));
    return m;
}

// ─── 1D, problema de contorno de Dirichlet: Newton sobre diferencias finitas ──
// R_i = residuo del operador con u'' y u' por diferencias centradas (orden 2).
// Jacobiano tridiagonal por diferencias finitas sobre R_i, resuelto con Thomas.
// Newton amortiguado (busqueda lineal por mitades sobre max|R|).
struct BVP1D {
    const PDEProblem& prob;
    int N; double h;
    Complex residual(const CVec& u, int i) const {
        AD j;
        j.v = u[i];
        j.dx = (u[i + 1] - u[i - 1]) / (2.0 * h);
        j.dxx = (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (h * h);
        return prob.pde_residual_ad(j, i * h, 0.0, 0.0);
    }
    double residual_norm(const CVec& u) const {
        double m = 0.0;
        for (int i = 1; i < N - 1; ++i) {
            double r = std::abs(residual(u, i));
            if (!std::isfinite(r)) return INFINITY;
            m = std::max(m, r);
        }
        return m;
    }
};

CVec solve_bvp_1d(const PDEProblem& prob, int N, bool& ok, double& res_out) {
    BVP1D B{prob, N, 1.0 / (N - 1)};
    Complex ua = prob.bc(0.0, 0.0, 0.0), ub = prob.bc(1.0, 0.0, 0.0);
    CVec u(N);
    for (int i = 0; i < N; ++i) u[i] = ua + (ub - ua) * (i * B.h);
    ok = false;
    double rn = B.residual_norm(u);
    for (int it = 0; it < 80 && std::isfinite(rn); ++it) {
        int M = N - 2;
        CVec a(M), b(M), c(M), d(M);
        for (int k = 0; k < M; ++k) {
            int i = k + 1;
            Complex R0 = B.residual(u, i);
            d[k] = -R0;
            for (int s = -1; s <= 1; ++s) {
                int jj = i + s;
                if (jj == 0 || jj == N - 1) continue; // valores de borde fijos
                double del = 1e-7 * (1.0 + std::abs(u[jj]));
                Complex keep = u[jj];
                u[jj] = keep + del;
                Complex dR = (B.residual(u, i) - R0) / del;
                u[jj] = keep;
                if (s == -1) a[k] = dR; else if (s == 0) b[k] = dR; else c[k] = dR;
            }
        }
        // Thomas
        for (int k = 1; k < M; ++k) {
            Complex w = a[k] / b[k - 1];
            b[k] -= w * c[k - 1];
            d[k] -= w * d[k - 1];
        }
        CVec du(M);
        du[M - 1] = d[M - 1] / b[M - 1];
        for (int k = M - 2; k >= 0; --k) du[k] = (d[k] - c[k] * du[k + 1]) / b[k];

        double lam = 1.0, rn_new = INFINITY;
        CVec trial = u;
        while (lam > 1e-4) {
            for (int k = 0; k < M; ++k) trial[k + 1] = u[k + 1] + lam * du[k];
            rn_new = B.residual_norm(trial);
            if (rn_new < rn || rn < 1e-12) break;
            lam *= 0.5;
        }
        u = trial;
        double step = lam * max_abs(du);
        rn = rn_new;
        if (step < 1e-13 * (1.0 + max_abs(u))) { ok = true; break; }
    }
    res_out = rn;
    if (!std::isfinite(rn)) ok = false;
    return u;
}

// ─── 1D, problema de valor inicial: RK4 desde una serie de Taylor ─────────────
// u'' se despeja del residuo, que es afin en u'' para todos los operadores de
// la suite: w = -R(w=0) / (R(w=1) - R(w=0)). El origen puede ser singular
// (Lane-Emden: 2u'/x), asi que se arranca en x0 > 0 con la expansion de Taylor
// consistente u = u0 + u0' x + w x^2/2, u' = u0' + w x, con w resuelto por
// secante sobre g(w) = w - u''(x; w) (sin conocer la solucion). El punto fijo
// w <- u''(x; w) no sirve: en Lane-Emden es w <- -2w - u^3 y diverge.
Complex second_derivative(const PDEProblem& prob, double x, Complex u, Complex du) {
    AD j; j.v = u; j.dx = du;
    j.dxx = 0.0; Complex R0 = prob.pde_residual_ad(j, x, 0.0, 0.0);
    j.dxx = 1.0; Complex R1 = prob.pde_residual_ad(j, x, 0.0, 0.0);
    return -R0 / (R1 - R0);
}

CVec solve_ivp_1d(const PDEProblem& prob, int N, int substeps, bool& ok) {
    double hg = 1.0 / (N - 1);
    Complex u0 = prob.ivp_u0, du0 = prob.ivp_du0;
    auto taylor = [&](double x, Complex& u, Complex& du) {
        auto g = [&](Complex w) {
            return w - second_derivative(prob, x, u0 + du0 * x + 0.5 * w * x * x, du0 + w * x);
        };
        Complex w = 0.0;
        for (int k = 0; k < 50; ++k) {
            Complex g0 = g(w), g1 = g(w + 1.0);
            if (std::abs(g1 - g0) == 0.0) break;
            Complex step = g0 / (g1 - g0);
            w -= step;
            if (std::abs(step) < 1e-15 * (1.0 + std::abs(w))) break;
        }
        u = u0 + du0 * x + 0.5 * w * x * x;
        du = du0 + w * x;
    };
    // NaN por defecto: si RK4 se corta, el resto NO queda en 0 (u = 0 puede
    // ser solucion de la EDP, p. ej. Lane-Emden, y engañaria al oraculo).
    CVec out(N, Complex(std::nan(""), 0.0));
    out[0] = u0;
    const int i_start = 2; // primeros nodos por Taylor (x <= 2h)
    Complex u, du;
    for (int i = 1; i <= i_start; ++i) { taylor(i * hg, u, du); out[i] = u; }
    double hs = hg / substeps;
    auto f = [&](double x, Complex uu, Complex vv, Complex& fu, Complex& fv) {
        fu = vv; fv = second_derivative(prob, x, uu, vv);
    };
    ok = true;
    for (int i = i_start; i < N - 1; ++i) {
        for (int s = 0; s < substeps; ++s) {
            double x = i * hg + s * hs;
            Complex k1u, k1v, k2u, k2v, k3u, k3v, k4u, k4v;
            f(x, u, du, k1u, k1v);
            f(x + 0.5 * hs, u + 0.5 * hs * k1u, du + 0.5 * hs * k1v, k2u, k2v);
            f(x + 0.5 * hs, u + 0.5 * hs * k2u, du + 0.5 * hs * k2v, k3u, k3v);
            f(x + hs, u + hs * k3u, du + hs * k3v, k4u, k4v);
            u += (hs / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
            du += (hs / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        }
        out[i + 1] = u;
        if (!std::isfinite(u.real()) || !std::isfinite(u.imag())) { ok = false; break; }
    }
    return out;
}

// ─── 2D, Dirichlet en [0,1]^2: SOR no lineal (Newton por punto) ──────────────
CVec solve_bvp_2d(const PDEProblem& prob, int N, bool& ok, double& res_out) {
    double h = 1.0 / (N - 1);
    CVec u(N * N);
    auto at = [&](int i, int j) -> Complex& { return u[i * N + j]; };
    // Borde: datos. Interior: interpolacion transfinita de los datos (arranque).
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
        double x = i * h, y = j * h;
        Complex g0 = prob.bc(0.0, y), g1 = prob.bc(1.0, y), f0 = prob.bc(x, 0.0), f1 = prob.bc(x, 1.0);
        Complex c00 = prob.bc(0.0, 0.0), c10 = prob.bc(1.0, 0.0), c01 = prob.bc(0.0, 1.0), c11 = prob.bc(1.0, 1.0);
        at(i, j) = (1 - x) * g0 + x * g1 + (1 - y) * f0 + y * f1
                 - (1 - x) * (1 - y) * c00 - x * (1 - y) * c10 - (1 - x) * y * c01 - x * y * c11;
    }
    auto local_residual = [&](int i, int j, Complex uc) {
        AD jt;
        Complex uE = at(i + 1, j), uW = at(i - 1, j), uN = at(i, j + 1), uS = at(i, j - 1);
        jt.v = uc;
        jt.dx = (uE - uW) / (2.0 * h); jt.dy = (uN - uS) / (2.0 * h);
        jt.dxx = (uE - 2.0 * uc + uW) / (h * h); jt.dyy = (uN - 2.0 * uc + uS) / (h * h);
        jt.dxy = (at(i + 1, j + 1) - at(i + 1, j - 1) - at(i - 1, j + 1) + at(i - 1, j - 1)) / (4.0 * h * h);
        return prob.pde_residual_ad(jt, i * h, j * h, 0.0);
    };
    double omega = 2.0 / (1.0 + std::sin(M_PI * h));
    ok = false;
    double scale = 1.0 + max_abs(u);
    for (int sweep = 0; sweep < 40000; ++sweep) {
        double max_step = 0.0;
        for (int i = 1; i < N - 1; ++i) for (int j = 1; j < N - 1; ++j) {
            Complex uc = at(i, j);
            Complex F = local_residual(i, j, uc);
            double del = 1e-7 * (1.0 + std::abs(uc));
            Complex dF = (local_residual(i, j, uc + del) - F) / del;
            Complex step = -omega * F / dF;
            at(i, j) = uc + step;
            max_step = std::max(max_step, std::abs(step));
        }
        if (!std::isfinite(max_step)) break;
        if (max_step < 1e-14 * scale) { ok = true; break; }
        if (sweep % 200 == 199 && max_step > 1e6 * scale) break; // divergencia
    }
    double rn = 0.0;
    for (int i = 1; i < N - 1; ++i) for (int j = 1; j < N - 1; ++j) rn = std::max(rn, std::abs(local_residual(i, j, at(i, j))));
    res_out = rn;
    return u;
}

} // namespace

Report reference(PDEProblem& prob) {
    Report rep;
    prob.numerical_truth.clear(); prob.truth_n = 0;
    if (prob.is_unsteady) { rep.note = "no estacionario: sin referencia numerica"; return rep; }

    if (prob.dim == 1 && prob.bc_kind == BCKind::DIRICHLET) {
        const int N1 = 2001, N2 = 2 * N1 - 1;
        bool ok1, ok2; double r1, r2;
        CVec u1 = solve_bvp_1d(prob, N1, ok1, r1);
        CVec u2 = solve_bvp_1d(prob, N2, ok2, r2);
        CVec ext(N1);
        double diff = 0.0;
        for (int i = 0; i < N1; ++i) {
            ext[i] = (4.0 * u2[2 * i] - u1[i]) / 3.0; // Richardson: orden 2 -> 4
            diff = std::max(diff, std::abs(u2[2 * i] - u1[i]));
        }
        prob.numerical_truth = ext; prob.truth_n = N1;
        rep = {ok1 && ok2, "Newton-DF (orden 2) + Richardson", N1, std::max(r1, r2), diff, ""};
        return rep;
    }
    if (prob.dim == 1 && prob.bc_kind == BCKind::IVP_LEFT) {
        const int N = 2001;
        bool ok1, ok2;
        CVec u1 = solve_ivp_1d(prob, N, 20, ok1);
        CVec u2 = solve_ivp_1d(prob, N, 40, ok2);
        double diff = 0.0;
        for (int i = 0; i < N; ++i) diff = std::max(diff, std::abs(u2[i] - u1[i]));
        prob.numerical_truth = u2; prob.truth_n = N;
        rep = {ok1 && ok2, "Taylor + RK4", N, 0.0, diff, ""};
        return rep;
    }
    if (prob.dim == 2 && prob.bc_kind == BCKind::DIRICHLET) {
        const int N1 = 65, N2 = 2 * N1 - 1;
        bool ok1, ok2; double r1, r2;
        CVec u1 = solve_bvp_2d(prob, N1, ok1, r1);
        CVec u2 = solve_bvp_2d(prob, N2, ok2, r2);
        CVec ext(N1 * N1);
        double diff = 0.0;
        for (int i = 0; i < N1; ++i) for (int j = 0; j < N1; ++j) {
            Complex fine = u2[(2 * i) * N2 + 2 * j], coarse = u1[i * N1 + j];
            ext[i * N1 + j] = (4.0 * fine - coarse) / 3.0;
            diff = std::max(diff, std::abs(fine - coarse));
        }
        prob.numerical_truth = ext; prob.truth_n = N1;
        rep = {ok1 && ok2, "SOR no lineal (orden 2) + Richardson", N1, std::max(r1, r2), diff, ""};
        return rep;
    }
    rep.note = "tipo de condicion de borde sin solver de referencia";
    return rep;
}

std::vector<Complex> solve(const PDEProblem& prob, int resolution) {
    if (prob.is_unsteady || resolution < 3) return {};
    bool ok; double r;
    if (prob.dim == 1 && prob.bc_kind == BCKind::DIRICHLET) return solve_bvp_1d(prob, resolution, ok, r);
    if (prob.dim == 1 && prob.bc_kind == BCKind::IVP_LEFT) return solve_ivp_1d(prob, resolution, 20, ok);
    if (prob.dim == 2 && prob.bc_kind == BCKind::DIRICHLET) return solve_bvp_2d(prob, resolution, ok, r);
    return {};
}

} // namespace NumericalSolver
