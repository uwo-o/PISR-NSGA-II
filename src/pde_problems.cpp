// =============================================================================
// pde_problems.cpp  —  Implementación de Laplace, Poisson y Helmholtz
// =============================================================================

#include "pde_problems.hpp"
#include <cmath>
#include <stdexcept>

static constexpr double PI = M_PI;

// ─── domain_points: rejilla uniforme interior (x,y) ∈ (0,1)² ─────────────────
std::vector<Point> PDEProblem::domain_points(int n) const {
    std::vector<Point> pts;
    if (dim == 1) {
        pts.reserve(n);
        for (int i = 1; i <= n; ++i)
            pts.push_back({(double)i / (n + 1), 0.0});
    } else {
        int sq = (int)std::ceil(std::sqrt((double)n));
        pts.reserve(sq * sq);
        for (int i = 1; i <= sq; ++i)
            for (int j = 1; j <= sq; ++j)
                pts.push_back({(double)i / (sq + 1), (double)j / (sq + 1)});
    }
    return pts;
}

// ─── boundary_points: puntos en los 4 lados de ∂Ω ────────────────────────────
std::vector<Point> PDEProblem::boundary_points(int n) const {
    std::vector<Point> pts;
    if (dim == 1) {
        pts.push_back({0.0, 0.0});
        pts.push_back({1.0, 0.0});
    } else {
        int per_side = std::max(2, n / 4);
        pts.reserve(per_side * 4);
        for (int i = 0; i < per_side; ++i) {
            double t = (double)i / (per_side - 1);
            pts.push_back({t, 0.0}); // sur
            pts.push_back({t, 1.0}); // norte
            pts.push_back({0.0, t}); // oeste
            pts.push_back({1.0, t}); // este
        }
    }
    return pts;
}

// ─── Solución exacta ──────────────────────────────────────────────────────────
Complex PDEProblem::exact(double x, double y) const {
    if (is_numerical && !numerical_truth.empty()) {
        return numerical_exact(x, y);
    }

    if (dim == 1) {
        switch (type) {
            case PDE::LAPLACE:   return x;
            case PDE::POISSON:   
            case PDE::HELMHOLTZ: return std::sin(PI * x);
            case PDE::SCHRODINGER: return std::exp(I_COMPLEX * PI * x); 
            case PDE::AIRY:
            case PDE::HARMONIC_OSCILLATOR:
            case PDE::GROSS_PITAEVSKII: 
                return (x < 0.1) ? 1.0 : 0.0; // u(0)=1, u(1)=0
            case PDE::FISHER:
                return (x < 0.1) ? 0.1 : 0.8;
            case PDE::DUFFING:
                return (x < 0.1) ? 1.0 : -0.5;
            case PDE::THOMAS_FERMI:
                return (x < 0.1) ? 1.0 : 0.2;
            default: return 0.0;
        }
    } else {
        switch (type) {
            case PDE::LAPLACE:
                return std::sin(PI*x) * std::sinh(PI*y) / std::sinh(PI);
            case PDE::POISSON:
            case PDE::HELMHOLTZ:
                return std::sin(PI*x) * std::sin(PI*y);
            case PDE::SCHRODINGER:
                return std::exp(I_COMPLEX * (PI*x + PI*y));
            case PDE::AIRY:
            case PDE::HARMONIC_OSCILLATOR:
            case PDE::GROSS_PITAEVSKII:
                return (x < 0.1 || y < 0.1) ? 1.0 : 0.0;
            case PDE::NONLINEAR_POISSON:
            case PDE::LIOUVILLE:
                return 1.0 / (1.0 + x*x + y*y);
            case PDE::SINE_GORDON:
                return std::sin(PI*x) * std::sin(PI*y);
            case PDE::FISHER:
                return 0.1 + 0.35 * (x + y);
            case PDE::DUFFING:
                return 1.0 - 0.75 * (x + y);
            case PDE::THOMAS_FERMI:
                return 1.0 - 0.4 * (x + y);
            case PDE::NAVIER_STOKES: {
                double Re = 1.0 / k2;
                double lambda = Re / 2.0 - std::sqrt(Re * Re / 4.0 + 4.0 * PI * PI);
                double val = y - std::exp(lambda * x) * std::sin(2.0 * PI * y) / (2.0 * PI * Re);
                return Complex(val, 0.0);
            }
            case PDE::NAVIER_STOKES_UNSTEADY: {
                // Taylor-Green Vortex (componente u)
                // u(x,y,t) = sin(x)cos(y)exp(-2*nu*t)
                // Usamos PI para escala
                double nu = k2;
                double val = std::sin(PI * x) * std::cos(PI * y) * std::exp(-2.0 * PI * PI * nu * 0.0); // t=0 por ahora en exact()
                return Complex(val, 0.0);
            }
            default: return 0.0;
        }
    }
}

Complex PDEProblem::numerical_exact(double x, double y) const {
    if (numerical_truth.empty()) return 0.0;
    if (dim == 1) {
        int N = (int)numerical_truth.size();
        double h = 1.0 / (N - 1);
        int i = (int)(x / h);
        if (i < 0) return numerical_truth[0];
        if (i >= N - 1) return numerical_truth[N-1];
        double t = (x - i * h) / h;
        return (1.0 - t) * numerical_truth[i] + t * numerical_truth[i+1];
    } else {
        int N = (int)std::sqrt(numerical_truth.size());
        double h = 1.0 / (N - 1);
        int i = (int)(x / h);
        int j = (int)(y / h);
        if (i < 0) i = 0; if (i >= N - 1) i = N - 2;
        if (j < 0) j = 0; if (j >= N - 1) j = N - 2;
        double tx = (x - i * h) / h;
        double ty = (y - j * h) / h;
        Complex c00 = numerical_truth[i * N + j];
        Complex c10 = numerical_truth[(i+1) * N + j];
        Complex c01 = numerical_truth[i * N + (j+1)];
        Complex c11 = numerical_truth[(i+1) * N + (j+1)];
        return (1.0 - tx) * (1.0 - ty) * c00 + tx * (1.0 - ty) * c10 +
               (1.0 - tx) * ty * c01 + tx * ty * c11;
    }
}

// ─── Derivada segunda u'' = F(x, u) para solvers numéricos ────────────────────
Complex PDEProblem::pde_second_derivative(double x, Complex u) const {
    switch (type) {
        case PDE::LAPLACE:   return 0.0;
        case PDE::POISSON:   return source(x, 0.0);
        case PDE::HELMHOLTZ: return source(x, 0.0) - k2 * u;
        case PDE::SCHRODINGER: return -(PI * PI) * u;
        case PDE::AIRY:      return x * u;
        case PDE::HARMONIC_OSCILLATOR:
            // Consistente con residuo: u'' = (x²-E)·u = (x²-1)·u  [E=1]
            return (x * x - 1.0) * u;
        case PDE::FISHER: return -u * (1.0 - u);
        case PDE::DUFFING: return -u - u * u * u;
        case PDE::THOMAS_FERMI: return u * u / (x + 0.5);
        default: return 0.0;
    }
}

// ─── Término fuente f(x,y): ∇²u + k²u = f ────────────────────────────────────
Complex PDEProblem::source(double x, double y) const {
    if (dim == 1) {
        switch (type) {
            case PDE::LAPLACE: return 0.0;
            case PDE::POISSON: return -PI * PI * std::sin(PI * x);
            case PDE::HELMHOLTZ: return (k2 - PI * PI) * std::sin(PI * x);
            case PDE::SCHRODINGER: return 0.0; 
            default: return 0.0;
        }
    } else {
        switch (type) {
            case PDE::LAPLACE: return 0.0;
            case PDE::POISSON:
                return -2.0 * PI * PI * std::sin(PI*x) * std::sin(PI*y);
            case PDE::HELMHOLTZ:
                return (k2 - 2.0*PI*PI) * std::sin(PI*x) * std::sin(PI*y);
            case PDE::NONLINEAR_POISSON: {
                double r2 = x*x + y*y;
                double den = 1.0 + r2;
                Complex laplacian = (8.0 * r2) / (den * den * den) - 4.0 / (den * den); 
                Complex u_val = 1.0 / den;
                return laplacian + u_val * u_val;
            }
            case PDE::LIOUVILLE: {
                double r2 = x*x + y*y;
                double den = 1.0 + r2;
                double laplacian = (8.0 * r2) / (den * den * den) - 4.0 / (den * den);
                double u_val = 1.0 / den;
                return laplacian + std::exp(u_val);
            }
            case PDE::SINE_GORDON: {
                double u = std::sin(PI*x) * std::sin(PI*y);
                return -2.0 * PI * PI * u - std::sin(u);
            }
            case PDE::NAVIER_STOKES: return 0.0;
            default: return 0.0;
        }
    }
}

// ─── Condición de frontera Dirichlet ───────────────────────────────────────────────
Complex PDEProblem::bc(double x, double y) const {
    if (type == PDE::AIRY) {
        // Función de Airy: Ai(0) ≈ 0.3550, Ai(1) ≈ 0.1353
        // Usamos los valores exactos para las BCs
        if (x < 0.01) return Complex(0.3550, 0.0);
        return Complex(0.1353, 0.0);
    }
    if (type == PDE::HARMONIC_OSCILLATOR) {
        // Estado base: ψ_0(x) = C*exp(-x²/2)
        // ψ_0(0) = C = 1 (normalizado), ψ_0(x→∞) → 0
        // En [0,L]: usamos valores del estado base Gaussiano
        if (dim == 1) {
            if (x < 0.01) return Complex(1.0, 0.0);
            return Complex(std::exp(-0.5), 0.0);
        } else {
            double v = std::exp(-0.5 * (x*x + y*y));
            return Complex(v, 0.0);
        }
    }
    if (type == PDE::FISHER) {
        if (dim == 1) {
            if (x < 0.01) return Complex(0.1, 0.0);
            return Complex(0.8, 0.0);
        } else {
            return Complex(0.1 + 0.35 * (x + y), 0.0);
        }
    }
    if (type == PDE::DUFFING) {
        if (dim == 1) {
            if (x < 0.01) return Complex(1.0, 0.0);
            return Complex(-0.5, 0.0);
        } else {
            return Complex(1.0 - 0.75 * (x + y), 0.0);
        }
    }
    if (type == PDE::THOMAS_FERMI) {
        if (dim == 1) {
            if (x < 0.01) return Complex(1.0, 0.0);
            return Complex(0.2, 0.0);
        } else {
            return Complex(1.0 - 0.4 * (x + y), 0.0);
        }
    }
    if (dim == 1) {
        switch (type) {
            case PDE::LAPLACE:   return x;
            case PDE::POISSON:   
            case PDE::HELMHOLTZ: return std::sin(PI * x);
            case PDE::SCHRODINGER: return std::exp(I_COMPLEX * PI * x);
            default: return 0.0;
        }
    } else {
        switch (type) {
            case PDE::LAPLACE: return std::sin(PI*x) * std::sinh(PI*y) / std::sinh(PI);
            case PDE::POISSON:
            case PDE::HELMHOLTZ: return std::sin(PI*x) * std::sin(PI*y);
            case PDE::SCHRODINGER: return std::exp(I_COMPLEX * (PI*x + PI*y));
            case PDE::NONLINEAR_POISSON:
            case PDE::LIOUVILLE: return 1.0 / (1.0 + x*x + y*y);
            case PDE::SINE_GORDON: return std::sin(PI*x) * std::sin(PI*y);
            case PDE::NAVIER_STOKES: return exact(x, y);
            default: return 0.0;
        }
    }
}

Complex PDEProblem::pde_residual_ad(const AD& ad, double x, double y) const {
    Complex laplacian = (dim == 1) ? ad.dxx : (ad.dxx + ad.dyy);
    Complex u = ad.v;

    switch (type) {
        case PDE::LAPLACE: return laplacian;
        case PDE::POISSON: return laplacian - source(x, y);
        case PDE::HELMHOLTZ: return laplacian + k2 * u - source(x, y);
        case PDE::SCHRODINGER: {
            // Ecuación de Schrödinger adimensional: -∇²ψ + Vψ = Eψ
            // Para espacio libre V=0, E=k². Residuo: ∇²ψ + Eψ = 0
            double E = (dim == 1) ? (PI * PI) : (2.0 * PI * PI); 
            return laplacian + E * u;
        }
        case PDE::NONLINEAR_POISSON: return laplacian + u * u - source(x, y);
        case PDE::LIOUVILLE: return laplacian + std::exp(u) - source(x, y);
        case PDE::AIRY: {
            double var = (dim == 1) ? x : (x + y);
            return laplacian - var * u;
        }
        case PDE::HARMONIC_OSCILLATOR: {
        double E = (dim == 1) ? 1.0 : 2.0;
        double V = (dim == 1) ? (x*x) : (x*x + y*y);
        return laplacian - V * u + E * u;
    }
        case PDE::GROSS_PITAEVSKII: {
            double g = 1.0;
            double mu = 1.0;
            return -laplacian + (std::norm(u) * g - mu) * u;
        }
        case PDE::SINE_GORDON: return laplacian - std::sin(u.real()) - source(x, y);
        case PDE::FISHER: return laplacian + u * (1.0 - u);
        case PDE::DUFFING: return laplacian + u + u * u * u;
        case PDE::THOMAS_FERMI: return laplacian - u * u / (x + y + 0.5);
        case PDE::NAVIER_STOKES: return 0.0;
        default: return 0.0;
    }
}


// ─── Nombre ───────────────────────────────────────────────────────────────────
std::string PDEProblem::name() const { return pde_name(type); }

// ─── Fabricación ─────────────────────────────────────────────────────────────
PDEProblem make_laplace(int dim) {
    PDEProblem p; p.type = PDE::LAPLACE; p.dim = dim;
    p.dim_u = Units::None; p.dim_x = Units::None; p.dim_y = Units::None;
    return p;
}
PDEProblem make_poisson(int dim) {
    PDEProblem p; p.type = PDE::POISSON; p.dim = dim;
    p.dim_u = Units::None; p.dim_x = Units::None; p.dim_y = Units::None;
    return p;
}
PDEProblem make_helmholtz(int dim, double k) {
    PDEProblem p; p.type = PDE::HELMHOLTZ; p.dim = dim; p.k2 = k*k;
    p.dim_u = Units::None; p.dim_x = Units::None; p.dim_y = Units::None;
    return p;
}
PDEProblem make_schrodinger(int dim) {
    PDEProblem p; p.type = PDE::SCHRODINGER; p.dim = dim;
    p.dim_u = Units::None; p.dim_x = Units::None; p.dim_y = Units::None;
    return p;
}
PDEProblem make_nonlinear_poisson() {
    PDEProblem p; p.type = PDE::NONLINEAR_POISSON; p.dim = 2;
    p.dim_u = Units::None; p.dim_x = Units::None; p.dim_y = Units::None;
    return p;
}
PDEProblem make_liouville() {
    PDEProblem p; p.type = PDE::LIOUVILLE; p.dim = 2;
    p.dim_u = Units::None; p.dim_x = Units::None; p.dim_y = Units::None;
    return p;
}
PDEProblem make_sine_gordon() {
    PDEProblem p; p.type = PDE::SINE_GORDON; p.dim = 2;
    p.dim_u = Units::None; p.dim_x = Units::None; p.dim_y = Units::None;
    return p;
}
PDEProblem make_airy(int dim) {
    PDEProblem p; p.type = PDE::AIRY; p.dim = dim; p.is_numerical = true;
    p.dim_u = Units::None; p.dim_x = Units::None; p.dim_y = Units::None;
    return p;
}
PDEProblem make_harmonic_oscillator(int dim) {
    PDEProblem p; p.type = PDE::HARMONIC_OSCILLATOR; p.dim = dim; p.is_numerical = true;
    p.dim_u = Units::None; p.dim_x = Units::None; p.dim_y = Units::None;
    return p;
}

PDEProblem make_navier_stokes() {
    PDEProblem p;
    p.type = PDE::NAVIER_STOKES;
    p.dim  = 2;
    p.k2   = 0.05; // nu = 1.0 / Re = 0.05 (Re = 20)
    p.is_numerical = false;
    p.dim_u = Dimension(2, -1, 0, 0, 0); // L^2 / T
    p.dim_x = Units::Length;
    p.dim_y = Units::Length;
    return p;
}

PDEProblem make_navier_stokes_unsteady() {
    PDEProblem p;
    p.type = PDE::NAVIER_STOKES_UNSTEADY;
    p.dim  = 2;
    p.k2   = 0.01; // Viscosidad nu
    p.dim_u = Units::Velocity;
    p.dim_x = Units::Length;
    p.dim_y = Units::Length;
    p.dim_t = Units::Time;
    return p;
}


PDEProblem make_fisher(int dim) {
    PDEProblem p; p.type = PDE::FISHER; p.dim = dim; p.is_numerical = true;
    p.dim_u = Units::None; p.dim_x = Units::None; p.dim_y = Units::None;
    return p;
}
PDEProblem make_duffing(int dim) {
    PDEProblem p; p.type = PDE::DUFFING; p.dim = dim; p.is_numerical = true;
    p.dim_u = Units::None; p.dim_x = Units::None; p.dim_y = Units::None;
    return p;
}
PDEProblem make_thomas_fermi(int dim) {
    PDEProblem p; p.type = PDE::THOMAS_FERMI; p.dim = dim; p.is_numerical = true;
    p.dim_u = Units::None; p.dim_x = Units::None; p.dim_y = Units::None;
    return p;
}
