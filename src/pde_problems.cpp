#include "pde_problems.hpp"
#include <cmath>
#include <algorithm>

namespace {
    const double PI_INTERNAL = 3.14159265358979323846;
}

std::string PDEProblem::name() const {
    return pde_name(type);
}

Complex PDEProblem::bc(double x, double y) const {
    return exact(x, y);
}

Complex PDEProblem::exact(double x, double y) const {
    if (dim == 1) {
        switch (type) {
            case PDE::LAPLACE:   return x;
            case PDE::POISSON:   
            case PDE::HELMHOLTZ: 
            case PDE::SINE_GORDON: return std::sin(PI_INTERNAL * x);
            case PDE::SCHRODINGER: return std::exp(I_COMPLEX * PI_INTERNAL * x);
            case PDE::HARMONIC_OSCILLATOR: return std::exp(-0.5 * x * x);
            case PDE::AIRY:      return std::exp(-0.5 * x);
            case PDE::FISHER:    return 1.0 / (1.0 + std::exp(-x));
            case PDE::DUFFING:   return 1.0 / std::cosh(x);
            case PDE::THOMAS_FERMI: return 1.0 / (x + 0.5);
            case PDE::NONLINEAR_POISSON:
            case PDE::LIOUVILLE: return 1.0 / (1.0 + x * x);
            case PDE::LANE_EMDEN: return 1.0 - (x * x) / 6.0;
            case PDE::TROESCH: return std::sinh(3.0 * x) / std::sinh(3.0);
            case PDE::GINZBURG_LANDAU: return std::tanh(x);
            case PDE::PAINLEVE1: return 0.5 * x * x;
            default: return 0.0;
        }
    } else {
        switch (type) {
            case PDE::LAPLACE:
                return std::sin(PI_INTERNAL * x) * std::sinh(PI_INTERNAL * y) / std::sinh(PI_INTERNAL);
            case PDE::POISSON:
            case PDE::HELMHOLTZ:
            case PDE::SINE_GORDON:
                return std::sin(PI_INTERNAL * x) * std::sin(PI_INTERNAL * y);
            case PDE::SCHRODINGER:
                return std::exp(I_COMPLEX * PI_INTERNAL * (x + y));
            case PDE::HARMONIC_OSCILLATOR:
                return std::exp(-0.5 * (x*x + y*y));
            case PDE::AIRY:
                return std::exp(-0.5 * (x + y));
            case PDE::FISHER:
                return 1.0 / (1.0 + std::exp(-(x + y)));
            case PDE::DUFFING:
                return 1.0 / std::cosh(x + y);
            case PDE::THOMAS_FERMI:
                return 1.0 / (x + y + 0.5);
            case PDE::NONLINEAR_POISSON:
            case PDE::LIOUVILLE:
                return 1.0 / (1.0 + x*x + y*y);
            case PDE::NAVIER_STOKES: {
                double Re = 1.0 / k2;
                double lambda = Re / 2.0 - std::sqrt(Re * Re / 4.0 + 4.0 * PI_INTERNAL * PI_INTERNAL);
                return y - std::exp(lambda * x) * std::sin(2.0 * PI_INTERNAL * y) / (2.0 * PI_INTERNAL * Re);
            }
            case PDE::NAVIER_STOKES_UNSTEADY:
                return std::sin(PI_INTERNAL * x) * std::sin(PI_INTERNAL * y); 
            case PDE::BRATU:
                return std::log(2.0 / (std::cosh(x + y) * std::cosh(x + y)));
            case PDE::ALLEN_CAHN: {
                double eps = std::sqrt(0.01);
                return std::tanh((x + y - 1.0) / (eps * std::sqrt(2.0))); 
            }
            default: return 0.0;
        }
    }
}

Complex PDEProblem::source(double x, double y) const {
    switch (type) {
        case PDE::LAPLACE: return 0.0;
        case PDE::BRATU: return 0.0; 
        case PDE::ALLEN_CAHN: return 0.0;
        case PDE::LANE_EMDEN: return 0.0;
        case PDE::POISSON: 
            return (dim == 1) ? (PI_INTERNAL*PI_INTERNAL*std::sin(PI_INTERNAL*x)) : (2.0*PI_INTERNAL*PI_INTERNAL*std::sin(PI_INTERNAL*x)*std::sin(PI_INTERNAL*y));
        case PDE::HELMHOLTZ: {
            double f_poisson = (dim == 1) ? (PI_INTERNAL*PI_INTERNAL*std::sin(PI_INTERNAL*x)) : (2.0*PI_INTERNAL*PI_INTERNAL*std::sin(PI_INTERNAL*x)*std::sin(PI_INTERNAL*y));
            return f_poisson - k2 * exact(x, y);
        }
        case PDE::NONLINEAR_POISSON: {
            double r2 = x*x + y*y; double den = 1.0 + r2;
            Complex lap = (8.0*r2)/(den*den*den) - 4.0/(den*den);
            return lap + exact(x, y) * exact(x, y);
        }
        case PDE::LIOUVILLE: {
            double r2 = x*x + y*y; double den = 1.0 + r2;
            Complex lap = (8.0*r2)/(den*den*den) - 4.0/(den*den);
            return lap + std::exp(exact(x, y));
        }
        default: return 0.0;
    }
}

Complex PDEProblem::pde_residual_ad(const AD& ad, double x, double y) const {
    Complex u = ad.v;
    Complex laplacian = ad.dxx + ad.dyy;
    switch (type) {
        case PDE::LAPLACE: return laplacian;
        case PDE::POISSON: return laplacian + source(x, y);
        case PDE::HELMHOLTZ: return laplacian + k2 * u + source(x, y);
        case PDE::SCHRODINGER: return laplacian + (static_cast<double>(dim)*PI_INTERNAL*PI_INTERNAL) * u;
        case PDE::HARMONIC_OSCILLATOR: return laplacian - (dim == 1 ? x*x : x*x+y*y)*u + (static_cast<double>(dim))*u;
        case PDE::NONLINEAR_POISSON: return laplacian + u*u - source(x, y);
        case PDE::LIOUVILLE: return laplacian + std::exp(u) - source(x, y);
        case PDE::SINE_GORDON: return laplacian - std::sin(u.real());
        case PDE::AIRY: return laplacian - (dim == 1 ? x : x+y)*u;
        case PDE::FISHER: return laplacian + u*(1.0-u);
        case PDE::DUFFING: return laplacian + u + u*u*u - std::sin(PI_INTERNAL * x); // Forced oscillator
        case PDE::THOMAS_FERMI: {
            double r = (dim == 1) ? std::abs(x) : std::sqrt(x*x + y*y);
            return laplacian - std::pow(u + 1e-6, 1.5) / std::sqrt(r + 1e-6);
        }
        case PDE::BRATU: return laplacian + 2.0 * std::exp(u);
        case PDE::ALLEN_CAHN: return 0.01 * laplacian - (u*u*u - u);
        case PDE::LANE_EMDEN: return laplacian + (2.0 / (x + 1e-6)) * ad.dx + std::pow(u + 1e-6, 3.0); 
        case PDE::TROESCH: return laplacian - 2.0 * std::sinh(2.0 * u.real()); // mu=2 is more stable for symbolic
        case PDE::GINZBURG_LANDAU: return laplacian + u - u*u*u;
        case PDE::PAINLEVE1: return laplacian - (u*u + x + y);
        default: return 0.0;
    }
}

std::vector<Point> PDEProblem::domain_points(int n) const {
    std::vector<Point> pts;
    double step = 1.0 / (std::sqrt(n));
    for (double i = 0.1; i < 0.9; i += step) {
        for (double j = 0.1; j < 0.9; j += step) {
            pts.push_back({i, j, 0.0});
        }
    }
    return pts;
}

std::vector<Point> PDEProblem::boundary_points(int n) const {
    std::vector<Point> pts;
    for (double s = 0; s <= 1.0; s += 0.05) {
        pts.push_back({0, s, 0.0}); pts.push_back({1, s, 0.0});
        pts.push_back({s, 0, 0.0}); pts.push_back({s, 1, 0.0});
    }
    return pts;
}

PDEProblem make_laplace(int d) { PDEProblem p; p.type = PDE::LAPLACE; p.dim = d; return p; }
PDEProblem make_poisson(int d) { PDEProblem p; p.type = PDE::POISSON; p.dim = d; return p; }
PDEProblem make_helmholtz(int d, double k) { PDEProblem p; p.type = PDE::HELMHOLTZ; p.dim = d; p.k2 = k*k; return p; }
PDEProblem make_schrodinger(int d) { PDEProblem p; p.type = PDE::SCHRODINGER; p.dim = d; return p; }
PDEProblem make_nonlinear_poisson() { PDEProblem p; p.type = PDE::NONLINEAR_POISSON; p.dim = 2; return p; }
PDEProblem make_liouville() { PDEProblem p; p.type = PDE::LIOUVILLE; p.dim = 2; return p; }
PDEProblem make_sine_gordon() { PDEProblem p; p.type = PDE::SINE_GORDON; p.dim = 2; return p; }
PDEProblem make_airy(int d) { PDEProblem p; p.type = PDE::AIRY; p.dim = d; p.is_numerical = true; return p; }
PDEProblem make_harmonic_oscillator(int d) { PDEProblem p; p.type = PDE::HARMONIC_OSCILLATOR; p.dim = d; p.is_numerical = true; return p; }
PDEProblem make_navier_stokes() { PDEProblem p; p.type = PDE::NAVIER_STOKES; p.dim = 2; p.k2 = 0.05; return p; }
PDEProblem make_navier_stokes_unsteady() { PDEProblem p; p.type = PDE::NAVIER_STOKES_UNSTEADY; p.dim = 2; p.k2 = 0.01; return p; }
PDEProblem make_fisher(int d) { PDEProblem p; p.type = PDE::FISHER; p.dim = d; p.is_numerical = true; return p; }
PDEProblem make_duffing(int d) { PDEProblem p; p.type = PDE::DUFFING; p.dim = d; p.is_numerical = true; return p; }
PDEProblem make_thomas_fermi(int d) { PDEProblem p; p.type = PDE::THOMAS_FERMI; p.dim = d; p.is_numerical = true; return p; }

PDEProblem make_bratu() { PDEProblem p; p.type = PDE::BRATU; p.dim = 2; return p; }
PDEProblem make_allen_cahn() { PDEProblem p; p.type = PDE::ALLEN_CAHN; p.dim = 2; p.k2 = 0.01; return p; }
PDEProblem make_lane_emden() { PDEProblem p; p.type = PDE::LANE_EMDEN; p.dim = 1; p.is_numerical = true; return p; }
PDEProblem make_troesch() { PDEProblem p; p.type = PDE::TROESCH; p.dim = 1; p.is_numerical = true; return p; }
PDEProblem make_ginzburg_landau() { PDEProblem p; p.type = PDE::GINZBURG_LANDAU; p.dim = 1; p.is_numerical = true; return p; }
PDEProblem make_painleve1() { PDEProblem p; p.type = PDE::PAINLEVE1; p.dim = 1; p.is_numerical = true; return p; }

Complex PDEProblem::pde_second_derivative(double x, Complex u) const {
    switch (type) {
        case PDE::LAPLACE: return 0.0;
        case PDE::POISSON: return source(x, 0.0);
        case PDE::HELMHOLTZ: return source(x, 0.0) - k2 * u;
        case PDE::SCHRODINGER: return -(PI_INTERNAL * PI_INTERNAL) * u;
        case PDE::AIRY:      return x * u;
        case PDE::HARMONIC_OSCILLATOR: return (x*x - 1.0) * u;
        case PDE::FISHER: return -u * (1.0 - u);
        case PDE::DUFFING: return -u - u * u * u;
        case PDE::THOMAS_FERMI: return u * u / (x + 0.5);
        case PDE::BRATU: return -2.0 * std::exp(u);
        case PDE::ALLEN_CAHN: return (u*u*u - u) / 0.01;
        case PDE::LANE_EMDEN: return -u*u*u - (2.0 / (x + 1e-6)) * (u - 1.0); // Rough approximation of u'
        case PDE::TROESCH: return 3.0 * std::sinh(3.0 * u.real());
        case PDE::GINZBURG_LANDAU: return -u + u*u*u;
        case PDE::PAINLEVE1: return u*u + x;
        default: return 0.0;
    }
}

Complex PDEProblem::numerical_exact(double x, double y) const {
    return exact(x, y);
}
