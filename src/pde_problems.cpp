#include "pde_problems.hpp"
#include "tree_node.hpp"
#include <cmath>
#include <algorithm>
#include <array>

namespace {
    const double PI_INTERNAL = 3.14159265358979323846;
}

std::string PDEProblem::name() const {
    return pde_name(type);
}

Complex PDEProblem::bc(double x, double y, double t) const {
    return exact(x, y, t);
}

Complex PDEProblem::exact(double x, double y, double t) const {
    if (dim == 1) {
        switch (type) {
            case PDE::LAPLACE:   return x;
            case PDE::POISSON:   
            case PDE::HELMHOLTZ: 
            case PDE::SINE_GORDON: return std::sin(PI_INTERNAL * x);
            case PDE::SCHRODINGER: return std::exp(I_COMPLEX * std::sqrt(k2) * x);
            case PDE::SCHRODINGER_UNSTEADY: return std::exp(I_COMPLEX * PI_INTERNAL * (x - PI_INTERNAL * t));
            case PDE::HARMONIC_OSCILLATOR: return std::exp(-0.5 * x * x);
            case PDE::AIRY:      return airy_ai(x);
            // Problemas con referencia NUMERICA (is_numerical): exact() solo
            // genera los datos de borde u(0), u(1) via bc(). La referencia en
            // el interior es la solucion numerica de ESE MISMO problema de
            // contorno (NumericalSolver::reference), no esta formula.
            case PDE::FISHER:    return 1.0 / (1.0 + std::exp(-x));
            case PDE::DUFFING:   return 1.0 / std::cosh(x);
            case PDE::THOMAS_FERMI: {
                // Problema de contorno truncado a [0,1]: u(0)=1 y u(1)=0.4240
                // (valor tabulado de la solucion semi-infinita en x=1). La
                // condicion en x=1 es parte de la DEFINICION del problema, no
                // una filtracion: lo que seria filtracion es usar la solucion
                // en el interior. Esta recta solo provee esos dos datos.
                double r = std::abs(x);
                return 1.0 - 0.5760 * r;
            }
            case PDE::NONLINEAR_POISSON:
            case PDE::LIOUVILLE: return 1.0 / (1.0 + x * x);
            // Lane-Emden es un problema de valor inicial (ivp_u0, ivp_du0); no
            // tiene forma cerrada para n=3 ni datos en x=1.
            case PDE::LANE_EMDEN: return std::numeric_limits<double>::quiet_NaN();
            case PDE::TROESCH: return std::sinh(3.0 * x) / std::sinh(3.0);
            // u'' + u - u^3 = 0 tiene solucion exacta tanh(x/sqrt 2). Antes la
            // referencia era tanh(x), que NO resuelve la ecuacion.
            case PDE::GINZBURG_LANDAU: return std::tanh(x / std::sqrt(2.0));
            case PDE::PAINLEVE1: return 0.5 * x * x;
            case PDE::CUSTOM_THESIS: return 1.0; // dato u(0) del problema ad-hoc
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
                return std::exp(I_COMPLEX * std::sqrt(k2) * (x + y));
            case PDE::SCHRODINGER_UNSTEADY:
                return std::exp(I_COMPLEX * PI_INTERNAL * (x + y - 2.0 * PI_INTERNAL * t));
            case PDE::HARMONIC_OSCILLATOR:
                return std::exp(-0.5 * (x*x + y*y));
            case PDE::AIRY:
                // u = Ai(a(x+y)) con 2a^3 = 1: Lap(u) = 2a^2 Ai''(a s) =
                // 2a^3 s Ai(a s) = (x+y) u.
                return airy_ai(std::cbrt(0.5) * (x + y));
            case PDE::FISHER:
                return 1.0 / (1.0 + std::exp(-(x + y)));
            case PDE::DUFFING:
                return 1.0 / std::cosh(x + y);
            case PDE::THOMAS_FERMI: {
                double r = std::sqrt(x*x + y*y);
                return 1.0 - 0.5760 * r; // mismo ancla que el caso 1D, ver comentario ahi
            }
            case PDE::NONLINEAR_POISSON:
            case PDE::LIOUVILLE:
                return 1.0 / (1.0 + x*x + y*y);
            case PDE::NAVIER_STOKES: {
                double Re = 1.0 / k2;
                double lambda = Re / 2.0 - std::sqrt(Re * Re / 4.0 + 4.0 * PI_INTERNAL * PI_INTERNAL);
                // Kovasznay estandar: psi = y - e^{lambda x} sin(2 pi y) / (2 pi).
                // (La amplitud es libre: J(psi, w) es lineal en ella. Antes era
                // 1/(2 pi Re) ~ 1.6e-3 con Re=100, y psi ~ y ya daba MSE 1e-7:
                // el problema era casi trivial.)
                return y - std::exp(lambda * x) * std::sin(2.0 * PI_INTERNAL * y) / (2.0 * PI_INTERNAL);
            }
            case PDE::NAVIER_STOKES_UNSTEADY: {
                double lambda = 2.0 * PI_INTERNAL * PI_INTERNAL * k2;
                return std::sin(PI_INTERNAL * x) * std::sin(PI_INTERNAL * y) * std::exp(-lambda * t); 
            }
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

Complex PDEProblem::source(double x, double y, double t) const {
    switch (type) {
        case PDE::LAPLACE: return 0.0;
        case PDE::SCHRODINGER_UNSTEADY: return 0.0;
        case PDE::BRATU: return 0.0; 
        case PDE::ALLEN_CAHN: return 0.0;
        case PDE::LANE_EMDEN: return 0.0;
        case PDE::POISSON: 
            return (dim == 1) ? (PI_INTERNAL*PI_INTERNAL*std::sin(PI_INTERNAL*x)) : (2.0*PI_INTERNAL*PI_INTERNAL*std::sin(PI_INTERNAL*x)*std::sin(PI_INTERNAL*y));
        case PDE::HELMHOLTZ: {
            double f_poisson = (dim == 1) ? (PI_INTERNAL*PI_INTERNAL*std::sin(PI_INTERNAL*x)) : (2.0*PI_INTERNAL*PI_INTERNAL*std::sin(PI_INTERNAL*x)*std::sin(PI_INTERNAL*y));
            return f_poisson - k2 * exact(x, y, t);
        }
        case PDE::NONLINEAR_POISSON: {
            double r2 = x*x + y*y; double den = 1.0 + r2;
            Complex lap = (8.0*r2)/(den*den*den) - 4.0/(den*den);
            return lap + exact(x, y, t) * exact(x, y, t);
        }
        case PDE::LIOUVILLE: {
            double r2 = x*x + y*y; double den = 1.0 + r2;
            Complex lap = (8.0*r2)/(den*den*den) - 4.0/(den*den);
            return lap + std::exp(exact(x, y, t));
        }
        default: return 0.0;
    }
}

Complex PDEProblem::pde_residual_ad(const AD& ad, double x, double y, double t) const {
    Complex u = ad.v;
    Complex laplacian = (dim == 1) ? ad.dxx : (ad.dxx + ad.dyy);
    switch (type) {
        case PDE::LAPLACE: return laplacian;
        case PDE::POISSON: return laplacian + source(x, y, t);
        case PDE::HELMHOLTZ: return laplacian + k2 * u + source(x, y, t);
        // Schrodinger estacionario: Lap(u) + dim*k^2*u = 0 con k^2 = k2. Antes
        // k = pi: k^2 = pi^2 (1D) y 2*pi^2 (2D) son autovalores de Dirichlet
        // en [0,1]^d (sin(pi x), sin(pi x)sin(pi y)), asi que el problema era
        // RESONANTE: u + C*sin(...) tambien era solucion para todo C, y el
        // residuo no podia elegir la referencia. Con k2=4 (k=2) no hay
        // resonancia (pi^2 > 4 en 1D, 2*pi^2 > 8 en 2D): solucion unica.
        case PDE::SCHRODINGER: return laplacian + (static_cast<double>(dim) * k2) * u;
        case PDE::SCHRODINGER_UNSTEADY: return I_COMPLEX * ad.dt + laplacian;
        case PDE::HARMONIC_OSCILLATOR: return laplacian - (dim == 1 ? x*x : x*x+y*y)*u + (static_cast<double>(dim))*u;
        case PDE::NONLINEAR_POISSON: return laplacian + u*u - source(x, y, t);
        case PDE::LIOUVILLE: return laplacian + std::exp(u) - source(x, y, t);
        case PDE::SINE_GORDON: return laplacian - std::sin(u);
        // Airy verdadero: u'' = x u (1D), Lap(u) = (x+y) u (2D). Antes la rama
        // 1D era u'' = u (consistente con e^{-x} pero no era Airy) y la 2D no
        // la satisfacia su propia referencia e^{-(x+y)}.
        case PDE::AIRY: return laplacian - (dim == 1 ? x : (x + y)) * u;
        case PDE::FISHER: return laplacian + u*(1.0-u);
        case PDE::DUFFING: {
            double forcing = (dim == 1) ? std::sin(PI_INTERNAL * x) : std::sin(PI_INTERNAL * x) * std::sin(PI_INTERNAL * y);
            return laplacian + u + u*u*u - forcing;
        }
        case PDE::THOMAS_FERMI: {
            double r = (dim == 1) ? std::abs(x) : std::sqrt(x*x + y*y);
            double eps = 1e-12;
            // |u|^{3/2}: la EDO solo esta definida para u>=0; el valor absoluto
            // extiende el operador a candidatos que cruzan cero sin producir NaN.
            return laplacian - std::pow(std::abs(u.real()) + eps, 1.5) / std::sqrt(r + eps);
        }
        case PDE::BRATU: return laplacian + 2.0 * std::exp(u);
        case PDE::ALLEN_CAHN: return 0.01 * laplacian - (u*u*u - u);
        // Lane-Emden n=3: u'' + (2/x) u' + u^3 = 0. Antes se usaban 2/(x+1e-6)
        // y (u+1e-6)^3, que cambian la ecuacion cerca del origen. El IVP
        // (u(0)=1, u'(0)=0) se impone por ansatz U = 1 + x^2 N(x), con lo que
        // 2U'/x = 4N + 2xN' es finito; el piso de 1e-300 solo evita 1/0 si
        // algun punto cae exactamente en x=0.
        case PDE::LANE_EMDEN: {
            double xs = (x >= 0.0) ? std::max(x, 1e-300) : std::min(x, -1e-300);
            return laplacian + (2.0 / xs) * ad.dx + u*u*u;
        }
        case PDE::TROESCH: return laplacian - 3.0 * std::sinh(3.0 * u);
        case PDE::GINZBURG_LANDAU: return laplacian + u - u*u*u;
        case PDE::PAINLEVE1: return laplacian - (u*u + x + y);
        case PDE::CUSTOM_THESIS: {
            // -u'' + 3(f'/f)u' + a*u = u^3 (ver custom_runs/thesis_eq.cpp).
            // f(x)=cosh(x) es un placeholder (f no fue especificada), f'/f=tanh(x).
            double a_param = k2;
            double f_prime_over_f = std::tanh(x);
            return -laplacian + 3.0 * f_prime_over_f * ad.dx + a_param * u - u * u * u;
        }
        // Navier-Stokes en funcion de corriente psi (u=psi_y, v=-psi_x,
        // vorticidad w=-Lap(psi)), con nu=k2. ECUACION REAL, no un proxy: el
        // jet debe venir de ad_eval_ext_t (spatial_order=4). Antes aqui habia
        // proxies de 2do orden (adveccion-difusion / calor) que la solucion de
        // Kovasznay ni siquiera satisfacia, y la busqueda entrenaba contra ellos.
        case PDE::NAVIER_STOKES: {
            Complex psi_x = ad.dx, psi_y = ad.dy;
            Complex L_x = ad.dxxx + ad.dxyy;
            Complex L_y = ad.dxxy + ad.dyyy;
            Complex biharmonic = ad.dxxxx + 2.0*ad.dxxyy + ad.dyyyy;
            return psi_y * L_x - psi_x * L_y - k2 * biharmonic;
        }
        case PDE::NAVIER_STOKES_UNSTEADY: {
            Complex uu = ad.dy, vv = -ad.dx;
            Complex w_x   = -(ad.dxxx + ad.dxyy);
            Complex w_y   = -(ad.dxxy + ad.dyyy);
            Complex lap_w = -(ad.dxxxx + 2.0*ad.dxxyy + ad.dyyyy);
            Complex w_t   = -(ad.dxxt + ad.dyyt);
            return w_t + uu * w_x + vv * w_y - k2 * lap_w;
        }
        default: return 0.0;
    }
}

std::vector<Point> PDEProblem::validation_points() const {
    std::vector<Point> pts;
    auto mid = [](int i, int n) { return (i + 0.5) / n; }; // nodos interiores, nunca exactamente en el borde
    if (dim == 1 && !is_unsteady) {
        const int n = 401;
        for (int i = 0; i < n; ++i) pts.push_back({mid(i, n), 0.0, 0.0});
    } else if (dim == 1) {
        const int nx = 41, nt = 21;
        for (int i = 0; i < nx; ++i) for (int k = 0; k < nt; ++k) pts.push_back({mid(i, nx), 0.0, mid(k, nt)});
    } else if (!is_unsteady) {
        const int n = 41;
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) pts.push_back({mid(i, n), mid(j, n), 0.0});
    } else {
        const int n = 21, nt = 11;
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) for (int k = 0; k < nt; ++k)
            pts.push_back({mid(i, n), mid(j, n), mid(k, nt)});
    }
    return pts;
}

// Datos de borde de un problema bien planteado:
//  - 1D estacionario DIRICHLET: x=0 y x=1.
//  - 1D IVP_LEFT / ANCHOR_LEFT: x=0 (el valor y, si es IVP, la derivada).
//  - 2D estacionario: las 4 aristas.
//  - No estacionario: frontera espacial para TODO t, mas la condicion inicial
//    en t=0 sobre todo el interior. Sin la condicion inicial el problema
//    queda indeterminado (p. ej. u=0 satisface ecuacion y frontera de
//    Navier-Stokes-Unsteady); sin la frontera para t>0, tambien.
std::vector<Point> PDEProblem::boundary_points(int /*n*/) const {
    std::vector<Point> pts;
    if (dim == 1 && !is_unsteady) {
        pts.push_back({0.0, 0.0, 0.0});
        if (bc_kind == BCKind::DIRICHLET) pts.push_back({1.0, 0.0, 0.0});
        return pts;
    }
    if (dim == 1) {
        const int nt = 21, nx = 41;
        for (int k = 0; k < nt; ++k) {
            double t = (double)k / (nt - 1);
            pts.push_back({0.0, 0.0, t}); pts.push_back({1.0, 0.0, t});
        }
        for (int i = 0; i < nx; ++i) pts.push_back({(i + 0.5) / nx, 0.0, 0.0});
        return pts;
    }
    const int ns = is_unsteady ? 21 : 41;
    const int nt = is_unsteady ? 11 : 1;
    for (int k = 0; k < nt; ++k) {
        double t = (nt == 1) ? 0.0 : (double)k / (nt - 1);
        for (int i = 0; i < ns; ++i) {
            double s = (double)i / (ns - 1);
            pts.push_back({0.0, s, t}); pts.push_back({1.0, s, t});
            pts.push_back({s, 0.0, t}); pts.push_back({s, 1.0, t});
        }
    }
    if (is_unsteady) {
        const int n = 21;
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
            pts.push_back({(i + 0.5) / n, (j + 0.5) / n, 0.0});
    }
    return pts;
}

Point PDEProblem::sample_boundary_point(std::mt19937& rng) const {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    if (dim == 1 && !is_unsteady) {
        if (bc_kind != BCKind::DIRICHLET) return {0.0, 0.0, 0.0};
        return {(U(rng) < 0.5) ? 0.0 : 1.0, 0.0, 0.0};
    }
    bool initial = is_unsteady && U(rng) < 0.5; // mitad condicion inicial, mitad frontera espacial
    if (dim == 1) {
        if (initial) return {U(rng), 0.0, 0.0};
        return {(U(rng) < 0.5) ? 0.0 : 1.0, 0.0, U(rng)};
    }
    if (initial) return {U(rng), U(rng), 0.0};
    double t = is_unsteady ? U(rng) : 0.0;
    double s = U(rng);
    int edge = std::uniform_int_distribution<int>(0, 3)(rng);
    if (edge == 0) return {0.0, s, t};
    if (edge == 1) return {1.0, s, t};
    if (edge == 2) return {s, 0.0, t};
    return {s, 1.0, t};
}

double PDEProblem::boundary_mismatch_sq(const AD& jet, const Point& p) const {
    if (bc_kind == BCKind::IVP_LEFT)
        return std::norm(jet.v - ivp_u0) + std::norm(jet.dx - ivp_du0);
    if (bc_kind == BCKind::ANCHOR_LEFT)
        return std::norm(jet.v - ivp_u0);

    double err = std::norm(jet.v - bc(p.x, p.y, p.t));
    if (spatial_order <= 2) return err;

    // Operador de orden 4: el problema de Dirichlet bien planteado (placa
    // empotrada / pared sin deslizamiento) necesita ademas la derivada normal.
    // Es un dato de borde estandar, no una pista de la solucion interior: se
    // da UNA derivada (la normal; la tangencial ya la determina u en el borde).
    double nx = 0.0, ny = 0.0;
    if (p.x == 0.0) nx = -1.0;
    else if (p.x == 1.0) nx = 1.0;
    else if (dim == 2 && p.y == 0.0) ny = -1.0;
    else if (dim == 2 && p.y == 1.0) ny = 1.0;
    else return err; // punto de condicion inicial (interior, t=0): solo el valor
    const double h = 1e-5;
    Complex g_dn = (bc(p.x + h * nx, p.y + h * ny, p.t) - bc(p.x - h * nx, p.y - h * ny, p.t)) / (2.0 * h);
    Complex u_dn = nx * jet.dx + ny * jet.dy;
    return err + std::norm(u_dn - g_dn);
}

PDEProblem make_laplace(int d) { PDEProblem p; p.type = PDE::LAPLACE; p.dim = d; return p; }
PDEProblem make_poisson(int d) { PDEProblem p; p.type = PDE::POISSON; p.dim = d; return p; }
PDEProblem make_helmholtz(int d, double k) { PDEProblem p; p.type = PDE::HELMHOLTZ; p.dim = d; p.k2 = k*k; return p; }
// Schrodinger: campo genuinamente complejo. k2=4 evita la resonancia de
// Dirichlet (ver pde_residual_ad).
PDEProblem make_schrodinger(int d) { PDEProblem p; p.type = PDE::SCHRODINGER; p.dim = d; p.k2 = 4.0; p.complex_field = true; return p; }
PDEProblem make_schrodinger_unsteady(int d) { PDEProblem p; p.type = PDE::SCHRODINGER_UNSTEADY; p.dim = d; p.is_unsteady = true; p.complex_field = true; return p; }
PDEProblem make_nonlinear_poisson() { PDEProblem p; p.type = PDE::NONLINEAR_POISSON; p.dim = 2; return p; }
PDEProblem make_liouville() { PDEProblem p; p.type = PDE::LIOUVILLE; p.dim = 2; return p; }
PDEProblem make_sine_gordon() { PDEProblem p; p.type = PDE::SINE_GORDON; p.dim = 2; return p; }
PDEProblem make_airy(int d) { PDEProblem p; p.type = PDE::AIRY; p.dim = d; p.dim_u = Units::Energy; return p; }
PDEProblem make_harmonic_oscillator(int d) { PDEProblem p; p.type = PDE::HARMONIC_OSCILLATOR; p.dim = d; p.dim_u = Units::Energy; return p; }
// Navier-Stokes en funcion de corriente: operador de orden espacial 4, asi que
// la frontera incluye psi y d(psi)/dn (y, en el no estacionario, la condicion
// inicial psi(x,y,0)).
// Kovasznay con Re = 20 (nu = 0.05), el caso clasico de la literatura.
PDEProblem make_navier_stokes() { PDEProblem p; p.type = PDE::NAVIER_STOKES; p.dim = 2; p.k2 = 0.05; p.spatial_order = 4; p.dim_u = Units::Viscosity; return p; }
PDEProblem make_navier_stokes_unsteady() { PDEProblem p; p.type = PDE::NAVIER_STOKES_UNSTEADY; p.dim = 2; p.k2 = 0.01; p.is_unsteady = true; p.spatial_order = 4; p.dim_u = Units::Viscosity; return p; }
PDEProblem make_fisher(int d) { PDEProblem p; p.type = PDE::FISHER; p.dim = d; p.is_numerical = true; p.dim_u = Units::None; return p; }
PDEProblem make_duffing(int d) { PDEProblem p; p.type = PDE::DUFFING; p.dim = d; p.is_numerical = true; p.dim_u = Units::Acceleration; return p; }
PDEProblem make_thomas_fermi(int d) {
    PDEProblem p; p.type = PDE::THOMAS_FERMI; p.dim = d; p.is_numerical = true; p.dim_u = Units::None;
    return p;
}

// Solo u(0)=1 dado (indeterminado a proposito, ver BCKind::ANCHOR_LEFT).
PDEProblem make_custom_thesis_eq(double a) {
    PDEProblem p; p.type = PDE::CUSTOM_THESIS; p.dim = 1; p.k2 = a;
    p.bc_kind = BCKind::ANCHOR_LEFT; p.ivp_u0 = 1.0;
    return p;
}


PDEProblem make_bratu() { PDEProblem p; p.type = PDE::BRATU; p.dim = 2; return p; }
PDEProblem make_allen_cahn() { PDEProblem p; p.type = PDE::ALLEN_CAHN; p.dim = 2; p.k2 = 0.01; return p; }
// Lane-Emden n=3 como problema de valor inicial: u(0)=1, u'(0)=0. Antes era un
// problema de contorno con u(1)=5/6 (el valor de la solucion de n=0), que no
// corresponde a la ecuacion de n=3.
PDEProblem make_lane_emden() {
    PDEProblem p; p.type = PDE::LANE_EMDEN; p.dim = 1; p.is_numerical = true;
    p.bc_kind = BCKind::IVP_LEFT; p.ivp_u0 = 1.0; p.ivp_du0 = 0.0;
    return p;
}
PDEProblem make_troesch() { PDEProblem p; p.type = PDE::TROESCH; p.dim = 1; p.is_numerical = true; return p; }
// Forma cerrada tanh(x/sqrt 2): ya no necesita referencia numerica.
PDEProblem make_ginzburg_landau() { PDEProblem p; p.type = PDE::GINZBURG_LANDAU; p.dim = 1; return p; }
PDEProblem make_painleve1() { PDEProblem p; p.type = PDE::PAINLEVE1; p.dim = 1; p.is_numerical = true; return p; }

std::vector<PDEProblem> benchmark_suite() {
    std::vector<PDEProblem> v;
    // Calibracion (forma cerrada, no singulares, no rigidas)
    v.push_back(make_laplace(1));
    v.push_back(make_poisson(1));
    v.push_back(make_harmonic_oscillator(1));
    for (int d : {1, 2}) {
        v.push_back(make_airy(d));
        v.push_back(make_fisher(d));
        v.push_back(make_duffing(d));
        v.push_back(make_thomas_fermi(d));
        v.push_back(make_schrodinger(d));
        v.push_back(make_schrodinger_unsteady(d));
    }
    v.push_back(make_navier_stokes());
    v.push_back(make_navier_stokes_unsteady());
    v.push_back(make_lane_emden());
    v.push_back(make_troesch());
    v.push_back(make_ginzburg_landau());
    v.push_back(make_painleve1());
    return v;
}

std::string problem_label(const PDEProblem& p) { return p.name() + (p.dim == 1 ? "_1D" : "_2D"); }

// Interpolacion cubica de Catmull-Rom sobre una grilla uniforme (extremos con
// extrapolacion lineal). Error O(h^3) contra O(h^2) de la lineal anterior, y
// continua en la primera derivada.
static Complex catmull_rom(const std::vector<Complex>& f, int N, double s) {
    double idx = std::clamp(s, 0.0, 1.0) * (N - 1);
    int i1 = std::clamp((int)std::floor(idx), 0, N - 2);
    int i2 = i1 + 1;
    double u = idx - i1;
    Complex p1 = f[i1], p2 = f[i2];
    Complex p0 = (i1 > 0) ? f[i1 - 1] : 2.0 * p1 - p2;
    Complex p3 = (i2 < N - 1) ? f[i2 + 1] : 2.0 * p2 - p1;
    return 0.5 * (2.0 * p1 + (p2 - p0) * u + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * (u * u)
                  + (3.0 * p1 - p0 - 3.0 * p2 + p3) * (u * u * u));
}

Complex PDEProblem::numerical_exact(double x, double y, double /*t*/) const {
    if (numerical_truth.empty() || truth_n < 2) return std::numeric_limits<double>::quiet_NaN();
    const int N = truth_n;
    if (dim == 1) return catmull_rom(numerical_truth, N, x);
    // Bicubica separable: interpolar en y cada columna vecina, luego en x.
    double ix = std::clamp(x, 0.0, 1.0) * (N - 1);
    int i1 = std::clamp((int)std::floor(ix), 0, N - 2);
    std::vector<Complex> col(N), across(4);
    int base = std::max(0, std::min(i1 - 1, N - 4));
    for (int k = 0; k < 4; ++k) {
        int i = base + k;
        for (int j = 0; j < N; ++j) col[j] = numerical_truth[i * N + j];
        across[k] = catmull_rom(col, N, y);
    }
    // across[k] = valor en x_{base+k}: Lagrange cubico sobre esos 4 nodos.
    double xs = ix - base;
    Complex r = 0.0;
    for (int k = 0; k < 4; ++k) {
        double w = 1.0;
        for (int m = 0; m < 4; ++m) if (m != k) w *= (xs - m) / double(k - m);
        r += w * across[k];
    }
    return r;
}

Complex PDEProblem::reference(double x, double y, double t) const {
    return is_numerical ? numerical_exact(x, y, t) : exact(x, y, t);
}

double airy_ai(double z) {
    // y'' = z y: coeficientes a_{n+3} = a_n / ((n+2)(n+3)).
    // Ai = c1 f - c2 g, f = 1 + z^3/6 + ..., g = z + z^4/12 + ...
    const double c1 = 0.355028053887817239260;   // Ai(0)
    const double c2 = 0.258819403792806798405;   // -Ai'(0)
    double f = 0.0, g = 0.0;
    double tf = 1.0, tg = z, z3 = z * z * z;
    for (int k = 0; k < 60; ++k) {
        f += tf; g += tg;
        int n = 3 * k;
        tf *= z3 / ((n + 2.0) * (n + 3.0));
        tg *= z3 / ((n + 3.0) * (n + 4.0));
        if (std::abs(tf) < 1e-18 * std::abs(f) && std::abs(tg) < 1e-18 * (std::abs(g) + 1e-300)) break;
    }
    return c1 * f - c2 * g;
}

// ─── Probing Físico (Análisis Asintótico y de Simetrías) ──────────────────────
// Todos los sondeos usan SOLO el operador (pde_residual_ad) evaluado sobre
// funciones genericas, y -- en el caso de la simetria -- los datos de borde
// evaluados SOBRE la frontera. Nunca la solucion en el interior.
//
// Los jets de sondeo traen derivadas hasta orden 4 (incluidas las mixtas), asi
// que los operadores de orden 4 (Navier-Stokes) tambien se sondean de verdad
// en vez de devolver residuo 0 trivial.
namespace {

struct Derivs { double d[5]; }; // f, f', f'', f''', f''''
Derivs sin_d(double s) { double a = std::sin(s), b = std::cos(s); return {{a, b, -a, -b, a}}; }
Derivs one_d() { return {{1.0, 0.0, 0.0, 0.0, 0.0}}; }

// Jet de f(x,y,t) = A(x) B(y) C(t).
AD product_jet(const Derivs& A, const Derivs& B, const Derivs& C) {
    auto P = [&](int a, int b, int c) { return Complex(A.d[a] * B.d[b] * C.d[c]); };
    AD j;
    j.v = P(0,0,0); j.dx = P(1,0,0); j.dy = P(0,1,0); j.dt = P(0,0,1);
    j.dxx = P(2,0,0); j.dyy = P(0,2,0); j.dtt = P(0,0,2);
    j.dxy = P(1,1,0); j.dxt = P(1,0,1); j.dyt = P(0,1,1);
    j.dxxx = P(3,0,0); j.dxxy = P(2,1,0); j.dxyy = P(1,2,0); j.dyyy = P(0,3,0);
    j.dxxt = P(2,0,1); j.dyyt = P(0,2,1);
    j.dxxxx = P(4,0,0); j.dxxyy = P(2,2,0); j.dyyyy = P(0,4,0);
    return j;
}

AD jet_lincomb(const AD& a, Complex sa, const AD& b, Complex sb) {
    AD r;
    r.v = sa*a.v + sb*b.v; r.dx = sa*a.dx + sb*b.dx; r.dy = sa*a.dy + sb*b.dy; r.dt = sa*a.dt + sb*b.dt;
    r.dxx = sa*a.dxx + sb*b.dxx; r.dyy = sa*a.dyy + sb*b.dyy; r.dtt = sa*a.dtt + sb*b.dtt;
    r.dxy = sa*a.dxy + sb*b.dxy; r.dxt = sa*a.dxt + sb*b.dxt; r.dyt = sa*a.dyt + sb*b.dyt;
    r.dxxx = sa*a.dxxx + sb*b.dxxx; r.dxxy = sa*a.dxxy + sb*b.dxxy;
    r.dxyy = sa*a.dxyy + sb*b.dxyy; r.dyyy = sa*a.dyyy + sb*b.dyyy;
    r.dxxt = sa*a.dxxt + sb*b.dxxt; r.dyyt = sa*a.dyyt + sb*b.dyyt;
    r.dxxxx = sa*a.dxxxx + sb*b.dxxxx; r.dxxyy = sa*a.dxxyy + sb*b.dxxyy; r.dyyyy = sa*a.dyyyy + sb*b.dyyyy;
    return r;
}

// Jet constante generico (valores arbitrarios no nulos en todos los ordenes).
AD generic_jet(double shift) {
    AD a;
    a.v = 1.0 + shift; a.dx = 0.5; a.dy = 0.45; a.dt = 0.1;
    a.dxx = -0.1; a.dyy = -0.13; a.dtt = 0.07;
    a.dxy = 0.21; a.dxt = 0.04; a.dyt = -0.03;
    a.dxxx = 0.3; a.dxxy = -0.2; a.dxyy = 0.15; a.dyyy = 0.25;
    a.dxxt = 0.05; a.dyyt = -0.07;
    a.dxxxx = 0.11; a.dxxyy = -0.09; a.dyyyy = 0.13;
    return a;
}

// Jet de u(1-x, y, t) en terminos del jet de u en (x,y,t): las derivadas con
// un numero impar de d/dx cambian de signo.
AD reflect_x(AD a) {
    a.dx = -a.dx; a.dxy = -a.dxy; a.dxt = -a.dxt;
    a.dxxx = -a.dxxx; a.dxyy = -a.dxyy;
    return a;
}
AD reflect_y(AD a) {
    a.dy = -a.dy; a.dxy = -a.dxy; a.dyt = -a.dyt;
    a.dyyy = -a.dyyy; a.dxxy = -a.dxxy;
    return a;
}

bool close(Complex a, Complex b, double tol) {
    double scale = std::max({std::abs(a), std::abs(b), 1.0});
    return std::isfinite(a.real()) && std::isfinite(b.real()) && std::abs(a - b) <= tol * scale;
}

} // namespace

PDEPriors probe_priors(const PDEProblem& prob) {
    PDEPriors priors;
    const AD dummy_ad = generic_jet(0.0);

    // 1. Invarianza traslacional
    Complex res_base = prob.pde_residual_ad(dummy_ad, 1.0, 1.0, 0.0);
    Complex res_x_shift = prob.pde_residual_ad(dummy_ad, 2.0, 1.0, 0.0);
    Complex res_y_shift = prob.pde_residual_ad(dummy_ad, 1.0, 2.0, 0.0);
    if (std::abs(res_base - res_x_shift) < 1e-9) priors.autonomous_x = true;
    if (std::abs(res_base - res_y_shift) < 1e-9) priors.autonomous_y = true;

    // 2. Simetría par (respecto de x=0)
    Complex res_x_neg = prob.pde_residual_ad(dummy_ad, -1.0, 1.0, 0.0);
    if (std::abs(res_base - res_x_neg) < 1e-9) priors.even_parity_x = true;

    // 3. Singularidad en el origen. Solo sesga la eleccion de esqueletos
    // (random_tree_special); la formulacion de la frontera la declara el
    // problema (bc_kind), no se infiere de aca.
    Complex res_origin = prob.pde_residual_ad(dummy_ad, 1e-6, 1e-6, 0.0);
    if (std::abs(res_origin) > 1e4 || !std::isfinite(res_origin.real())) priors.pole_at_origin = true;

    // 3b. Operador lineal homogeneo y autonomo -> modos exponenciales.
    {
        AD u1 = generic_jet(0.0), u2 = reflect_x(generic_jet(0.3));
        const Complex a(2.0, 0.0), b(0.0, 3.0);
        AD comb = jet_lincomb(u1, a, u2, b);
        bool lin = true;
        for (auto [px, py, pt] : {std::array<double, 3>{0.3, 0.6, 0.2}, std::array<double, 3>{0.7, 0.2, 0.5}}) {
            Complex r0 = prob.pde_residual_ad(AD{}, px, py, pt);
            Complex r1 = prob.pde_residual_ad(u1, px, py, pt), r2 = prob.pde_residual_ad(u2, px, py, pt);
            Complex rc = prob.pde_residual_ad(comb, px, py, pt);
            double scale = 1.0 + std::abs(r1) + std::abs(r2);
            if (!(std::abs(r0) < 1e-9 * scale) || !(std::abs(rc - (a * r1 + b * r2)) < 1e-9 * scale)) lin = false;
        }
        Complex rt0 = prob.pde_residual_ad(u1, 1.0, 1.0, 0.2), rt1 = prob.pde_residual_ad(u1, 1.0, 1.0, 0.7);
        bool auto_t = !prob.is_unsteady || std::abs(rt0 - rt1) < 1e-9;
        priors.exponential_modes = lin && priors.autonomous_x && (prob.dim == 1 || priors.autonomous_y) && auto_t;
    }

    // 4. Homogeneidad (invarianza de escala aproximada)
    AD scaled_ad = dummy_ad; double L = 2.0;
    scaled_ad.v = dummy_ad.v * L;
    scaled_ad.dxx = dummy_ad.dxx / L;
    Complex res_scaled = prob.pde_residual_ad(scaled_ad, L, L, L*L);
    if (std::abs(res_base - res_scaled / L) < 1e-3) priors.scale_invariant = true;

    // 5. Orden maximo de derivacion: el residuo cambia al anular los terminos
    // de cada orden (de mayor a menor).
    auto changes = [&](auto modify) {
        AD a = dummy_ad; modify(a);
        return std::abs(prob.pde_residual_ad(a, 1.0, 1.0, 0.0) - res_base) > 1e-9;
    };
    if (changes([](AD& a){ a.dxxxx = 0; a.dxxyy = 0; a.dyyyy = 0; })) priors.max_deriv_order = 4;
    else if (changes([](AD& a){ a.dxxx = 0; a.dxxy = 0; a.dxyy = 0; a.dyyy = 0; a.dxxt = 0; a.dyyt = 0; })) priors.max_deriv_order = 3;
    else if (changes([](AD& a){ a.dxx = 0; a.dyy = 0; a.dtt = 0; a.dxy = 0; })) priors.max_deriv_order = 2;
    else if (changes([](AD& a){ a.dx = 0; a.dy = 0; a.dt = 0; })) priors.max_deriv_order = 1;

    // 6. Autonomo en t
    if (prob.is_unsteady) {
        Complex res_t_shift = prob.pde_residual_ad(dummy_ad, 1.0, 1.0, 1.0);
        if (std::abs(res_base - res_t_shift) < 1e-9) priors.autonomous_t = true;
    }

    // 7. Simetria especular x <-> 1-x (y analoga en y). Operador: con u(1-x)
    // tambien solucion, el residuo del jet reflejado en 1-x debe coincidir
    // (a menos de signo) con el original, para varios jets y puntos. Datos de
    // borde: evaluados SOLO en la frontera (y en t=0 para la condicion inicial,
    // que es dato del problema). Antes se comparaba bc() en puntos interiores,
    // es decir, la solucion: eso era fuga de informacion.
    if (prob.bc_kind == BCKind::DIRICHLET) {
        auto op_symmetric = [&](auto reflect, bool in_x) {
            for (double shift : {0.0, 0.37}) {
                AD J = generic_jet(shift);
                AD Jr = reflect(J);
                for (double a0 : {0.15, 0.3, 0.42}) for (double b0 : {0.2, 0.55, 0.8}) {
                    double x0 = in_x ? a0 : b0, y0 = in_x ? b0 : a0;
                    double xr = in_x ? 1.0 - x0 : x0, yr = in_x ? y0 : 1.0 - y0;
                    Complex r = prob.pde_residual_ad(J, x0, y0, 0.5);
                    Complex rr = prob.pde_residual_ad(Jr, xr, yr, 0.5);
                    if (!close(r, rr, 1e-9) && !close(r, -rr, 1e-9)) return false;
                }
            }
            return true;
        };
        std::vector<double> ts = prob.is_unsteady ? std::vector<double>{0.0, 0.5, 1.0} : std::vector<double>{0.0};
        auto bc_symmetric_x = [&]() {
            if (prob.dim == 1) {
                for (double t : ts) if (!close(prob.bc(0.0, 0.0, t), prob.bc(1.0, 0.0, t), 1e-9)) return false;
                if (prob.is_unsteady)
                    for (double x : {0.1, 0.27, 0.4}) if (!close(prob.bc(x, 0.0, 0.0), prob.bc(1.0 - x, 0.0, 0.0), 1e-9)) return false;
                return true;
            }
            for (double t : ts) for (double s : {0.1, 0.27, 0.4, 0.5, 0.73}) {
                if (!close(prob.bc(0.0, s, t), prob.bc(1.0, s, t), 1e-9)) return false;            // aristas laterales
                if (!close(prob.bc(s, 0.0, t), prob.bc(1.0 - s, 0.0, t), 1e-9)) return false;      // arista inferior
                if (!close(prob.bc(s, 1.0, t), prob.bc(1.0 - s, 1.0, t), 1e-9)) return false;      // arista superior
            }
            if (prob.is_unsteady)
                for (double x : {0.1, 0.3}) for (double y : {0.2, 0.6})
                    if (!close(prob.bc(x, y, 0.0), prob.bc(1.0 - x, y, 0.0), 1e-9)) return false; // condicion inicial
            return true;
        };
        auto bc_symmetric_y = [&]() {
            for (double t : ts) for (double s : {0.1, 0.27, 0.4, 0.5, 0.73}) {
                if (!close(prob.bc(s, 0.0, t), prob.bc(s, 1.0, t), 1e-9)) return false;
                if (!close(prob.bc(0.0, s, t), prob.bc(0.0, 1.0 - s, t), 1e-9)) return false;
                if (!close(prob.bc(1.0, s, t), prob.bc(1.0, 1.0 - s, t), 1e-9)) return false;
            }
            if (prob.is_unsteady)
                for (double x : {0.2, 0.6}) for (double y : {0.1, 0.3})
                    if (!close(prob.bc(x, y, 0.0), prob.bc(x, 1.0 - y, 0.0), 1e-9)) return false;
            return true;
        };
        priors.mirror_symmetric_x = op_symmetric(reflect_x, true) && bc_symmetric_x();
        if (prob.dim == 2) priors.mirror_symmetric_y = op_symmetric(reflect_y, false) && bc_symmetric_y();
    }

    // 8. Separabilidad del operador (2D): funcion de sondeo GENERICA
    // (sin x + sin y, sin x sin y), nunca la solucion. Si la derivada cruzada
    // del residuo es ~0, el operador no mezcla x e y para esa forma.
    if (prob.dim == 2) {
        auto probe_mixing = [&](bool multiplicative) -> bool {
            auto eval_R = [&](double x, double y) -> Complex {
                AD probe = multiplicative
                    ? product_jet(sin_d(x), sin_d(y), one_d())
                    : jet_lincomb(product_jet(sin_d(x), one_d(), one_d()), 1.0,
                                  product_jet(one_d(), sin_d(y), one_d()), 1.0);
                return prob.pde_residual_ad(probe, x, y, 0.0);
            };
            double h = 1e-3;
            for (double x0 : {0.2, 0.4, 0.6}) {
                for (double y0 : {0.25, 0.5, 0.75}) {
                    Complex rpp = eval_R(x0+h, y0+h), rpm = eval_R(x0+h, y0-h);
                    Complex rmp = eval_R(x0-h, y0+h), rmm = eval_R(x0-h, y0-h);
                    if (!std::isfinite(rpp.real()) || !std::isfinite(rmm.real())) return false;
                    Complex d2xy = (rpp - rpm - rmp + rmm) / (4.0*h*h);
                    double scale = std::max({std::abs(rpp), std::abs(rmm), 1.0});
                    if (std::abs(d2xy) > 1e-2 * scale) return false;
                }
            }
            return true;
        };
        priors.additive_separable = probe_mixing(false);
        if (!priors.additive_separable) priors.multiplicative_separable = probe_mixing(true);
    }

    // 8b. Suma modal: el operador es afin-lineal en u (superposicion), sondeado
    // escalando la funcion generica sin x sin y.
    if (prob.dim == 2) {
        AD probe0;
        Complex r0 = prob.pde_residual_ad(probe0, 0.37, 0.61, 0.0);
        bool is_affine_linear = std::isfinite(r0.real());
        for (double x0 : {0.2, 0.5, 0.8}) {
            for (double y0 : {0.3, 0.6}) {
                AD base = product_jet(sin_d(x0), sin_d(y0), one_d());
                Complex r1 = prob.pde_residual_ad(base, x0, y0, 0.0);
                Complex r2 = prob.pde_residual_ad(jet_lincomb(base, 2.0, base, 0.0), x0, y0, 0.0);
                Complex rz = prob.pde_residual_ad(probe0, x0, y0, 0.0);
                if (!std::isfinite(r1.real()) || !std::isfinite(r2.real())) { is_affine_linear = false; break; }
                Complex expected_r2 = rz + 2.0 * (r1 - rz);
                double scale = std::max({std::abs(r1), std::abs(r2), 1.0});
                if (std::abs(r2 - expected_r2) > 1e-6 * scale) { is_affine_linear = false; break; }
            }
            if (!is_affine_linear) break;
        }
        priors.modal_sum_separable = is_affine_linear;
    }

    // 8c. Invarianza rotacional/diagonal: el operador trata x e y igual.
    if (prob.dim == 2) {
        bool rot_ok = true;
        for (double x0 : {0.2, 0.5, 0.8}) {
            for (double y0 : {0.3, 0.6}) {
                Complex r1 = prob.pde_residual_ad(product_jet(sin_d(x0), one_d(), one_d()), x0, y0, 0.0);
                Complex r2 = prob.pde_residual_ad(product_jet(one_d(), sin_d(x0), one_d()), y0, x0, 0.0);
                if (!std::isfinite(r1.real()) || !std::isfinite(r2.real())) { rot_ok = false; break; }
                double scale = std::max({std::abs(r1), std::abs(r2), 1.0});
                if (std::abs(r1 - r2) > 1e-6 * scale) { rot_ok = false; break; }
            }
            if (!rot_ok) break;
        }
        priors.rotation_invariant = rot_ok;
    }

    // 8d. Amortiguamiento lineal en la primera derivada.
    {
        auto probe_dx = [&](double dx_val) -> Complex {
            AD p = generic_jet(-0.5); p.dx = dx_val;
            return prob.pde_residual_ad(p, 0.4, 0.4, 0.0);
        };
        Complex ra = probe_dx(0.0), rb = probe_dx(1.0);
        bool damped_x = std::isfinite(ra.real()) && std::isfinite(rb.real()) && std::abs(rb - ra) > 1e-6;
        bool damped_y = false;
        if (prob.dim == 2) {
            auto probe_dy = [&](double dy_val) -> Complex {
                AD p = generic_jet(-0.5); p.dy = dy_val;
                return prob.pde_residual_ad(p, 0.4, 0.4, 0.0);
            };
            Complex rc = probe_dy(0.0), rd = probe_dy(1.0);
            damped_y = std::isfinite(rc.real()) && std::isfinite(rd.real()) && std::abs(rd - rc) > 1e-6;
        }
        priors.damped_oscillator = damped_x || damped_y;
    }

    // 8e. Separabilidad triple u = f(x)g(y)h(t) (2D no estacionario).
    if (prob.dim == 2 && prob.is_unsteady) {
        auto eval_R3 = [&](double x, double y, double t) -> Complex {
            return prob.pde_residual_ad(product_jet(sin_d(x), sin_d(y), sin_d(t)), x, y, t);
        };
        bool triple_ok = true;
        double h = 1e-3;
        for (double x0 : {0.2, 0.4, 0.6}) {
            for (double y0 : {0.25, 0.5, 0.75}) {
                for (double t0 : {0.3, 0.5, 0.7}) {
                    Complex r000 = eval_R3(x0, y0, t0);
                    if (!std::isfinite(r000.real())) { triple_ok = false; break; }
                    double scale = std::max(std::abs(r000), 1.0);
                    Complex rxy = (eval_R3(x0+h,y0+h,t0) - eval_R3(x0+h,y0-h,t0)
                                 - eval_R3(x0-h,y0+h,t0) + eval_R3(x0-h,y0-h,t0)) / (4.0*h*h);
                    Complex rxt = (eval_R3(x0+h,y0,t0+h) - eval_R3(x0+h,y0,t0-h)
                                 - eval_R3(x0-h,y0,t0+h) + eval_R3(x0-h,y0,t0-h)) / (4.0*h*h);
                    Complex ryt = (eval_R3(x0,y0+h,t0+h) - eval_R3(x0,y0+h,t0-h)
                                 - eval_R3(x0,y0-h,t0+h) + eval_R3(x0,y0-h,t0-h)) / (4.0*h*h);
                    if (std::abs(rxy) > 1e-2 * scale || std::abs(rxt) > 1e-2 * scale || std::abs(ryt) > 1e-2 * scale) {
                        triple_ok = false; break;
                    }
                }
                if (!triple_ok) break;
            }
            if (!triple_ok) break;
        }
        priors.triple_separable = triple_ok;
    }

    return priors;
}
