#pragma once
// =============================================================================
// common.hpp  —  Tipos base, parámetros globales y struct AD
// =============================================================================

#include <vector>
#include <string>
#include <cmath>
#include <limits>

// ─── PDE types ────────────────────────────────────────────────────────────────
enum class PDE { LAPLACE, POISSON, HELMHOLTZ };

inline std::string pde_name(PDE p) {
    switch (p) {
        case PDE::LAPLACE:   return "Laplace";
        case PDE::POISSON:   return "Poisson";
        case PDE::HELMHOLTZ: return "Helmholtz";
    }
    return "Unknown";
}

// ─── 2-D point ────────────────────────────────────────────────────────────────
struct Point { double x, y; };

// ─── Automatic Differentiation value ─────────────────────────────────────────
// Holds u(x,y) and its first/second partials computed in a single tree pass.
struct AD {
    double v   = 0.0; // u
    double dx  = 0.0; // ∂u/∂x
    double dy  = 0.0; // ∂u/∂y
    double dxx = 0.0; // ∂²u/∂x²
    double dyy = 0.0; // ∂²u/∂y²
};

// ─── NSGA-II individual (method-agnostic) ─────────────────────────────────────
struct Individual {
    double mse_domain   = std::numeric_limits<double>::max();
    double mse_boundary = std::numeric_limits<double>::max();
    int    rank         = 0;
    double crowding     = 0.0;
};

// ─── Parámetros globales ──────────────────────────────────────────────────────
namespace Config {
    constexpr int    POP_SIZE       = 80;
    constexpr int    MAX_GEN        = 100;
    constexpr int    N_DOMAIN       = 256;   // puntos de colocación interiores
    constexpr int    N_BOUNDARY     = 64;    // puntos en ∂Ω
    constexpr double ERC_SIGMA      = 0.15;  // perturbación Gaussiana en ERCs
    constexpr int    MAX_TREE_DEPTH = 4;
    constexpr int    CODON_LENGTH   = 48;    // longitud del genotipo BNF (Koza)
    constexpr double CROSSOVER_PROB = 0.85;
    constexpr double MUTATION_PROB  = 0.15;
}
