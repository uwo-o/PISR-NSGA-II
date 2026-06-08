#pragma once
#include <array>
#include <string>

// Represents dimensions as powers of (L, T, M, Q, Theta)
// L: Length, T: Time, M: Mass, Q: Charge, Theta: Temperature
struct Dimension {
    signed char l, t, m, q, th;

    constexpr Dimension(signed char l=0, signed char t=0, signed char m=0, signed char q=0, signed char th=0)
        : l(l), t(t), m(m), q(q), th(th) {}

    bool operator==(const Dimension& other) const {
        return l == other.l && t == other.t && m == other.m && q == other.q && th == other.th;
    }

    bool operator!=(const Dimension& other) const {
        return !(*this == other);
    }

    Dimension operator+(const Dimension& other) const {
        return Dimension(l + other.l, t + other.t, m + other.m, q + other.q, th + other.th);
    }

    Dimension operator-(const Dimension& other) const {
        return Dimension(l - other.l, t - other.t, m - other.m, q - other.q, th - other.th);
    }

    bool is_adimensional() const {
        return l == 0 && t == 0 && m == 0 && q == 0 && th == 0;
    }

    std::string to_string() const {
        std::string s = "[";
        if (l != 0) s += "L^" + std::to_string(l) + " ";
        if (t != 0) s += "T^" + std::to_string(t) + " ";
        if (m != 0) s += "M^" + std::to_string(m) + " ";
        if (q != 0) s += "Q^" + std::to_string(q) + " ";
        if (th != 0) s += "Th^" + std::to_string(th) + " ";
        if (s.back() == ' ') s.pop_back();
        s += "]";
        if (s == "[]") return "[1]";
        return s;
    }
};

namespace Units {
    constexpr Dimension None(0, 0, 0, 0, 0);
    constexpr Dimension Length(1, 0, 0, 0, 0);
    constexpr Dimension Time(0, 1, 0, 0, 0);
    constexpr Dimension Mass(0, 0, 1, 0, 0);
    constexpr Dimension Temperature(0, 0, 0, 0, 1);
    constexpr Dimension Velocity(1, -1, 0, 0, 0);
    constexpr Dimension Acceleration(1, -2, 0, 0, 0);
    constexpr Dimension Force(1, -2, 1, 0, 0);
    constexpr Dimension Energy(2, -2, 1, 0, 0);
    constexpr Dimension Area(2, 0, 0, 0, 0);
    constexpr Dimension Viscosity(2, -1, 0, 0, 0); // L^2 / T
}
