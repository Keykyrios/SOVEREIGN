#pragma once
/// @file kernels.hpp
/// @brief Hawkes excitation kernels — power-law with exponential cutoff.
///
/// Reference: Lima & Choi (1805.09570v3)
/// φ(t) = α·(1 + t/β)^{-(1+ε)}, ε ∈ (0,1)
/// Branching ratio: ∫₀^∞ φ(t)dt = α·β/ε

#include <cmath>

namespace sovereign {

/// Power-law kernel: φ(t) = α·(1 + t/β)^{-(1+ε)}
struct PowerLawKernel {
    double alpha   = 0.5;   ///< Excitation magnitude
    double beta    = 1.0;   ///< Time scale
    double epsilon = 0.5;   ///< Tail index ∈ (0,1) for long memory

    double operator()(double t) const {
        return alpha * std::pow(1.0 + t / beta, -(1.0 + epsilon));
    }

    /// Branching ratio: ∫₀^∞ φ(t)dt = α·β/ε
    double branching_ratio() const {
        return alpha * beta / epsilon;
    }

    /// Integral from 0 to T: ∫₀^T φ(s)ds = (α·β/ε)·[1 - (1+T/β)^{-ε}]
    double integral(double T) const {
        return (alpha * beta / epsilon) * (1.0 - std::pow(1.0 + T / beta, -epsilon));
    }
};

/// Exponential kernel (for comparison / fast decay): φ(t) = α·exp(-β·t)
struct ExponentialKernel {
    double alpha = 0.5;
    double beta  = 1.0;

    double operator()(double t) const {
        return alpha * std::exp(-beta * t);
    }

    double branching_ratio() const { return alpha / beta; }
    double integral(double T) const {
        return (alpha / beta) * (1.0 - std::exp(-beta * T));
    }
};

/// Sum-of-exponentials kernel for efficient recursive computation:
/// φ(t) = Σ_k α_k · exp(-β_k · t)
/// Allows O(1) intensity update per event via:
///   A_k(t) = exp(-β_k·(t - t_last)) · (A_k(t_last) + α_k)
struct SumExpKernel {
    static constexpr int MAX_COMPONENTS = 8;
    int    n_components = 3;
    double alpha[MAX_COMPONENTS] = {0.3, 0.15, 0.05};
    double beta[MAX_COMPONENTS]  = {10.0, 1.0, 0.1};

    double operator()(double t) const {
        double val = 0;
        for (int k = 0; k < n_components; ++k)
            val += alpha[k] * std::exp(-beta[k] * t);
        return val;
    }

    double branching_ratio() const {
        double r = 0;
        for (int k = 0; k < n_components; ++k) r += alpha[k] / beta[k];
        return r;
    }
};

} // namespace sovereign
