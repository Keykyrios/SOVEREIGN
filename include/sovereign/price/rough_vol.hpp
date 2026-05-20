#pragma once
/// @file rough_vol.hpp
/// @brief Layer 1.1 — Rough Bergomi stochastic volatility.
///
/// References:
///   Bayer, Friz & Gatheral (RSVPsubmitted2) — rBergomi model
///   Bennedsen, Lunde & Pakkanen (1507.03004v4) — hybrid discretization
///   Abi Jaber & El Euch (2019) — multi-factor Markovian approximation
///
/// Model:
///   v_t = ξ₀(t) · ℰ(η Ŵ^H_t)   (stochastic exponential)
///   dS/S = sqrt(v_t) dW_t + dJ_t  (with Lévy jumps from Layer 1.2)
///   Corr(dW, dŴ) = ρ ≈ -0.9
///
///   Stochastic Hurst: dH(t) = κ_H(H̄ - H)dt + σ_H dB(t)
///
/// Volterra kernel K(t) = √(2H) · t^{H-1/2} is approximated by
/// a sum of K exponentials: K(t) ≈ Σ_k c_k · exp(-λ_k · t)
/// Each component is an OU process updated in O(1) per tick.
/// Total cost: O(K) per tick instead of O(T) for dense Riemann.

#include <sovereign/config.hpp>
#include <sovereign/core/random.hpp>
#include <sovereign/core/state.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <array>

namespace sovereign {

/// Multi-scale Markovian approximation of fractional kernel
/// K(t) = √(2H) · t^{H-1/2} ≈ Σ_k c_k · exp(-λ_k · t)
/// Each OU process: dX_k = -λ_k X_k dt + c_k dW
/// Then Ŵ^H(t) ≈ Σ_k X_k(t)
class MarkovianFBM {
    static constexpr int N_FACTORS = 12;  ///< 12 exponential components

    double H_ = 0.1;
    double gamma_;   // γ = 1/2 - H
    std::array<double, N_FACTORS> lambda_;  ///< Decay rates (log-spaced)
    std::array<double, N_FACTORS> c_;       ///< Weights
    std::array<double, N_FACTORS> X_;       ///< OU states

    /// Compute weights c_k such that Σ_k c_k/λ_k^γ ≈ √(2H) · Γ(H+1/2)
    void compute_weights() {
        gamma_ = 0.5 - H_;
        double sqrt2H = std::sqrt(2.0 * H_);

        // Log-space λ from 0.01 (memory ~100 time units) to 10000 (memory ~0.1ms)
        double log_lam_min = std::log(0.01);
        double log_lam_max = std::log(10000.0);

        for (int k = 0; k < N_FACTORS; ++k) {
            double frac = static_cast<double>(k) / (N_FACTORS - 1);
            lambda_[k] = std::exp(log_lam_min + frac * (log_lam_max - log_lam_min));

            // Weight: c_k ∝ λ_k^{γ-1} · Δlog(λ) (Abi Jaber & El Euch 2019)
            // γ = 1/2 - H, so γ-1 = -1/2 - H
            double d_log_lam = (log_lam_max - log_lam_min) / (N_FACTORS - 1);
            c_[k] = sqrt2H * std::pow(lambda_[k], gamma_ - 1.0) * d_log_lam
                   / std::tgamma(gamma_);
        }

        // DO NOT normalize c_k. The quadrature weights approximate the fractional
        // kernel correctly. Rescaling them breaks the power-law structure.
        // Variance target is handled externally via xi_0.
    }

public:
    MarkovianFBM() { X_.fill(0); compute_weights(); }

    void init(double H) {
        H_ = H;
        compute_weights();
        reset(); // Reset state when kernel changes to prevent mixing histories
    }

    double cached_H() const { return H_; }

    /// Update all OU factors with new Brownian increment dW.
    /// Returns approximated Ŵ^H(t_n) = Σ_k X_k
    double step(double Z, double dt) {
        double W_hat = 0;
        for (int k = 0; k < N_FACTORS; ++k) {
            // Exact OU update: X_{k,n+1} = X_{k,n} · exp(-λ_k·dt) + c_k · exact_diffusion
            double decay = std::exp(-lambda_[k] * dt);
            if (decay < 1e-15) decay = 0.0; // Prevent subnormal float stalls
            double exact_vol = std::sqrt((1.0 - std::exp(-2.0 * lambda_[k] * dt)) / (2.0 * lambda_[k]));
            X_[k] = X_[k] * decay + c_[k] * exact_vol * Z;
            W_hat += X_[k];
        }
        return W_hat;
    }

    /// Exact analytic variance of the multi-factor OU process at time t.
    /// Used as the exact martingale compensator for the Rough Bergomi variance.
    /// V(t) = Σ_k c_k^2 * (1 - exp(-2·λ_k·t)) / (2·λ_k)
    double exact_variance(double t) const {
        double v = 0.0;
        for (int k = 0; k < N_FACTORS; ++k) {
            v += (c_[k] * c_[k] / (2.0 * lambda_[k])) * (1.0 - std::exp(-2.0 * lambda_[k] * t));
        }
        return v;
    }

    void reset() { X_.fill(0); }
};


class RoughVolEngine {
    const RoughVolConfig& cfg_;
    int N_;

    // Per-asset Markovian fBm approximation (replaces Volterra + ring buffer)
    std::vector<MarkovianFBM> markov_fbm_;

    // Current stochastic Hurst per asset
    Eigen::VectorXd H_current_;

    // Pre-allocated scratch vectors (avoid heap alloc in hot loop)
    Eigen::VectorXd Z_price_, Z_vol_, Z_hurst_;

    /// Reflection boundary for H ∈ (0.01, 0.99)
    static double reflect_H(double x) {
        if (x < 0.01) return 0.01 + (0.01 - x);
        if (x > 0.99) return 0.99 - (x - 0.99);
        return x;
    }

public:
    RoughVolEngine(const RoughVolConfig& cfg, int n_assets)
        : cfg_(cfg), N_(n_assets)
    {
        markov_fbm_.resize(N_);
        H_current_ = Eigen::VectorXd::Constant(N_, cfg_.hurst);
        Z_price_.resize(N_);
        Z_vol_.resize(N_);
        Z_hurst_.resize(N_);

        for (int i = 0; i < N_; ++i) {
            markov_fbm_[i].init(cfg_.hurst);
        }
    }

    /// Advance all assets by one timestep dt.
    /// corr_chol: Cholesky of cross-asset correlation Ω(t) [N×N]
    void step(SimulationState& state, double dt,
              const Eigen::LLT<Eigen::MatrixXd>& corr_chol,
              Xoshiro256& rng)
    {
        // Generate correlated price Brownians across assets
        rng.fill_normal(Z_price_);
        rng.fill_normal(Z_vol_);
        rng.fill_normal(Z_hurst_);

        // SYSTEMIC MARKET FACTOR: r_i = β·Z_market + √(1-β²)·ε_i
        // All real financial assets share a common market factor (Beta).
        // Without this, N independent random walks produce zero cross-correlation
        // and the RMT cleaner correctly cleans the matrix back to Identity forever.
        // β=0.5 gives 25% variance explained by market.
        // Market eigenvalue = 1 + (N-1)·β² = 13.25 >> MP edge ≈ 6.37 → survives cleaning
        double beta = 0.5;
        double Z_market = rng.normal();
        double beta_complement = std::sqrt(1.0 - beta * beta);
        for (int i = 0; i < N_; ++i) {
            Z_price_(i) = beta * Z_market + beta_complement * Z_price_(i);
        }

        // Cross-asset correlation on price Brownians
        Z_price_ = corr_chol.matrixL() * Z_price_;

        double sqrt_dt = std::sqrt(dt);

        #pragma omp parallel for
        for (int i = 0; i < N_; ++i) {
            auto& a = state.assets[i];

            // ── Stochastic Hurst OU process ─────────────────────────
            // Removed logistic map, using simple Euler with reflection to allow H > 0.5
            H_current_(i) += cfg_.hurst_kappa * (cfg_.hurst_mean - H_current_(i)) * dt
                           + cfg_.hurst_sigma * sqrt_dt * Z_hurst_(i);
            H_current_(i) = reflect_H(H_current_(i));
            a.hurst = H_current_(i);

            // ── Correlated vol Brownian: Z_v = ρ·Z_p + sqrt(1-ρ²)·Z_vol ──
            double dW_p = Z_price_(i) * sqrt_dt;
            double Z_v = cfg_.rho * Z_price_(i)
                       + std::sqrt(1.0 - cfg_.rho * cfg_.rho) * Z_vol_(i);

            // ── Re-init Markovian kernel when H drifts significantly ──
            double h_diff = std::abs(a.hurst - markov_fbm_[i].cached_H());
            if (h_diff > 0.001) {
                markov_fbm_[i].init(a.hurst);
            }

            // ── Markovian fBm: O(K) per tick instead of O(T) ─────────
            double W_hat = markov_fbm_[i].step(Z_v, dt);
            // Removed W_hat clamp — it distorts rough path properties

            // ── Rough Bergomi variance ──────────────────────────────
            // v_t = ξ₀ · exp(η·Ŵ^H - 0.5·η²·V_exact(t))
            double t_now = (state.step + 1) * dt;
            double exact_var = markov_fbm_[i].exact_variance(t_now);
            double log_v = std::log(cfg_.xi_0)
                         + cfg_.eta * W_hat
                         - 0.5 * cfg_.eta * cfg_.eta * exact_var;
            log_v = std::clamp(log_v, -10.0, 5.0);  // variance in [~0, ~150]
            a.variance = std::exp(log_v);
            a.variance = std::max(a.variance, 1e-10);
            a.volatility = std::sqrt(a.variance);

            // ── Price dynamics: dS/S = sqrt(v)·dW + jump (added elsewhere)
            double drift = -0.5 * a.variance * dt;  // risk-neutral
            double diffusion = a.volatility * dW_p;
            a.log_price += drift + diffusion;
            // Removed log_price clamp — allows true drawdown tails

            double new_price = std::exp(a.log_price);
            a.price = new_price;
        }
    }

    const Eigen::VectorXd& hurst_vector() const { return H_current_; }
};

} // namespace sovereign
