#pragma once
/// @file rough_vol.hpp
/// @brief Layer 1.1 — Rough Bergomi stochastic volatility.
///
/// References:
///   Bayer, Friz & Gatheral (RSVPsubmitted2) — rBergomi model
///   Bennedsen, Lunde & Pakkanen (1507.03004v4) — hybrid discretization
///
/// Model:
///   v_t = ξ₀(t) · ℰ(η Ŵ^H_t)   (stochastic exponential)
///   dS/S = sqrt(v_t) dW_t + dJ_t  (with Lévy jumps from Layer 1.2)
///   Corr(dW, dŴ) = ρ ≈ -0.9
///
///   Stochastic Hurst: dH(t) = κ_H(H̄ - H)dt + σ_H dB(t)

#include <sovereign/config.hpp>
#include <sovereign/core/random.hpp>
#include <sovereign/core/state.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace sovereign {

class RoughVolEngine {
    const RoughVolConfig& cfg_;
    int N_;

    // Per-asset Volterra evaluators
    std::vector<VolterraFBM> volterra_;

    // Stored Brownian increments for Volterra reconstruction
    // dW_price[i][j] = j-th increment of price Brownian for asset i
    std::vector<std::vector<double>> dW_price_;
    // dW_vol[i][j] = j-th increment of vol Brownian (correlated with price via ρ)
    std::vector<std::vector<double>> dW_vol_;

    // Current stochastic Hurst per asset
    Eigen::VectorXd H_current_;

    /// Logistic map to constrain H ∈ (0, 0.5)
    static double logistic_H(double x) {
        return 0.5 / (1.0 + std::exp(-x));
    }
    static double inv_logistic_H(double h) {
        return -std::log(0.5 / h - 1.0);
    }

public:
    RoughVolEngine(const RoughVolConfig& cfg, int n_assets)
        : cfg_(cfg), N_(n_assets)
    {
        volterra_.resize(N_);
        dW_price_.resize(N_);
        dW_vol_.resize(N_);
        H_current_ = Eigen::VectorXd::Constant(N_, cfg_.hurst);

        for (int i = 0; i < N_; ++i) {
            volterra_[i].init(cfg_.hurst, 1e-4, cfg_.n_hybrid_steps);
        }
    }

    /// Advance all assets by one timestep dt.
    /// corr_chol: Cholesky of cross-asset correlation Ω(t) [N×N]
    void step(SimulationState& state, double dt,
              const Eigen::LLT<Eigen::MatrixXd>& corr_chol,
              Xoshiro256& rng)
    {
        // Generate correlated price Brownians across assets
        Eigen::VectorXd Z_price(N_), Z_vol(N_), Z_hurst(N_);
        rng.fill_normal(Z_price);
        rng.fill_normal(Z_vol);
        rng.fill_normal(Z_hurst);

        // Cross-asset correlation on price Brownians
        Z_price = corr_chol.matrixL() * Z_price;

        double sqrt_dt = std::sqrt(dt);

        for (int i = 0; i < N_; ++i) {
            auto& a = state.assets[i];

            // ── Stochastic Hurst OU process ─────────────────────────
            double h_logit = inv_logistic_H(H_current_(i));
            h_logit += cfg_.hurst_kappa * (inv_logistic_H(cfg_.hurst_mean) - h_logit) * dt
                     + cfg_.hurst_sigma * sqrt_dt * Z_hurst(i);
            H_current_(i) = logistic_H(h_logit);
            a.hurst = H_current_(i);

            // ── Correlated vol Brownian: dŴ = ρ·dW + sqrt(1-ρ²)·dZ ──
            double dW_p = Z_price(i) * sqrt_dt;
            double dW_v = cfg_.rho * dW_p
                        + std::sqrt(1.0 - cfg_.rho * cfg_.rho) * Z_vol(i) * sqrt_dt;

            // Store for Volterra
            dW_price_[i].push_back(dW_p);
            dW_vol_[i].push_back(dW_v);

            // ── Volterra fBm: Ŵ^H(t_n) via hybrid scheme ───────────
            int n = static_cast<int>(dW_vol_[i].size());
            Eigen::Map<const Eigen::VectorXd> dW_vec(dW_vol_[i].data(), n);

            // Update Volterra evaluator with current H
            volterra_[i].init(a.hurst, dt, cfg_.n_hybrid_steps);
            double W_hat = volterra_[i].evaluate(n, dW_vec);

            // ── Rough Bergomi variance ──────────────────────────────
            // v_t = ξ₀ · exp(η·Ŵ^H - 0.5·η²·t^{2H})
            double t_now = (state.step + 1) * dt;
            double twoH = 2.0 * a.hurst;
            double log_v = std::log(cfg_.xi_0)
                         + cfg_.eta * W_hat
                         - 0.5 * cfg_.eta * cfg_.eta * std::pow(t_now, twoH);
            a.variance = std::exp(log_v);
            a.variance = std::max(a.variance, 1e-10);  // floor
            a.volatility = std::sqrt(a.variance);

            // ── Price dynamics: dS/S = sqrt(v)·dW + jump (added elsewhere)
            double drift = -0.5 * a.variance * dt;  // risk-neutral
            double diffusion = a.volatility * dW_p;
            a.log_price += drift + diffusion;

            double new_price = std::exp(a.log_price);
            a.return_1 = (new_price - a.price) / a.price;
            a.cum_return += a.return_1;
            a.price = new_price;
        }
    }

    const Eigen::VectorXd& hurst_vector() const { return H_current_; }
};

} // namespace sovereign
