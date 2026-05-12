#pragma once
/// @file levy_jumps.hpp
/// @brief Layer 1.2 — CGMY tempered stable Lévy process with CIR subordinator.
///
/// Reference: Carr, Geman, Madan & Yor — Stochastic Volatility for Lévy Processes
///
/// CGMY Lévy measure: ν(x) = C·exp(-G|x|)/|x|^{1+Y}  (x<0)
///                           C·exp(-Mx)/x^{1+Y}        (x>0)
/// Y ∈ (1,2): infinite variation + infinite activity
///
/// CIR time-change: dy = κ(η-y)dt + λ√y dW
/// X(t) = L(∫₀ᵗ y(s)ds) where L is CGMY process

#include <sovereign/config.hpp>
#include <sovereign/core/random.hpp>
#include <sovereign/core/state.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace sovereign {

class CGMYEngine {
    const LevyConfig& cfg_;
    int N_;

    // CIR subordinator state per asset
    Eigen::VectorXd y_cir_;

    // Accumulated time-change per asset
    Eigen::VectorXd tau_;

    /// Simulate one CGMY increment via series representation.
    /// Uses the shot-noise / Rosinski representation for tempered stable.
    double sample_cgmy(double C, double G, double M, double Y,
                       double dt, Xoshiro256& rng) const
    {
        // Truncated series representation for CGMY
        // For Y ∈ (1,2): use compound Poisson + Gaussian residual
        //
        // Small jumps (|x| < ε): approximate as Gaussian
        // Large jumps (|x| ≥ ε): simulate as compound Poisson

        double epsilon = 0.01;  // Truncation threshold

        // ── Large jumps: compound Poisson ──
        // Positive jumps rate: C·∫_ε^∞ exp(-Mx)/x^{1+Y} dx
        // Negative jumps rate: C·∫_ε^∞ exp(-Gx)/x^{1+Y} dx
        double rate_pos = C * upper_incomplete_gamma_rate(M, Y, epsilon);
        double rate_neg = C * upper_incomplete_gamma_rate(G, Y, epsilon);

        double total_rate = (rate_pos + rate_neg) * dt;
        int n_jumps = sample_poisson(total_rate, rng);

        double jump_sum = 0.0;
        for (int j = 0; j < n_jumps; ++j) {
            bool is_positive = rng.uniform() < rate_pos / (rate_pos + rate_neg);
            double decay = is_positive ? M : G;
            // Sample |x| from truncated tempered stable tail
            double x = sample_tempered_stable_tail(C, decay, Y, epsilon, rng);
            jump_sum += is_positive ? x : -x;
        }

        // ── Small jumps: Gaussian approximation ──
        // Variance of small jumps: 2C·∫_0^ε x^{1-Y} dx = 2C·ε^{2-Y}/(2-Y)
        double small_var = 2.0 * C * std::pow(epsilon, 2.0 - Y) / (2.0 - Y) * dt;
        // Mean of small jumps (drift correction)
        double small_mean = C * (std::pow(epsilon, 1.0 - Y) / (1.0 - Y))
                          * (1.0 / M - 1.0 / G) * dt;
        jump_sum += small_mean + std::sqrt(small_var) * rng.normal();

        return jump_sum;
    }

    /// Upper incomplete gamma-style rate integral
    double upper_incomplete_gamma_rate(double decay, double Y, double eps) const {
        // ∫_ε^∞ exp(-decay·x)/x^{1+Y} dx
        // Approximation via integration by parts / series
        return std::pow(eps, -Y) * std::exp(-decay * eps) / Y
             + decay * std::pow(eps, -Y + 1) * std::exp(-decay * eps) / (Y * (Y - 1));
    }

    /// Sample from tempered stable tail (rejection sampling)
    double sample_tempered_stable_tail(double C, double decay, double Y,
                                        double eps, Xoshiro256& rng) const
    {
        // Rejection from Pareto proposal: f(x) ∝ x^{-(1+Y)} for x ≥ ε
        // with exponential tilt acceptance: accept with prob exp(-decay·(x-ε))
        for (int attempt = 0; attempt < 10000; ++attempt) {
            // Sample from Pareto(ε, Y): x = ε·U^{-1/Y}
            double u = rng.uniform();
            double x = eps * std::pow(u, -1.0 / Y);
            // Accept with prob exp(-decay·(x - ε))
            if (rng.uniform() < std::exp(-decay * (x - eps))) {
                return x;
            }
        }
        return eps;  // Fallback
    }

    int sample_poisson(double lambda, Xoshiro256& rng) const {
        if (lambda < 30.0) {
            // Knuth's algorithm
            double L = std::exp(-lambda);
            int k = 0;
            double p = 1.0;
            do { ++k; p *= rng.uniform(); } while (p > L);
            return k - 1;
        } else {
            // Normal approximation
            return std::max(0, static_cast<int>(
                std::round(lambda + std::sqrt(lambda) * rng.normal())));
        }
    }

public:
    CGMYEngine(const LevyConfig& cfg, int n_assets)
        : cfg_(cfg), N_(n_assets)
    {
        y_cir_ = Eigen::VectorXd::Constant(N_, cfg_.cir_eta);
        tau_   = Eigen::VectorXd::Zero(N_);
    }

    /// Advance CGMY jumps + CIR subordinator by dt
    void step(SimulationState& state, double dt, Xoshiro256& rng) {
        double sqrt_dt = std::sqrt(dt);

        for (int i = 0; i < N_; ++i) {
            auto& a = state.assets[i];

            // ── CIR subordinator: dy = κ(η - y)dt + λ√y dW ──
            double dW = rng.normal() * sqrt_dt;
            double dy = cfg_.cir_kappa * (cfg_.cir_eta - y_cir_(i)) * dt
                      + cfg_.cir_lambda * std::sqrt(std::max(y_cir_(i), 0.0)) * dW;
            y_cir_(i) = std::max(y_cir_(i) + dy, 0.0);

            // Accumulated operational time
            double d_tau = y_cir_(i) * dt;
            tau_(i) += d_tau;

            // ── CGMY jump in operational time ──
            // Regime-dependent parameters (base * regime modifier)
            double regime_scale = 1.0 + 0.5 * (a.regime - 2);  // regime 2 = normal
            double C_eff = cfg_.C * std::max(regime_scale, 0.1);

            double jump = sample_cgmy(C_eff, cfg_.G, cfg_.M, cfg_.Y, d_tau, rng);

            // Apply jump to log-price
            a.log_price += jump;
            a.jump_component = jump;

            // Update price
            double new_price = std::exp(a.log_price);
            double ret = (new_price - a.price) / a.price;
            a.return_1 += ret;  // Accumulate with diffusive return
            a.price = new_price;
        }
    }

    const Eigen::VectorXd& cir_state() const { return y_cir_; }
};

} // namespace sovereign
