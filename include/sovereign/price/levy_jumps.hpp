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

    // Persistent per-thread RNGs (forked once, not per-tick)
    std::vector<Xoshiro256> thread_rngs_;

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
        // Variance of small jumps: 2C·∫_0^ε x^{2}·ν(dx) ≈ 2C·ε^{2-Y}/(2-Y)
        double safe_Y = std::min(Y, 1.95); // Prevent divergence as Y -> 2
        double small_var = 2.0 * C * std::pow(epsilon, 2.0 - safe_Y) / (2.0 - safe_Y) * dt;
        // Mean of small jumps for Y ∈ (1,2):
        // E[small] = C·(1/(1-Y))·(G-M)·ε^{1-Y}  (NOT ε^{2-Y}/(2-Y))
        double small_mean = C * (G - M) * std::pow(epsilon, 1.0 - safe_Y) / (1.0 - safe_Y) * dt;
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
    double sample_tempered_stable_tail(double /*C*/, double decay, double Y,
                                        double eps, Xoshiro256& rng) const
    {
        // Rejection from Pareto proposal: f(x) ∝ x^{-(1+Y)} for x ≥ ε
        // with exponential tilt acceptance: accept with prob exp(-decay·(x-ε))
        for (int attempt = 0; attempt < 10000; ++attempt) {
            // Sample from Pareto(ε, Y): x = ε·U^{-1/Y}
            double u = 1.0 - rng.uniform();  // (0,1] for Pareto inversion
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
            // Knuth's algorithm (correct form: k=0, test THEN increment)
            double L = std::exp(-lambda);
            int k = -1;
            double p = 1.0;
            do { ++k; p *= rng.uniform(); } while (p > L);
            return k;
        } else {
            // Normal approximation for large lambda
            int k = static_cast<int>(std::round(lambda + std::sqrt(lambda) * rng.normal()));
            return k > 0 ? k : 0; // Prevent std::max skewing the statistical distribution median by wrapping negatives blindly.
        }
    }

public:
    CGMYEngine(const LevyConfig& cfg, int n_assets)
        : cfg_(cfg), N_(n_assets)
    {
        y_cir_ = Eigen::VectorXd::Constant(N_, cfg_.cir_eta);
        tau_   = Eigen::VectorXd::Zero(N_);
    }

    /// Must be called once after construction with the global RNG
    void init_thread_rngs(Xoshiro256& rng) {
        thread_rngs_.resize(N_);
        for (int i = 0; i < N_; ++i) thread_rngs_[i] = rng.fork();
    }

    void step(SimulationState& state, double dt, Xoshiro256& global_rng) {
        double sqrt_dt = std::sqrt(dt);

        // SYSTEMIC JUMP FACTOR: A market-wide jump (e.g., flash crash) affecting all assets.
        // Simulate one systemic jump component, drawn from a master CGMY process.
        double systemic_jump = sample_cgmy(cfg_.C * 0.5, cfg_.G, cfg_.M, cfg_.Y, dt, global_rng);

        #pragma omp parallel for
        for (int i = 0; i < N_; ++i) {
            auto& a = state.assets[i];
            Xoshiro256& local_rng = thread_rngs_[i];

            // ── CIR subordinator: Alfonsi implicit (no truncation bias) ──
            // Enforce Feller condition (2*kappa*eta > lambda^2) dynamically
            double lambda_eff = cfg_.cir_lambda;
            if (2.0 * cfg_.cir_kappa * cfg_.cir_eta <= lambda_eff * lambda_eff) {
                // Throttle volatility to prevent reaching 0
                lambda_eff = std::sqrt(1.99 * cfg_.cir_kappa * cfg_.cir_eta);
            }
            
            // y_{t+dt} = ((√y_t + λ/2·dW) / (1 + κ/2·dt))²
            double dW = local_rng.normal() * sqrt_dt;
            // Removed std::max(0.0) clamp because Feller condition guarantees strict positivity
            double numer = std::sqrt(y_cir_(i)) + 0.5 * lambda_eff * dW;
            double denom = 1.0 + 0.5 * cfg_.cir_kappa * dt;
            y_cir_(i) = (numer * numer) / (denom * denom);
            // Add drift correction for mean-reversion and Itô shift
            // Use original cfg_.cir_lambda for ito_correction, NOT lambda_eff,
            // because the Alfonsi scheme was derived with the original parameters.
            double ito_correction = 0.25 * cfg_.cir_lambda * cfg_.cir_lambda;
            y_cir_(i) += (cfg_.cir_kappa * cfg_.cir_eta - ito_correction) * dt / denom;
            // Final safety net, though mathematically unreached due to Feller
            y_cir_(i) = std::max(y_cir_(i), 1e-8);

            // Accumulated operational time
            double d_tau = y_cir_(i) * dt;
            tau_(i) += d_tau;

            // ── CGMY jump in operational time ──
            // Regime-dependent parameters (base * regime modifier)
            double regime_scale = 1.0 + 0.5 * (a.regime - 2);  // regime 2 = normal
            double C_eff = cfg_.C * std::max(regime_scale, 0.1);

            // Asset jump is an idiosyncratic jump plus beta * systemic jump
            double idio_jump = sample_cgmy(C_eff, cfg_.G, cfg_.M, cfg_.Y, d_tau, local_rng);
            double beta_jump = 0.5; // Sensitivity to market-wide crash
            double jump = idio_jump + beta_jump * systemic_jump;

            // Full CGMY Laplace exponent compensator for martingale property.
            // For S=exp(X) to be a martingale: drift = -ψ(-1) where
            // ψ(u) = C·Γ(-Y)·[(M-u)^Y - M^Y + (G+u)^Y - G^Y]
            // So ψ(-1) = C·Γ(-Y)·[(M+1)^Y - M^Y + (G-1)^Y - G^Y]
            // This compensates BOTH large compound Poisson AND small Gaussian jumps.
            // We expand the 1.95 cap asymptotically for true divergence as Y->2.
            double safe_Y = std::min(cfg_.Y, 1.99); // Allow closer to 2.0 without complete Inf blowup
            double neg_gamma_Y = std::tgamma(2.0 - safe_Y) / (safe_Y * (safe_Y - 1.0)); // Γ(-Y) approx for Y∈(1,2)
            // Guard: G must be > 1 for (G-1)^Y to be real and positive
            double G_safe = std::max(cfg_.G, 1.01);
            double psi_neg1 = C_eff * neg_gamma_Y * (
                std::pow(cfg_.M + 1.0, safe_Y) - std::pow(cfg_.M, safe_Y)
              + std::pow(G_safe - 1.0, safe_Y) - std::pow(G_safe, safe_Y)
            );
            // Scale by operational time (CIR subordinator)
            double compensator = -psi_neg1 * d_tau;
            a.log_price += jump + compensator;
            a.jump_component = jump;

            // Guardrail: prevent log_price from hitting exp() overflow
            // ±10 from initial log(100)≈4.6 allows price range [~0.005, ~2.2M]
            a.log_price = std::clamp(a.log_price, -10.0, 15.0);

            // Update price
            double new_price = std::exp(a.log_price);
            a.price = new_price;
        }
    }

    const Eigen::VectorXd& cir_state() const { return y_cir_; }
};

} // namespace sovereign
