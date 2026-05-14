#pragma once
/// @file gerber_shiu.hpp
/// @brief Layer 5 — Gerber-Shiu ruin theory with endogenous feedback.
///
/// References:
///   Gerber & Shiu — On the Time Value of Ruin
///   Li & Lu — Generalized Gerber-Shiu with interest (232.txt)
///
/// Surplus: U_i(t) = U_i(0) + ∫c(s)ds - Σ Z_n·1(T_n≤t)
/// Φ(u) = E[w(U(T⁻),|U(T)|)·exp(-δT)·1(T<∞) | U(0)=u]
/// Feedback: Γ→Layer4 (spread widening), Γ→Layer2 (Hawkes boost)

#include <sovereign/config.hpp>
#include <sovereign/core/random.hpp>
#include <sovereign/core/state.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace sovereign {

class RuinEngine {
    const RuinConfig& cfg_;
    int N_;

    // Per-asset surplus tracking
    Eigen::VectorXd surplus_;
    Eigen::VectorXd ruin_prob_;
    Eigen::VectorXd gerber_shiu_;

    // Claim history for each asset
    struct ClaimHistory {
        static constexpr int MAX_CLAIMS = 1000;
        std::vector<double> times;
        std::vector<double> sizes;
        int head = 0;
        double total_claims = 0;
        int n_claims = 0;
        ClaimHistory() : times(MAX_CLAIMS, 0.0), sizes(MAX_CLAIMS, 0.0) {}
    };
    std::vector<ClaimHistory> claims_;

    /// Cramér-Lundberg approximation for ruin probability:
    /// Ψ(u) ≈ (λ·μ_Z / c) · exp(-R·u)  where R is adjustment coefficient
    /// R solves: λ(M_Z(R) - 1) = c·R  (Lundberg equation)
    double cramer_lundberg(double u, double claim_rate, double mean_claim,
                           double premium_rate) const
    {
        if (premium_rate <= claim_rate * mean_claim) return 1.0; // Net loss
        // Exponential claims: R = θ/((1+θ)·μ)  [Gerber & Shiu 1998 Eq 2.22]
        double theta = premium_rate / (claim_rate * mean_claim) - 1.0;
        if (theta <= 0) return 1.0;
        double R = theta / ((1.0 + theta) * mean_claim);  // CORRECT per paper
        double psi_0 = 1.0 / (1.0 + theta);  // Ψ(0)
        return psi_0 * std::exp(-R * u);
    }

    /// Gerber-Shiu penalty function via numerical integration
    /// IDE: (λ+δ)Φ = cΦ' + λ∫₀ᵘ Φ(u-x)p(x)dx + λω(u)
    /// Discretize on grid and solve via iteration
    double solve_gerber_shiu(double u, double claim_rate, double mean_claim,
                              double premium_rate, double delta) const
    {
        const int N_GRID = 200;
        double du = u / N_GRID;
        if (du < 1e-10) return 1.0;

        std::vector<double> phi(N_GRID + 1, 0.0);
        // Boundary: Φ(0) = w(0, 0) = 1 (simple penalty)
        phi[0] = 1.0;

        // Picard iteration
        for (int iter = 0; iter < 50; ++iter) {
            std::vector<double> phi_new(N_GRID + 1, 0.0);
            phi_new[0] = 1.0;

            for (int i = 1; i <= N_GRID; ++i) {
                double ui = i * du;
                // Convolution: ∫₀ᵘ Φ(u-x)·p(x)dx
                // p(x) = (1/μ)·exp(-x/μ) for exponential claims
                double conv = 0;
                for (int j = 0; j < i; ++j) {
                    double xj = (i - j) * du;
                    conv += phi[j] * std::exp(-xj / mean_claim) / mean_claim * du;
                }

                // ω(u) = ∫_u^∞ w(u, x-u)·p(x)dx ≈ exp(-u/μ)/μ
                double omega = std::exp(-ui / mean_claim) / mean_claim;

                // IDE discretization: c·Φ'(u) - (λ+δ)Φ(u) + λ·conv + λ·ω = 0
                // c·(Φ(u) - Φ(u-du))/du - (λ+δ)Φ(u) + λ·conv + λ·ω = 0
                // Φ(u)·(c/du - (λ+δ)) = c·Φ(u-du)/du - λ·conv - λ·ω
                double rhs = premium_rate * phi[i - 1] / du
                           - claim_rate * conv - claim_rate * omega;
                double denom = premium_rate / du - (claim_rate + delta);
                
                // If denominator <= 0, process is structurally doomed (premium too low)
                if (denom <= 1e-10) {
                    phi_new[i] = 1.0;
                } else {
                    phi_new[i] = rhs / denom;
                    phi_new[i] = std::clamp(phi_new[i], 0.0, 1.0);
                }
            }

            // Check convergence
            double diff = 0;
            for (int i = 0; i <= N_GRID; ++i)
                diff += std::abs(phi_new[i] - phi[i]);
            phi = phi_new;
            if (diff < 1e-8) return phi[N_GRID];
        }

        // Divergence Fallback: Picard iteration failed to converge
        return 0.99;
    }

public:
    RuinEngine(const RuinConfig& cfg, int n_assets)
        : cfg_(cfg), N_(n_assets)
    {
        surplus_     = Eigen::VectorXd::Constant(N_, cfg_.initial_surplus);
        ruin_prob_   = Eigen::VectorXd::Zero(N_);
        gerber_shiu_ = Eigen::VectorXd::Zero(N_);
        claims_.resize(N_);
    }

    /// Process adverse selection event as a "claim" against surplus
    void register_claim(int asset, double time, double size) {
        auto& c = claims_[asset];
        if (c.n_claims >= ClaimHistory::MAX_CLAIMS) {
            c.total_claims -= c.sizes[c.head];
        } else {
            c.n_claims++;
        }
        c.times[c.head] = time;
        c.sizes[c.head] = size;
        c.total_claims += size;
        c.head = (c.head + 1) % ClaimHistory::MAX_CLAIMS;
        surplus_(asset) -= size;
    }

    /// Advance surplus dynamics by dt
    void step(SimulationState& state, double dt, Xoshiro256& rng) {
        for (int i = 0; i < N_; ++i) {
            auto& a = state.assets[i];

            // Premium = spread × fill_rate × dt
            // fill_rate = 200 fills/unit-time (realistic MM activity scale)
            double min_spread = std::max(a.price * 0.0005, 0.01);
            double eff_spread = std::max(a.mm_spread, min_spread);
            double fill_rate  = 200.0;  // fills per unit simulation time
            double premium_rate = eff_spread * fill_rate;
            surplus_(i) += premium_rate * dt;

            // Adverse selection claims — only during actual book exhaustion
            double lob_stress = std::abs(a.lob_impact) * 1e4;
            if (lob_stress > 10.0) {  // crisis-level book walkthrough only
                double claim = std::min(lob_stress * cfg_.initial_surplus * 0.0001,
                                       cfg_.initial_surplus * 0.001);
                register_claim(i, state.t, claim);
            }

            // Poisson background claims (rate 0.05/year — rare)
            if (rng.uniform() < 0.05 * dt) {
                double claim = rng.exponential(1.0 / (cfg_.initial_surplus * 0.002));
                register_claim(i, state.t, claim);
            }

            surplus_(i) = std::max(surplus_(i), -cfg_.initial_surplus * 10);
            a.surplus = surplus_(i);

            // Only compute ruin probability with enough statistical data
            if (state.t < 0.1 || claims_[i].n_claims < 5) {
                ruin_prob_(i) = 0.0;
                a.ruin_prob = 0.0;
                continue;
            }

            // Claim rate: use ring buffer window when full, else global time
            auto& ch = claims_[i];
            double window;
            if (ch.n_claims >= ClaimHistory::MAX_CLAIMS) {
                // Buffer wrapped: use actual ring buffer time span
                double t_oldest = ch.times[ch.head];
                double t_newest = ch.times[(ch.head - 1 + ClaimHistory::MAX_CLAIMS)
                                           % ClaimHistory::MAX_CLAIMS];
                window = std::max(t_newest - t_oldest, dt);
            } else {
                // Buffer not full: use global sim time
                window = std::max(state.t, dt);
            }
            int effective_claims = std::min(ch.n_claims, ClaimHistory::MAX_CLAIMS);
            double claim_rate = static_cast<double>(effective_claims) / window;
            double mean_claim = ch.total_claims / ch.n_claims;

            ruin_prob_(i) = cramer_lundberg(
                std::max(surplus_(i), 0.0), claim_rate, mean_claim, premium_rate);
            ruin_prob_(i) = std::clamp(ruin_prob_(i), 0.0, 1.0);
            a.ruin_prob = ruin_prob_(i);

            // Gerber-Shiu (expensive — compute less frequently)
            if (state.step % 200 == 0 && surplus_(i) > 0) {
                gerber_shiu_(i) = solve_gerber_shiu(
                    surplus_(i), claim_rate, mean_claim,
                    premium_rate, cfg_.delta_discount);
                a.gerber_shiu = gerber_shiu_(i);
            }
        }

        // Blend local ruin into global contagion vector with decay.
        // Old cwiseMax was a one-way ratchet (ruin could never decrease).
        // Now: ruin_vector exponentially decays toward local ruin_prob_
        // Exact analytical integration prevents Explicit Euler explosions when decay_rate * dt > 1
        double decay_rate = 5.0;  // reversion speed to local fundamental risk
        double exp_decay = std::exp(-decay_rate * dt);
        for (int i = 0; i < N_; ++i) {
            state.ruin_vector(i) = ruin_prob_(i) + (state.ruin_vector(i) - ruin_prob_(i)) * exp_decay;
            state.ruin_vector(i) = std::clamp(state.ruin_vector(i), 0.0, 1.0);
        }
    }

    const Eigen::VectorXd& surplus() const { return surplus_; }
    const Eigen::VectorXd& ruin_probs() const { return ruin_prob_; }
};

} // namespace sovereign
