#pragma once
/// @file multivariate.hpp
/// @brief Layer 2 — Multivariate tensor Hawkes process (N×K×D).
///
/// Intensity: λ_i^{k,d}(t) = μ_i^{k,d} + Σ_j Σ_l Σ_e ∫₀ᵗ φ_{ij}^{kl,de}(t-s) dN_j^{l,e}(s)
///
/// For computational tractability, we use sum-of-exponentials kernels
/// with O(1) recursive intensity updates (no history storage needed).
///
/// Simulation via Ogata's modified thinning algorithm.

#include <sovereign/config.hpp>
#include <sovereign/core/clock.hpp>
#include <sovereign/core/random.hpp>
#include <sovereign/core/state.hpp>
#include <sovereign/hawkes/kernels.hpp>
#include <sovereign/hawkes/stability.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace sovereign {

class HawkesEngine {
    const HawkesConfig& cfg_;
    int N_, K_, D_;
    int dim_;  ///< Total dimension = N*K*D

    // Baseline intensity μ [dim_]
    Eigen::VectorXd mu_;

    // Per-asset baseline modulation (ruin -> Hawkes feedback)
    Eigen::VectorXd baseline_mod_;

    // Excitation: α_{ij} for self and cross-asset [N×N]
    Eigen::MatrixXd alpha_;
    Eigen::MatrixXd beta_mat_;
    Eigen::MatrixXd epsilon_mat_;

    // Recursive auxiliary variables for sum-of-exponentials
    // A_m(i,j) tracks the decayed history contribution
    // 10-component sum-of-exp: proper long-memory power-law approximation
    // (Hardiman, Bercot & Bouchaud 2013 — 3 terms loses critical reflexivity)
    static constexpr int N_EXP = 10;
    struct RecursiveState {
        double A[N_EXP] = {};
        double last_time = 0;
    };
    std::vector<std::vector<RecursiveState>> recursive_;

    double exp_alpha_[N_EXP] = {};
    double exp_beta_[N_EXP]  = {};

    // Max single-event jump for thinning bound
    double max_alpha_sum_ = 0;

    HawkesStabilityProjector projector_;

    int idx(int asset, int order_type, int depth) const {
        return asset * K_ * D_ + order_type * D_ + depth;
    }

    void fit_sum_exp_to_powerlaw() {
        double max_row_sum = 0.0;
        for (int i = 0; i < N_; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < N_; ++j) {
                if (i != j) row_sum += alpha_(i, j) / (cfg_.alpha_self + 1e-12);
            }
            max_row_sum = std::max(max_row_sum, row_sum);
        }
        double B_target = cfg_.max_spectral_radius / (max_row_sum + 1e-10);
        B_target = std::max(B_target, 0.01);

        // 10 log-spaced decay rates from 100 (fast) to 0.01 (slow)
        // Memory horizon: 1/beta_min = 100 time units
        double B_sum = 0;
        double eps = cfg_.epsilon;  // Power-law tail index ∈ (0,1)
        for (int m = 0; m < N_EXP; ++m) {
            exp_beta_[m] = 100.0 * std::pow(0.01 / 100.0, (double)m / (N_EXP - 1));
            // Power-law weights: α_m ∝ β_m^{-ε} (Lima & Choi 1805.09570v3)
            // This correctly approximates φ(t) = (1+t/β)^{-(1+ε)} via
            // sum-of-exponentials, preserving true long-memory structure.
            // Uniform weights (old code) degraded this to generic multi-exp decay.
            exp_alpha_[m] = std::pow(exp_beta_[m], -eps);
            B_sum += exp_alpha_[m] / exp_beta_[m];
        }
        // Scale so total branching ratio = B_target
        double scale = B_target / B_sum;
        max_alpha_sum_ = 0;
        for (int m = 0; m < N_EXP; ++m) {
            exp_alpha_[m] *= scale;
            max_alpha_sum_ += exp_alpha_[m];
        }

        // Final safety check
        double B_actual = 0;
        for (int m = 0; m < N_EXP; ++m)
            B_actual += exp_alpha_[m] / exp_beta_[m];
        if (B_actual * max_row_sum > cfg_.max_spectral_radius) {
            double s = cfg_.max_spectral_radius / (B_actual * max_row_sum + 1e-10);
            for (int m = 0; m < N_EXP; ++m) exp_alpha_[m] *= s;
            max_alpha_sum_ *= s;
        }
    }

public:
    HawkesEngine(const HawkesConfig& cfg, int n_assets)
        : cfg_(cfg), N_(n_assets), K_(cfg.n_order_types), D_(cfg.n_depth_levels),
          dim_(n_assets * cfg.n_order_types * cfg.n_depth_levels),
          projector_(cfg.max_spectral_radius, cfg.dykstra_max_iter)
    {
        // Baseline intensity: divide by K*D so aggregate per-asset = base_intensity
        double mu_per_dim = cfg_.base_intensity / (K_ * D_);
        mu_ = Eigen::VectorXd::Constant(dim_, mu_per_dim);
        baseline_mod_ = Eigen::VectorXd::Ones(N_);

        // Cross-asset excitation matrix [N×N]
        alpha_ = Eigen::MatrixXd::Zero(N_, N_);
        beta_mat_ = Eigen::MatrixXd::Constant(N_, N_, cfg_.beta);
        epsilon_mat_ = Eigen::MatrixXd::Constant(N_, N_, cfg_.epsilon);

        for (int i = 0; i < N_; ++i) {
            // Power-law scale (Zipf's law) to simulate heterogeneous market capitalization
            double cap_scale = 1.0 / std::sqrt(static_cast<double>(i + 1));
            alpha_(i, i) = cfg_.alpha_self * cap_scale;
            for (int j = 0; j < N_; ++j) {
                if (i != j) alpha_(i, j) = cfg_.alpha_cross * cap_scale;
            }
        }

        // Project for stationarity at init (not just after modulate_by_graph)
        projector_.dykstra_project(alpha_, beta_mat_, epsilon_mat_);

        // Recursive state
        recursive_.resize(N_, std::vector<RecursiveState>(N_));

        fit_sum_exp_to_powerlaw();
    }

    /// Compute the shared excitation component for asset i at time t
    double compute_excitation(int i, double t) const {
        double lam = 0;
        for (int j = 0; j < N_; ++j) {
            const auto& rs = recursive_[i][j];
            double a_ij = (j == i) ? cfg_.alpha_self : alpha_(i, j);
            double w = (cfg_.alpha_self > 1e-12) ? (a_ij / cfg_.alpha_self) : 0.0;
            if (w < 1e-15) continue;  // skip zero-weight pairs

            double dt_last = std::max(0.0, t - rs.last_time); // Guard against inverted times
            
            double contrib = 0;
            for (int m = 0; m < N_EXP; ++m)
                contrib += rs.A[m] * std::exp(-exp_beta_[m] * dt_last);
            lam += w * contrib;
        }
        return lam;
    }

    /// Get current intensity for asset i, order type k, depth d, given precomputed excitation
    double intensity(int i, int k, int d, double exc) const {
        return std::max(mu_(idx(i, k, d)) * baseline_mod_(i) + exc, 0.0);
    }

    /// Get aggregate intensity for asset i using precomputed excitation
    double aggregate_intensity(int i, double exc) const {
        double total_mu = 0;
        for (int k = 0; k < K_; ++k)
            for (int d = 0; d < D_; ++d)
                total_mu += mu_(idx(i, k, d)) * baseline_mod_(i);
        return std::max(total_mu + K_ * D_ * exc, 0.0);
    }

    /// Record an event: asset j fired at time t with mark (size)
    /// Excitation proportional to mark: big orders excite more
    void record_event(int j, double t, double mark_size = 1.0) {
        // Normalize mark around 1.0: mark_size from exponential(0.1) has mean ~11
        // Divide by mean to preserve configured branching ratio
        constexpr double MEAN_MARK = 11.0;  // 1.0 + E[Exp(0.1)] = 1 + 10
        double mark_weight = std::clamp(mark_size / MEAN_MARK, 0.1, 3.0);
        for (int i = 0; i < N_; ++i) {
            auto& rs = recursive_[i][j];
            double dt = std::max(0.0, t - rs.last_time);
            // Decay existing + add mark-weighted impulse
            for (int m = 0; m < N_EXP; ++m) {
                rs.A[m] = rs.A[m] * std::exp(-exp_beta_[m] * dt)
                        + exp_alpha_[m] * mark_weight;
            }
            rs.last_time = t;
        }
    }

    /// Ogata thinning: generate events in [t, t+dt] for all assets.
    /// Simulates true endogenous avalanches without artificial limits.
    void generate_events(SimulationClock& clock, SimulationState& state,
                         double dt, Xoshiro256& rng)
    {
        double t_start = state.t;
        double t_end = t_start + dt;

        // Correct thinning bound: current intensity + max possible jump
        // An event adds mark_weight * sum(w * alpha) to EVERY (k, d) bin for EVERY asset
        double lambda_bar = 0;
        std::vector<double> current_exc(N_);
        for (int i = 0; i < N_; ++i) {
            current_exc[i] = compute_excitation(i, t_start);
            lambda_bar += aggregate_intensity(i, current_exc[i]);
        }

        double t_cur = t_start;
        int n_accepted = 0;

        while (t_cur < t_end) {
            if (lambda_bar < 1e-10) break;
            if (n_accepted >= 1000000) break;  // Allow massive avalanches, but stop true infinite loops
            double tau = rng.exponential(lambda_bar);
            t_cur += tau;
            if (t_cur >= t_end) break;

            double lambda_actual = 0;
            for (int i = 0; i < N_; ++i) {
                current_exc[i] = compute_excitation(i, t_cur);
                lambda_actual += aggregate_intensity(i, current_exc[i]);
            }

            if (rng.uniform() * lambda_bar <= lambda_actual) {
                double u = rng.uniform() * lambda_actual;
                double cumul = 0;
                bool accepted = false;
                for (int i = 0; i < N_ && !accepted; ++i) {
                    for (int k = 0; k < K_ && !accepted; ++k) {
                        for (int d = 0; d < D_ && !accepted; ++d) { // Check all D_ levels
                            cumul += intensity(i, k, d, current_exc[i]);
                            if (u <= cumul) {
                                Event ev;
                                ev.time = t_cur;
                                ev.asset_id = i;
                                ev.event_type = k;
                                ev.depth_level = d;
                                ev.size = 1.0 + rng.exponential(0.1);
                                double alpha_skew = 0.1 * (baseline_mod_(i) - 1.0); // Panic momentum
                                ev.is_buy = rng.uniform() < std::clamp(0.5 - alpha_skew, 0.1, 0.9);
                                clock.schedule(ev);
                                record_event(i, t_cur, ev.size);  // mark-dependent excitation
                                // Unsynchronized increment of state.total_events — safe since generate_events is single-threaded
                                state.total_events++;
                                // Update hawkes_intensity array correctly for the single accepted event
                                state.assets[i].hawkes_intensity(k * D_ + d) =
                                    intensity(i, k, d, compute_excitation(i, t_cur));
                                ++n_accepted;
                                accepted = true;
                            }
                        }
                    }
                }
                
                // Ogata update: The intensity strictly decays between events.
                // The true supremum for the remaining interval is the exact intensity just AFTER the event.
                lambda_bar = 0;
                for (int i = 0; i < N_; ++i) {
                    lambda_bar += aggregate_intensity(i, compute_excitation(i, t_cur));
                }
            }
            // FIX #15: Lower floor from 1.0 to 1e-6. The old floor wasted
            // 10,000+ proposals per tick during low-activity periods.
            lambda_bar = std::max(lambda_bar, 1e-6);
        }
    }

    /// Modulate cross-asset excitation by graph distance.
    /// FIX #16: Preserve Zipf cap_scale (1/√(i+1)) that was being erased.
    void modulate_by_graph(const Eigen::MatrixXd& graph_distance) {
        for (int i = 0; i < N_; ++i) {
            double cap_scale_i = 1.0 / std::sqrt(static_cast<double>(i + 1));
            for (int j = 0; j < N_; ++j)
                if (i != j)
                    alpha_(i, j) = cfg_.alpha_cross * cap_scale_i
                                 * std::exp(-0.5 * graph_distance(i, j));
        }
        projector_.dykstra_project(alpha_, beta_mat_, epsilon_mat_);
    }

    /// Ruin -> Hawkes feedback: modulate baseline intensity
    void set_baseline_modulation(int asset, double mod) {
        if (asset >= 0 && asset < N_)
            baseline_mod_(asset) = std::clamp(mod, 0.5, 5.0);
    }

    void aggregate_intensities(double t, Eigen::VectorXd& out) const {
        for (int i = 0; i < N_; ++i) out(i) = aggregate_intensity(i, compute_excitation(i, t));
    }

    double total_branching_ratio() const {
        double B_self = 0;
        for (int m = 0; m < N_EXP; ++m)
            B_self += exp_alpha_[m] / exp_beta_[m];
        double max_row = 0.0;
        for (int i = 0; i < N_; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < N_; ++j) {
                if (i != j) row_sum += alpha_(i, j) / (cfg_.alpha_self + 1e-12);
            }
            max_row = std::max(max_row, row_sum);
        }
        return B_self * max_row;
    }
};

} // namespace sovereign
