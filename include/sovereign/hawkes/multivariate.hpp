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

    // Excitation: α_{ij} for self and cross-asset [N×N]
    Eigen::MatrixXd alpha_;
    Eigen::MatrixXd beta_mat_;
    Eigen::MatrixXd epsilon_mat_;

    // Recursive auxiliary variables for sum-of-exponentials
    // A_m(i,j) tracks the decayed history contribution
    // Using 3-component sum-of-exp approximation of power-law
    static constexpr int N_EXP = 3;
    struct RecursiveState {
        double A[N_EXP] = {0, 0, 0};
        double last_time = 0;
    };
    // recursive_[i][j] = recursive state for influence j→i
    std::vector<std::vector<RecursiveState>> recursive_;

    // Sum-of-exp approximation of power-law kernel
    double exp_alpha_[N_EXP] = {0.3, 0.15, 0.05};
    double exp_beta_[N_EXP]  = {10.0, 1.0, 0.1};

    HawkesStabilityProjector projector_;

    /// Flatten index (asset, order_type, depth) → linear
    int idx(int asset, int order_type, int depth) const {
        return asset * K_ * D_ + order_type * D_ + depth;
    }

    void fit_sum_exp_to_powerlaw() {
        // Effective branching ratio from asset j to asset i:
        //   ρ_eff(i←j) = (α_ij/α_self) × Σ_m exp_alpha[m]/exp_beta[m]
        // Total branching FROM one event = Σ_j ρ_eff(i←j)
        //
        // We need Σ_j (α_ij/α_self) × B_self < ρ_max
        // where B_self = Σ_m exp_alpha[m]/exp_beta[m]
        //
        // Worst-case row sum of (α_ij/α_self): self=1, cross = N-1 cross/self
        // After Dykstra, alpha_ may be rescaled — compute actual max row sum
        double max_row_sum = 1.0;  // Self
        for (int j = 0; j < N_; ++j) {
            if (j != 0) max_row_sum += alpha_(0, j) / (cfg_.alpha_self + 1e-12);
        }
        // Target per-event branching ratio (self kernel only)
        double B_target = cfg_.max_spectral_radius / (max_row_sum + 1e-10);
        B_target = std::max(B_target, 0.01);  // Floor

        // Shape: fast/medium/slow components with fixed ratios
        // B_self = a*(0.6/10 + 0.3/1 + 0.1/0.1) = a*(0.06 + 0.3 + 1.0) = 1.36*a
        // Solve: 1.36 * a = B_target  →  a = B_target / 1.36
        double a = B_target / 1.36;
        exp_alpha_[0] = a * 0.6;  exp_beta_[0] = 10.0;
        exp_alpha_[1] = a * 0.3;  exp_beta_[1] = 1.0;
        exp_alpha_[2] = a * 0.1;  exp_beta_[2] = 0.1;

        // Verify
        double B_actual = 0;
        for (int m = 0; m < N_EXP; ++m)
            B_actual += exp_alpha_[m] / exp_beta_[m];
        // If still too high, rescale
        if (B_actual * max_row_sum > cfg_.max_spectral_radius) {
            double scale = cfg_.max_spectral_radius / (B_actual * max_row_sum + 1e-10);
            for (int m = 0; m < N_EXP; ++m) exp_alpha_[m] *= scale;
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

        // Cross-asset excitation matrix [N×N]
        alpha_ = Eigen::MatrixXd::Zero(N_, N_);
        beta_mat_ = Eigen::MatrixXd::Constant(N_, N_, cfg_.beta);
        epsilon_mat_ = Eigen::MatrixXd::Constant(N_, N_, cfg_.epsilon);

        for (int i = 0; i < N_; ++i) {
            alpha_(i, i) = cfg_.alpha_self;
            for (int j = 0; j < N_; ++j) {
                if (i != j) alpha_(i, j) = cfg_.alpha_cross;
            }
        }

        // Project for stationarity
        projector_.dykstra_project(alpha_, beta_mat_, epsilon_mat_);

        // Recursive state
        recursive_.resize(N_, std::vector<RecursiveState>(N_));

        fit_sum_exp_to_powerlaw();
    }

    /// Get current intensity for asset i, order type k, depth d
    double intensity(int i, int k, int d, double t) const {
        double lam = mu_(idx(i, k, d));
        // Add contributions from all assets via decayed history
        for (int j = 0; j < N_; ++j) {
            const auto& rs = recursive_[i][j];
            double a_ij = (j == i) ? cfg_.alpha_self : alpha_(i, j);
            double dt_last = t - rs.last_time;
            for (int m = 0; m < N_EXP; ++m) {
                // Normalise: exp_alpha_[m] was set proportional to alpha_self
                // We need a_ij-relative weight
                double w = (cfg_.alpha_self > 1e-12)
                           ? (a_ij / cfg_.alpha_self) : 0.0;
                lam += w * rs.A[m] * std::exp(-exp_beta_[m] * dt_last);
            }
        }
        return std::max(lam, 0.0);
    }

    /// Get aggregate intensity for asset i (sum over all k, d)
    double aggregate_intensity(int i, double t) const {
        double total = 0;
        for (int k = 0; k < K_; ++k)
            for (int d = 0; d < D_; ++d)
                total += intensity(i, k, d, t);
        return total;
    }

    /// Record an event: asset j fired at time t → update recursive states
    void record_event(int j, double t) {
        for (int i = 0; i < N_; ++i) {
            auto& rs = recursive_[i][j];
            double dt = t - rs.last_time;
            // Decay existing + add new impulse
            for (int m = 0; m < N_EXP; ++m) {
                rs.A[m] = rs.A[m] * std::exp(-exp_beta_[m] * dt) + exp_alpha_[m];
            }
            rs.last_time = t;
        }
    }

    /// Ogata thinning: generate events in [t, t+dt] for all assets.
    /// Hard cap: at most max_events_per_step to prevent explosion near criticality.
    void generate_events(SimulationClock& clock, SimulationState& state,
                         double dt, Xoshiro256& rng,
                         int max_events_per_step = 200)
    {
        double t_start = state.t;
        double t_end = t_start + dt;

        double lambda_bar = 0;
        for (int i = 0; i < N_; ++i)
            lambda_bar += aggregate_intensity(i, t_start);
        lambda_bar = std::min(lambda_bar * 1.5, 1e6);  // Hard cap on upper bound

        double t_cur = t_start;
        int n_accepted = 0;

        while (t_cur < t_end && n_accepted < max_events_per_step) {
            if (lambda_bar < 1e-10) break;
            double tau = rng.exponential(lambda_bar);
            t_cur += tau;
            if (t_cur >= t_end) break;

            double lambda_actual = 0;
            for (int i = 0; i < N_; ++i)
                lambda_actual += aggregate_intensity(i, t_cur);

            if (rng.uniform() * lambda_bar <= lambda_actual) {
                double u = rng.uniform() * lambda_actual;
                double cumul = 0;
                bool accepted = false;
                for (int i = 0; i < N_ && !accepted; ++i) {
                    for (int k = 0; k < K_ && !accepted; ++k) {
                        for (int d = 0; d < std::min(D_, 5) && !accepted; ++d) {
                            cumul += intensity(i, k, d, t_cur);
                            if (u <= cumul) {
                                Event ev;
                                ev.time = t_cur;
                                ev.asset_id = i;
                                ev.event_type = k;
                                ev.depth_level = d;
                                ev.size = 1.0 + rng.exponential(0.1);
                                ev.is_buy = rng.uniform() < 0.5;
                                clock.schedule(ev);
                                record_event(i, t_cur);
                                state.total_events++;
                                state.assets[i].hawkes_intensity(k * D_ + d) =
                                    intensity(i, k, d, t_cur);
                                ++n_accepted;
                                accepted = true;
                            }
                        }
                    }
                }
            }

            // Update upper bound
            lambda_bar = 0;
            for (int i = 0; i < N_; ++i)
                lambda_bar += aggregate_intensity(i, t_cur);
            lambda_bar = std::min(lambda_bar * 1.5, 1e6);
            lambda_bar = std::max(lambda_bar, 1.0);
        }
    }

    /// Modulate cross-asset excitation by graph distance
    void modulate_by_graph(const Eigen::MatrixXd& graph_distance) {
        for (int i = 0; i < N_; ++i)
            for (int j = 0; j < N_; ++j)
                if (i != j)
                    alpha_(i, j) = cfg_.alpha_cross
                                 * std::exp(-0.5 * graph_distance(i, j));
        projector_.dykstra_project(alpha_, beta_mat_, epsilon_mat_);
    }

    Eigen::VectorXd aggregate_intensities(double t) const {
        Eigen::VectorXd v(N_);
        for (int i = 0; i < N_; ++i) v(i) = aggregate_intensity(i, t);
        return v;
    }

    double total_branching_ratio() const {
        double B_self = 0;
        for (int m = 0; m < N_EXP; ++m)
            B_self += exp_alpha_[m] / exp_beta_[m];
        double max_row = 1.0;
        for (int j = 1; j < N_; ++j)
            max_row += alpha_(0, j) / (cfg_.alpha_self + 1e-12);
        return B_self * max_row;
    }
};

} // namespace sovereign
