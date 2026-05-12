#pragma once
/// @file regime.hpp
/// @brief Layer 1.2b — Hidden Markov regime switching (K=5 states).
///
/// Regime transitions are doubly-stochastic: base Q-matrix modulated
/// by aggregate Hawkes intensity (high activity → faster transitions).

#include <sovereign/config.hpp>
#include <sovereign/core/random.hpp>
#include <sovereign/core/state.hpp>
#include <Eigen/Dense>
#include <cmath>

namespace sovereign {

class RegimeEngine {
    const RegimeConfig& cfg_;
    int N_;
    int K_;

    /// Generator (Q-matrix) for the continuous-time Markov chain [K×K]
    /// Q_ij = rate of transition from i to j, Q_ii = -Σ_{j≠i} Q_ij
    Eigen::MatrixXd Q_;

    /// Regimes: 0=crisis, 1=stressed, 2=normal, 3=calm, 4=euphoric
    void build_generator() {
        K_ = cfg_.n_regimes;
        Q_ = Eigen::MatrixXd::Zero(K_, K_);

        // Tridiagonal + nearest-neighbor transitions
        // Higher rates for crisis↔stressed (regime dynamics faster at extremes)
        for (int i = 0; i < K_; ++i) {
            if (i > 0)     Q_(i, i-1) = 0.5 + 0.3 * (K_ - 1 - i);
            if (i < K_-1)  Q_(i, i+1) = 0.5 + 0.3 * i;
            Q_(i, i) = -Q_.row(i).sum() + Q_(i, i);  // Row sum = 0
        }
    }

public:
    RegimeEngine(const RegimeConfig& cfg, int n_assets)
        : cfg_(cfg), N_(n_assets), K_(cfg.n_regimes)
    {
        build_generator();
    }

    /// Advance regime for all assets by dt.
    /// hawkes_agg: aggregate Hawkes intensity per asset (modulates transition rate)
    void step(SimulationState& state, double dt,
              const Eigen::VectorXd& hawkes_agg, Xoshiro256& rng)
    {
        for (int i = 0; i < N_; ++i) {
            auto& a = state.assets[i];
            int k = a.regime;

            // Modulate transition rates by Hawkes intensity
            // Higher intensity → faster transitions (doubly stochastic)
            double intensity_mod = 1.0 + 0.1 * std::max(hawkes_agg(i) - 10.0, 0.0);

            // Check for transition: competing exponentials
            double total_rate = 0.0;
            for (int j = 0; j < K_; ++j) {
                if (j != k) total_rate += Q_(k, j) * intensity_mod;
            }

            // Probability of any transition in [t, t+dt]
            double p_transition = 1.0 - std::exp(-total_rate * dt);
            if (rng.uniform() < p_transition) {
                // Which state do we jump to? Proportional to rates
                double u = rng.uniform() * total_rate;
                double cumulative = 0;
                for (int j = 0; j < K_; ++j) {
                    if (j == k) continue;
                    cumulative += Q_(k, j) * intensity_mod;
                    if (u <= cumulative) {
                        a.regime = j;
                        break;
                    }
                }
            }
        }
    }

    const Eigen::MatrixXd& generator() const { return Q_; }
};

} // namespace sovereign
