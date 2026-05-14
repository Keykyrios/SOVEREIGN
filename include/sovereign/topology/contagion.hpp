#pragma once
/// @file contagion.hpp
/// @brief Layer 6.4 — Laplacian diffusion of ruin probability.
///
/// ∂Γ/∂τ = -D·L(t)·Γ(τ) where L is graph Laplacian, D is diffusivity.
/// High ruin at node i spreads to neighbors via correlation topology.

#include <sovereign/config.hpp>
#include <sovereign/core/state.hpp>
#include <Eigen/Dense>
#include <cmath>

namespace sovereign {

class ContagionEngine {
    const TopologyConfig& cfg_;
    int N_;

    // Cached LLT — only recomputed when topology changes
    Eigen::LLT<Eigen::MatrixXd> cached_llt_;
    bool llt_valid_ = false;

public:
    ContagionEngine(const TopologyConfig& cfg, int n_assets)
        : cfg_(cfg), N_(n_assets) {}

    /// Signal that topology has changed and LLT needs recomputation
    void invalidate_cache() { llt_valid_ = false; }

    /// Diffuse ruin probability through the correlation network.
    /// Uses Implicit Euler: (I + D·dt·L)·Γ(t+dt) = Γ(t)
    /// Unconditionally stable — no CFL constraint on dt.
    void step(SimulationState& state, double dt) {
        if (!llt_valid_) {
            // Build Laplacian from correlation
            Eigen::MatrixXd W = state.correlation.cwiseMax(0.0);
            for (int i = 0; i < N_; ++i) W(i, i) = 0;

            Eigen::MatrixXd L = Eigen::MatrixXd::Zero(N_, N_);
            for (int i = 0; i < N_; ++i) {
                L(i, i) = W.row(i).sum();
                for (int j = 0; j < N_; ++j)
                    if (i != j) L(i, j) = -W(i, j);
            }

            Eigen::MatrixXd A = Eigen::MatrixXd::Identity(N_, N_)
                              + cfg_.contagion_diffusivity * dt * L;
            cached_llt_.compute(A);
            llt_valid_ = true;
        }

        state.ruin_vector = cached_llt_.solve(state.ruin_vector);

        // Clamp to [0, 1]
        for (int i = 0; i < N_; ++i) {
            state.ruin_vector(i) = std::clamp(state.ruin_vector(i), 0.0, 1.0);
            state.assets[i].ruin_prob = state.ruin_vector(i);
        }
    }
};

} // namespace sovereign
