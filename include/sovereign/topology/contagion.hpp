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

public:
    ContagionEngine(const TopologyConfig& cfg, int n_assets)
        : cfg_(cfg), N_(n_assets) {}

    /// Diffuse ruin probability through the correlation network
    void step(SimulationState& state, double dt) {
        // Build Laplacian from correlation
        Eigen::MatrixXd W = state.correlation.cwiseMax(0.0);
        for (int i = 0; i < N_; ++i) W(i, i) = 0;

        Eigen::MatrixXd L = Eigen::MatrixXd::Zero(N_, N_);
        for (int i = 0; i < N_; ++i) {
            L(i, i) = W.row(i).sum();
            for (int j = 0; j < N_; ++j)
                if (i != j) L(i, j) = -W(i, j);
        }

        // Explicit Euler: Γ(t+dt) = Γ(t) - D·dt·L·Γ(t)
        Eigen::VectorXd dGamma = -cfg_.contagion_diffusivity * dt * L * state.ruin_vector;
        state.ruin_vector += dGamma;

        // Clamp to [0, 1]
        for (int i = 0; i < N_; ++i) {
            state.ruin_vector(i) = std::clamp(state.ruin_vector(i), 0.0, 1.0);
            state.assets[i].ruin_prob = state.ruin_vector(i);
        }
    }
};

} // namespace sovereign
