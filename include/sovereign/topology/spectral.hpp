#pragma once
/// @file spectral.hpp
/// @brief Layer 6.3 — Spectral diagnostics: Fiedler value, centrality, clustering.
///
/// Fiedler value λ₂(Laplacian) → fragmentation crisis when → 0
/// Betweenness centrality → systemically important assets

#include <sovereign/core/state.hpp>
#include <sovereign/topology/graphs.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <vector>

namespace sovereign {

class SpectralEngine {
    int N_;

public:
    explicit SpectralEngine(int n_assets) : N_(n_assets) {}

    /// Compute graph Laplacian from adjacency and its Fiedler value
    void update(SimulationState& state, const GraphEngine& graphs) {
        // Build adjacency from distance matrix: w_ij = exp(-d_ij)
        // This ensures graph is always fully connected (w > 0 always).
        // d_ij = sqrt(2(1 - ρ_ij)) ∈ [0, 2], so w_ij ∈ [exp(-2), 1]
        Eigen::MatrixXd W(N_, N_);
        for (int i = 0; i < N_; ++i) {
            W(i, i) = 0;
            for (int j = i + 1; j < N_; ++j) {
                double rho = std::clamp(state.correlation(i, j), -1.0, 1.0);
                double d   = std::sqrt(std::max(2.0 * (1.0 - rho), 0.0));
                double w   = std::exp(-d);
                W(i, j) = W(j, i) = w;
            }
        }

        // Laplacian: L = D - W
        Eigen::MatrixXd L = Eigen::MatrixXd::Zero(N_, N_);
        for (int i = 0; i < N_; ++i) {
            double deg = W.row(i).sum();
            L(i, i) = deg;
            state.degree(i) = deg;
            for (int j = 0; j < N_; ++j)
                if (i != j) L(i, j) = -W(i, j);
        }

        // Eigendecompose Laplacian
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(L);
        auto evals = solver.eigenvalues();

        // Fiedler = algebraic connectivity (smallest non-zero eigenvalue)
        state.fiedler_value = 0.0;
        for (int i = 1; i < N_; ++i) {
            if (evals(i) > 1e-6) {
                state.fiedler_value = evals(i);
                break;
            }
        }

        compute_betweenness(state, graphs);
        compute_clustering(state, graphs);
    }

    void compute_betweenness(SimulationState& state, const GraphEngine& graphs) const {
        state.betweenness.setZero(N_);
        const auto& adj = graphs.pmfg_adjacency();

        // Brandes' Algorithm for exact Betweenness Centrality on graphs with cycles
        for (int s = 0; s < N_; ++s) {
            std::vector<int> S;
            std::vector<std::vector<int>> P(N_);
            std::vector<double> sigma(N_, 0.0);
            sigma[s] = 1.0;
            std::vector<int> d(N_, -1);
            d[s] = 0;
            std::vector<int> Q;
            Q.push_back(s);

            int head = 0;
            while(head < static_cast<int>(Q.size())) {
                int v = Q[head++];
                S.push_back(v);
                for (int w : adj[v]) {
                    // w found for the first time?
                    if (d[w] < 0) {
                        Q.push_back(w);
                        d[w] = d[v] + 1;
                    }
                    // shortest path to w via v?
                    if (d[w] == d[v] + 1) {
                        sigma[w] += sigma[v];
                        P[w].push_back(v);
                    }
                }
            }

            std::vector<double> delta(N_, 0.0);
            while(!S.empty()) {
                int w = S.back();
                S.pop_back();
                for (int v : P[w]) {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
                if (w != s) {
                    state.betweenness(w) += delta[w];
                }
            }
        }

        // For undirected graphs, divide by 2
        state.betweenness /= 2.0;

        // Normalize to [0, 1] using theoretical maximum for N nodes
        // Do NOT normalize dynamically per-tick, which erases macro-structural shifts!
        if (N_ > 2) {
            double theoretical_max = (N_ - 1.0) * (N_ - 2.0) / 2.0;
            state.betweenness /= theoretical_max;
        }
    }

    void compute_clustering(SimulationState& state, const GraphEngine& graphs) const {
        double total = 0;
        const auto& adj = graphs.pmfg_adjacency();
        for (int i = 0; i < N_; ++i) {
            int ki = adj[i].size(); // Combinatorial degree
            if (ki < 2) continue;
            int triangles = 0;
            // Count actual triangles in the graph
            for (int j : adj[i]) {
                for (int k : adj[i]) {
                    if (j >= k) continue;
                    // Check if j and k are connected
                    if (std::find(adj[j].begin(), adj[j].end(), k) != adj[j].end()) {
                        triangles++;
                    }
                }
            }
            total += 2.0 * triangles / (ki * (ki - 1));
        }
        state.clustering_coeff = total / N_;
    }
};

} // namespace sovereign
