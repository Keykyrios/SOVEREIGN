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

        // Fiedler = second-smallest eigenvalue (always > 0 for connected graph)
        state.fiedler_value = (N_ > 1) ? std::max(evals(1), 0.0) : 0.0;

        compute_betweenness(state, graphs);
        compute_clustering(state);
    }

    void compute_betweenness(SimulationState& state, const GraphEngine& graphs) const {
        state.betweenness.setZero(N_);
        const auto& adj = graphs.mst_adjacency();

        // For MST: betweenness of edge (u,v) = n_u * n_v
        // where n_u, n_v are sizes of subtrees on each side
        // Vertex betweenness ≈ sum of edge betweennesses through it
        for (int v = 0; v < N_; ++v) {
            // BFS from v, count reachable nodes through each neighbor
            for (int nb : adj[v]) {
                std::vector<bool> visited(N_, false);
                visited[v] = true;
                int count = 0;
                std::vector<int> queue = {nb};
                visited[nb] = true;
                while (!queue.empty()) {
                    std::vector<int> next;
                    for (int u : queue) {
                        count++;
                        for (int w : adj[u])
                            if (!visited[w]) { visited[w] = true; next.push_back(w); }
                    }
                    queue = std::move(next);
                }
                state.betweenness(v) += count * (N_ - count);
            }
        }
        // Normalize
        double max_b = state.betweenness.maxCoeff();
        if (max_b > 0) state.betweenness /= max_b;
    }

    void compute_clustering(SimulationState& state) const {
        double total = 0;
        for (int i = 0; i < N_; ++i) {
            int ki = static_cast<int>(state.degree(i));
            if (ki < 2) continue;
            int triangles = 0;
            // Count triangles through i using correlation threshold
            for (int j = 0; j < N_; ++j) {
                if (j == i || state.correlation(i, j) < 0.3) continue;
                for (int k = j + 1; k < N_; ++k) {
                    if (k == i) continue;
                    if (state.correlation(i, k) > 0.3 &&
                        state.correlation(j, k) > 0.3)
                        triangles++;
                }
            }
            total += 2.0 * triangles / (ki * (ki - 1) + 1e-10);
        }
        state.clustering_coeff = total / N_;
    }
};

} // namespace sovereign
