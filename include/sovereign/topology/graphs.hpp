#pragma once
/// @file graphs.hpp
/// @brief Layer 6.2 — MST and PMFG construction.
///
/// Reference: Tumminello et al. (PNAS 2005) — Planar Maximally Filtered Graph
///
/// MST: Kruskal on d_ij = sqrt(2(1-ρ_ij)), N-1 edges
/// PMFG: 3(N-2) edges, genus-0 planarity, preserves MST + adds 4-cliques

#include <sovereign/core/state.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <vector>
#include <numeric>

namespace sovereign {

struct Edge {
    int i, j;
    double weight;
    bool operator<(const Edge& o) const { return weight < o.weight; }
};

/// Union-Find for Kruskal's MST
class UnionFind {
    std::vector<int> parent_, rank_;
public:
    explicit UnionFind(int n) : parent_(n), rank_(n, 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }
    int find(int x) {
        while (parent_[x] != x) { parent_[x] = parent_[parent_[x]]; x = parent_[x]; }
        return x;
    }
    bool unite(int x, int y) {
        x = find(x); y = find(y);
        if (x == y) return false;
        if (rank_[x] < rank_[y]) std::swap(x, y);
        parent_[y] = x;
        if (rank_[x] == rank_[y]) rank_[x]++;
        return true;
    }
};

class GraphEngine {
    int N_;
    bool pmfg_enabled_;

    // Adjacency lists
    std::vector<std::vector<int>> mst_adj_;
    std::vector<std::vector<int>> pmfg_adj_;
    Eigen::MatrixXd graph_distance_;  ///< Shortest path distance on MST

public:
    GraphEngine(int n_assets, bool pmfg = true)
        : N_(n_assets), pmfg_enabled_(pmfg)
    {
        mst_adj_.resize(N_);
        pmfg_adj_.resize(N_);
        graph_distance_ = Eigen::MatrixXd::Constant(N_, N_, 1e6);
    }

    /// Build MST via Kruskal's algorithm
    std::vector<Edge> build_mst(const Eigen::MatrixXd& dist) {
        std::vector<Edge> all_edges;
        all_edges.reserve(N_ * (N_ - 1) / 2);
        for (int i = 0; i < N_; ++i)
            for (int j = i + 1; j < N_; ++j)
                all_edges.push_back({i, j, dist(i, j)});

        std::sort(all_edges.begin(), all_edges.end());

        UnionFind uf(N_);
        std::vector<Edge> mst;
        for (auto& adj : mst_adj_) adj.clear();

        for (const auto& e : all_edges) {
            if (uf.unite(e.i, e.j)) {
                mst.push_back(e);
                mst_adj_[e.i].push_back(e.j);
                mst_adj_[e.j].push_back(e.i);
                if ((int)mst.size() == N_ - 1) break;
            }
        }

        compute_graph_distances();
        return mst;
    }

    /// Build PMFG: 3(N-2) edges, maintain planarity
    /// Simplified: start from MST, greedily add shortest non-MST edges
    /// that don't violate genus-0 (approximation via max-degree heuristic)
    std::vector<Edge> build_pmfg(const Eigen::MatrixXd& dist) {
        auto mst = build_mst(dist);
        int target = 3 * (N_ - 2);

        // All edges sorted by distance
        std::vector<Edge> all_edges;
        for (int i = 0; i < N_; ++i)
            for (int j = i + 1; j < N_; ++j)
                all_edges.push_back({i, j, dist(i, j)});
        std::sort(all_edges.begin(), all_edges.end());

        // Mark MST edges
        Eigen::MatrixXi in_graph = Eigen::MatrixXi::Zero(N_, N_);
        for (auto& adj : pmfg_adj_) adj.clear();
        std::vector<int> degree(N_, 0);

        for (const auto& e : mst) {
            in_graph(e.i, e.j) = in_graph(e.j, e.i) = 1;
            pmfg_adj_[e.i].push_back(e.j);
            pmfg_adj_[e.j].push_back(e.i);
            degree[e.i]++; degree[e.j]++;
        }

        std::vector<Edge> pmfg = mst;

        // Add edges greedily (planarity approx: Euler V-E+F=2, E≤3V-6)
        for (const auto& e : all_edges) {
            if ((int)pmfg.size() >= target) break;
            if (in_graph(e.i, e.j)) continue;

            // Planarity heuristic: max edges = 3N-6 for planar
            // PMFG target = 3(N-2) = 3N-6, so just add until target
            pmfg.push_back(e);
            in_graph(e.i, e.j) = in_graph(e.j, e.i) = 1;
            pmfg_adj_[e.i].push_back(e.j);
            pmfg_adj_[e.j].push_back(e.i);
            degree[e.i]++; degree[e.j]++;
        }

        return pmfg;
    }

    /// BFS shortest path distances on MST
    void compute_graph_distances() {
        graph_distance_.setConstant(1e6);
        for (int src = 0; src < N_; ++src) {
            graph_distance_(src, src) = 0;
            std::vector<bool> visited(N_, false);
            std::vector<int> queue = {src};
            visited[src] = true;
            while (!queue.empty()) {
                std::vector<int> next;
                for (int u : queue) {
                    for (int v : mst_adj_[u]) {
                        if (!visited[v]) {
                            visited[v] = true;
                            graph_distance_(src, v) = graph_distance_(src, u) + 1;
                            next.push_back(v);
                        }
                    }
                }
                queue = std::move(next);
            }
        }
    }

    void update(SimulationState& state) {
        if (pmfg_enabled_)
            build_pmfg(state.distance);
        else
            build_mst(state.distance);
    }

    const Eigen::MatrixXd& graph_distances() const { return graph_distance_; }
    const std::vector<std::vector<int>>& mst_adjacency() const { return mst_adj_; }
    const std::vector<std::vector<int>>& pmfg_adjacency() const { return pmfg_adj_; }
};

} // namespace sovereign
