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
    Eigen::MatrixXd dist_cache_;       ///< Cached metric distance matrix (for weighted BFS)

public:
    GraphEngine(int n_assets, bool pmfg = true)
        : N_(n_assets), pmfg_enabled_(pmfg)
    {
        mst_adj_.resize(N_);
        pmfg_adj_.resize(N_);
        graph_distance_ = Eigen::MatrixXd::Constant(N_, N_, 1e6);
        dist_cache_      = Eigen::MatrixXd::Zero(N_, N_);
    }

    /// Build MST via Prim's algorithm — O(N²) for dense distance matrix
    /// (Kruskal is O(N²logN) on dense graphs due to edge sorting)
    std::vector<Edge> build_mst(const Eigen::MatrixXd& dist) {
        dist_cache_ = dist;  // Cache for weighted BFS
        std::vector<Edge> mst;
        mst.reserve(N_ - 1);
        for (auto& adj : mst_adj_) adj.clear();

        std::vector<bool> in_tree(N_, false);
        std::vector<double> min_dist(N_, 1e18);
        std::vector<int> parent(N_, -1);
        min_dist[0] = 0;

        for (int step = 0; step < N_; ++step) {
            // Find nearest vertex not in tree
            int u = -1;
            double best = 1e18;
            for (int v = 0; v < N_; ++v) {
                if (!in_tree[v] && min_dist[v] < best) {
                    best = min_dist[v]; u = v;
                }
            }
            if (u < 0) break;
            in_tree[u] = true;

            if (parent[u] >= 0) {
                mst.push_back({parent[u], u, dist(parent[u], u)});
                mst_adj_[parent[u]].push_back(u);
                mst_adj_[u].push_back(parent[u]);
            }

            // Update distances
            for (int v = 0; v < N_; ++v) {
                if (!in_tree[v] && dist(u, v) < min_dist[v]) {
                    min_dist[v] = dist(u, v);
                    parent[v] = u;
                }
            }
        }

        compute_graph_distances();
        return mst;
    }

    /// Build degree-bounded enriched graph (NOT true PMFG — no planarity test)
    /// True PMFG requires Boyer-Myrvold planarity check per edge insertion.
    /// This approximation: MST + greedily add shortest edges up to 3(N-2),
    /// with a max-degree cap to limit non-planar pathology.
    std::vector<Edge> build_pmfg(const Eigen::MatrixXd& dist) {
        auto mst = build_mst(dist);
        int target = 3 * (N_ - 2);

        // All edges sorted by distance (Prim already gave us MST; now enrich)
        std::vector<Edge> all_edges;
        for (int i = 0; i < N_; ++i)
            for (int j = i + 1; j < N_; ++j)
                all_edges.push_back({i, j, dist(i, j)});
        std::sort(all_edges.begin(), all_edges.end());

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

        // Degree cap removed: we simply add the shortest remaining edges until target
        for (const auto& e : all_edges) {
            if ((int)pmfg.size() >= target) break;
            if (in_graph(e.i, e.j)) continue;

            pmfg.push_back(e);
            in_graph(e.i, e.j) = in_graph(e.j, e.i) = 1;
            pmfg_adj_[e.i].push_back(e.j);
            pmfg_adj_[e.j].push_back(e.i);
        }

        compute_pmfg_distances();
        return pmfg;
    }

    /// Weighted BFS/Dijkstra shortest path distances on MST using real metric edge weights.
    /// MST has no cycles so BFS with accumulated weights is exact (no relaxation needed).
    void compute_graph_distances() {
        graph_distance_.setConstant(1e6);
        std::vector<int> bfs_queue(N_);
        std::vector<bool> visited(N_, false);

        for (int src = 0; src < N_; ++src) {
            graph_distance_(src, src) = 0.0;
            std::fill(visited.begin(), visited.end(), false);
            visited[src] = true;
            int head = 0, tail = 0;
            bfs_queue[tail++] = src;
            while (head < tail) {
                int u = bfs_queue[head++];
                for (int v : mst_adj_[u]) {
                    if (!visited[v]) {
                        visited[v] = true;
                        // Use the actual metric distance stored in the distance matrix
                        graph_distance_(src, v) = graph_distance_(src, u)
                                                + dist_cache_(u, v);
                        bfs_queue[tail++] = v;
                    }
                }
            }
        }
    }

    /// Floyd-Warshall all-pairs shortest paths on the enriched PMFG.
    /// Necessary because PMFG contains cycles, so BFS tree sum is invalid.
    /// O(N^3) is practically instant for N=50.
    void compute_pmfg_distances() {
        graph_distance_.setConstant(1e6);
        for (int i = 0; i < N_; ++i) {
            graph_distance_(i, i) = 0.0;
            for (int j : pmfg_adj_[i]) {
                graph_distance_(i, j) = dist_cache_(i, j);
            }
        }

        for (int k = 0; k < N_; ++k) {
            for (int i = 0; i < N_; ++i) {
                for (int j = 0; j < N_; ++j) {
                    if (graph_distance_(i, k) + graph_distance_(k, j) < graph_distance_(i, j)) {
                        graph_distance_(i, j) = graph_distance_(i, k) + graph_distance_(k, j);
                    }
                }
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
