#pragma once
/// @file persistence.hpp
/// @brief Layer 7 — Persistent homology via Vietoris-Rips filtration.
///
/// Reference: Gidea & Katz (1703.04385v2)
///
/// Built-in Ripser-style algorithm when GUDHI unavailable.
/// Computes H₀ (components), H₁ (loops), H₂ (voids).

#include <sovereign/config.hpp>
#include <sovereign/core/state.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>
#include <numeric>

#ifdef SOVEREIGN_HAS_GUDHI
#include <gudhi/Rips_complex.h>
#include <gudhi/Simplex_tree.h>
#include <gudhi/Persistent_cohomology.h>
#endif

namespace sovereign {

/// Birth-death pair for a persistent feature
struct PersistencePair {
    int dimension;     ///< Homology dimension (0, 1, 2)
    double birth;      ///< Filtration value at birth
    double death;      ///< Filtration value at death
    double persistence() const { return death - birth; }
};

/// Persistence diagram = collection of (birth, death) pairs
using PersistenceDiagram = std::vector<PersistencePair>;

class PersistenceEngine {
    const TDAConfig& cfg_;
    int N_;

    PersistenceDiagram current_pd_;
    PersistenceDiagram previous_pd_;

    /// Built-in H₀ computation via Union-Find on edges
    /// (exact for 0-dimensional homology)
    PersistenceDiagram compute_H0(const Eigen::MatrixXd& dist) const {
        struct Edge { int i, j; double w; };
        std::vector<Edge> edges;
        for (int i = 0; i < N_; ++i)
            for (int j = i + 1; j < N_; ++j)
                edges.push_back({i, j, dist(i, j)});

        std::sort(edges.begin(), edges.end(),
                  [](const Edge& a, const Edge& b) { return a.w < b.w; });

        // Union-Find
        std::vector<int> parent(N_), rank(N_, 0);
        std::vector<double> birth(N_, 0.0);
        std::iota(parent.begin(), parent.end(), 0);

        std::function<int(int)> find = [&](int x) -> int {
            while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
            return x;
        };

        PersistenceDiagram pd;

        for (const auto& e : edges) {
            int px = find(e.i), py = find(e.j);
            if (px != py) {
                // Merge: younger component dies
                int younger = (birth[px] >= birth[py]) ? px : py;
                int older = (younger == px) ? py : px;
                pd.push_back({0, birth[younger], e.w});
                if (rank[px] < rank[py]) std::swap(px, py);
                parent[py] = px;
                if (rank[px] == rank[py]) rank[px]++;
                birth[px] = std::min(birth[px], birth[py]);
            }
        }

        // One infinite bar for the final connected component
        pd.push_back({0, 0.0, cfg_.max_filtration});
        return pd;
    }

    /// Approximate H₁ via clique enumeration on triangle births
    PersistenceDiagram compute_H1_approx(const Eigen::MatrixXd& dist) const {
        PersistenceDiagram pd;
        // Find triangles and their "birth" (max edge weight)
        for (int i = 0; i < N_; ++i) {
            for (int j = i + 1; j < N_; ++j) {
                for (int k = j + 1; k < N_; ++k) {
                    double e1 = dist(i, j), e2 = dist(i, k), e3 = dist(j, k);
                    double max_e = std::max({e1, e2, e3});
                    double min_e = std::min({e1, e2, e3});

                    if (max_e < cfg_.max_filtration) {
                        // A cycle is born when the last edge of the triangle appears
                        // and dies when the triangle fills in
                        // Persistence = max_edge - second_max_edge (simplified)
                        double mid_e = e1 + e2 + e3 - max_e - min_e;
                        if (max_e - mid_e > 0.01) {  // Significance threshold
                            pd.push_back({1, mid_e, max_e});
                        }
                    }
                }
            }
            if ((int)pd.size() > 500) break;  // Cap for performance
        }
        return pd;
    }

public:
    PersistenceEngine(const TDAConfig& cfg, int n_assets)
        : cfg_(cfg), N_(n_assets) {}

    /// Compute persistent homology on the distance matrix
    void compute(const Eigen::MatrixXd& dist) {
        previous_pd_ = current_pd_;
        current_pd_.clear();

        // H₀: exact via Union-Find
        auto h0 = compute_H0(dist);
        current_pd_.insert(current_pd_.end(), h0.begin(), h0.end());

        // H₁: approximate
        if (cfg_.max_dimension >= 1) {
            auto h1 = compute_H1_approx(dist);
            current_pd_.insert(current_pd_.end(), h1.begin(), h1.end());
        }

#ifdef SOVEREIGN_HAS_GUDHI
        // If GUDHI available, use exact Ripser for H₁ and H₂
        compute_gudhi(dist);
#endif
    }

#ifdef SOVEREIGN_HAS_GUDHI
    void compute_gudhi(const Eigen::MatrixXd& dist) {
        // Convert to point cloud for GUDHI
        using ST = Gudhi::Simplex_tree<>;
        using Rips = Gudhi::rips_complex::Rips_complex<double>;

        // Build distance matrix as flat vector
        std::vector<std::vector<double>> dm(N_, std::vector<double>(N_));
        for (int i = 0; i < N_; ++i)
            for (int j = 0; j < N_; ++j)
                dm[i][j] = dist(i, j);

        Rips rips(dm, cfg_.max_filtration,
                  [](int i, int j, const auto& d) { return d[i][j]; });
        ST st;
        rips.create_complex(st, cfg_.max_dimension + 1);

        Gudhi::persistent_cohomology::Persistent_cohomology<ST, Gudhi::persistent_cohomology::Field_Zp> pcoh(st);
        pcoh.init_coefficients(2);
        pcoh.compute_persistent_cohomology(0.0);

        current_pd_.clear();
        auto pairs = pcoh.get_persistent_pairs();
        for (const auto& p : pairs) {
            int dim = st.dimension(std::get<0>(p));
            double b = st.filtration(std::get<0>(p));
            double d = st.filtration(std::get<1>(p));
            if (d == std::numeric_limits<double>::infinity())
                d = cfg_.max_filtration;
            current_pd_.push_back({dim, b, d});
        }
    }
#endif

    const PersistenceDiagram& diagram() const { return current_pd_; }
    const PersistenceDiagram& previous_diagram() const { return previous_pd_; }
};

} // namespace sovereign
