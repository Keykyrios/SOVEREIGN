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

    /// Exact Z2 Persistence Homology up to dimension 2 via standard matrix reduction.
    /// Replaces the fake 4-cycle approximation.
    PersistenceDiagram compute_Z2_homology(const Eigen::MatrixXd& dist) const {
        PersistenceDiagram pd;

        struct Simplex {
            int dim;
            double filtration;
            std::vector<int> vertices;
            int idx;
            bool operator<(const Simplex& o) const {
                if (std::abs(filtration - o.filtration) > 1e-9) return filtration < o.filtration;
                return dim < o.dim;
            }
        };

        std::vector<Simplex> simplices;
        // Vertices
        for(int i = 0; i < N_; ++i) simplices.push_back({0, 0.0, {i}, 0});
        // Edges
        for(int i = 0; i < N_; ++i)
            for(int j = i + 1; j < N_; ++j)
                if (dist(i, j) <= cfg_.max_filtration)
                    simplices.push_back({1, dist(i, j), {i, j}, 0});
        // Triangles
        for(int i = 0; i < N_; ++i)
            for(int j = i + 1; j < N_; ++j)
                for(int k = j + 1; k < N_; ++k) {
                    double f = std::max({dist(i,j), dist(i,k), dist(j,k)});
                    if (f <= cfg_.max_filtration)
                        simplices.push_back({2, f, {i, j, k}, 0});
                }
        // Tetrahedra (needed to kill 2-cycles / voids)
        if (cfg_.max_dimension >= 2) {
            for(int i = 0; i < N_; ++i)
                for(int j = i + 1; j < N_; ++j)
                    for(int k = j + 1; k < N_; ++k)
                        for(int l = k + 1; l < N_; ++l) {
                            double f = std::max({dist(i,j), dist(i,k), dist(i,l), 
                                                 dist(j,k), dist(j,l), dist(k,l)});
                            if (f <= cfg_.max_filtration)
                                simplices.push_back({3, f, {i, j, k, l}, 0});
                        }
        }

        std::sort(simplices.begin(), simplices.end());
        int M = simplices.size();
        for(int i = 0; i < M; ++i) simplices[i].idx = i;

        auto hash_verts = [](const std::vector<int>& v) {
            long long h = 0;
            for(int x : v) h = (h * 100) + (x + 1);
            return h;
        };
        std::vector<long long> hashes(M);
        for(int i = 0; i < M; ++i) hashes[i] = hash_verts(simplices[i].vertices);
        
        auto find_face = [&](const std::vector<int>& v) -> int {
            long long h = hash_verts(v);
            for(int i = M - 1; i >= 0; --i) if(hashes[i] == h) return i;
            return -1;
        };

        std::vector<int> low(M, -1);
        std::vector<int> pivot_to_col(M, -1);
        std::vector<std::vector<int>> B(M);
        
        for(int j = 0; j < M; ++j) {
            const auto& s = simplices[j];
            if (s.dim > 0) {
                for(size_t k = 0; k < s.vertices.size(); ++k) {
                    std::vector<int> face = s.vertices;
                    face.erase(face.begin() + k);
                    int face_idx = find_face(face);
                    if (face_idx >= 0) B[j].push_back(face_idx);
                }
                std::sort(B[j].begin(), B[j].end(), std::greater<int>());
            }
        }

        auto add_cols = [](const std::vector<int>& a, const std::vector<int>& b) {
            std::vector<int> res;
            auto it_a = a.begin(), it_b = b.begin();
            while(it_a != a.end() && it_b != b.end()) {
                if(*it_a > *it_b) res.push_back(*it_a++);
                else if(*it_b > *it_a) res.push_back(*it_b++);
                else { ++it_a; ++it_b; }
            }
            while(it_a != a.end()) res.push_back(*it_a++);
            while(it_b != b.end()) res.push_back(*it_b++);
            return res;
        };

        for(int j = 0; j < M; ++j) {
            std::vector<int> col = B[j];
            while(!col.empty()) {
                int l = col.front();
                if (pivot_to_col[l] != -1) {
                    col = add_cols(col, B[pivot_to_col[l]]);
                } else {
                    pivot_to_col[l] = j;
                    low[j] = l;
                    break;
                }
            }
            B[j] = col;
            if (!col.empty()) {
                int l = col.front();
                double birth = simplices[l].filtration;
                double death = simplices[j].filtration;
                if (death > birth && death < cfg_.max_filtration && simplices[l].dim <= cfg_.max_dimension) {
                    pd.push_back({simplices[l].dim, birth, death});
                }
            }
        }

        for(int j = 0; j < M; ++j) {
            if (low[j] == -1 && pivot_to_col[j] == -1 && simplices[j].dim <= cfg_.max_dimension) {
                pd.push_back({simplices[j].dim, simplices[j].filtration, cfg_.max_filtration});
            }
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

#ifdef SOVEREIGN_HAS_GUDHI
        // Exact persistence via GUDHI (default path)
        compute_gudhi(dist);
#else
        // Fallback: exact Z2 reduction (no GUDHI)
        auto homology = compute_Z2_homology(dist);
        current_pd_.insert(current_pd_.end(), homology.begin(), homology.end());
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
