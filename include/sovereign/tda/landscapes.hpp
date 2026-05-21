#pragma once
/// @file landscapes.hpp
/// @brief Layer 7.2 — Persistence landscapes and topological risk index.
///
/// Reference: Gidea & Katz (1703.04385v2) Eq. 2.1–2.3
///
/// λ_k(x) = k-th largest tent function over (birth, death) pairs
/// TRI(t) = Σ_k (d_k - b_k)² · w(b_k, d_k)
/// W₂ = Wasserstein-2 distance between consecutive diagrams

#include <sovereign/config.hpp>
#include <sovereign/tda/persistence.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>

namespace sovereign {

class LandscapeEngine {
    const TDAConfig& cfg_;
    int res_;  ///< Landscape resolution

    /// Compute the tent function for a single (b,d) pair
    /// Λ(x) = max(0, min(x-b, d-x))
    static double tent(double x, double b, double d) {
        return std::max(0.0, std::min(x - b, d - x));
    }

public:
    LandscapeEngine(const TDAConfig& cfg) : cfg_(cfg), res_(cfg.landscape_resolution) {}

    /// Compute k-th persistence landscape from diagram (dimension dim)
    Eigen::VectorXd landscape(const PersistenceDiagram& pd, int dim, int k) const {
        // Filter pairs by dimension
        std::vector<std::pair<double, double>> pairs;
        for (const auto& p : pd)
            if (p.dimension == dim)
                pairs.emplace_back(p.birth, p.death);

        Eigen::VectorXd L = Eigen::VectorXd::Zero(res_);
        double x_min = 0, x_max = cfg_.max_filtration;
        double dx = (x_max - x_min) / (res_ - 1);

        for (int i = 0; i < res_; ++i) {
            double x = x_min + i * dx;
            // Compute all tent values at x
            std::vector<double> tents;
            tents.reserve(pairs.size());
            for (const auto& [b, d] : pairs)
                tents.push_back(tent(x, b, d));

            // k-th largest
            std::sort(tents.rbegin(), tents.rend());
            if (k < (int)tents.size())
                L(i) = tents[k];
        }
        return L;
    }

    /// L^p norm of landscape (trapezoidal rule)
    double lp_norm(const Eigen::VectorXd& L, double p) const {
        double dx = cfg_.max_filtration / (res_ - 1);
        double sum = 0;
        // FIX #7: Trapezoidal rule — half-weight at endpoints.
        for (int i = 0; i < L.size(); ++i) {
            double w = (i == 0 || i == L.size() - 1) ? 0.5 : 1.0;
            sum += w * std::pow(std::abs(L(i)), p) * dx;
        }
        return std::pow(sum, 1.0 / p);
    }

    /// Topological Risk Index: TRI(t) = Σ (d-b)^p · w(b,d)
    double compute_tri(const PersistenceDiagram& pd) const {
        double tri = 0;
        int count = 0;
        double p = cfg_.tri_weight_power;
        for (const auto& pair : pd) {
            // Include ALL topological features (H₀ + H₁ + H₂)
            // H₀ merging persistence captures connectivity fragility
            // H₁ loops (when available) capture systemic risk cycles
            double pers = pair.persistence();
            if (pers < 1e-12) continue;  // Skip zero-persistence noise
            double weight = 1.0 / (1.0 + pair.birth);
            tri += std::pow(pers, p) * weight;
        }
        return tri;
    }

    /// Wasserstein-2 distance between two persistence diagrams
    /// Simplified: match pairs greedily by persistence, unmatched → diagonal
    double wasserstein_2(const PersistenceDiagram& pd1,
                          const PersistenceDiagram& pd2) const
    {
        // Extract H₁ pairs from both
        auto extract = [](const PersistenceDiagram& pd, int dim) {
            std::vector<std::pair<double, double>> pairs;
            for (const auto& p : pd)
                if (p.dimension == dim)
                    pairs.emplace_back(p.birth, p.death);
            // Sort by persistence descending
            std::sort(pairs.begin(), pairs.end(),
                [](const auto& a, const auto& b) {
                    return (a.second - a.first) > (b.second - b.first);
                });
            return pairs;
        };

        auto p1 = extract(pd1, 1);
        auto p2 = extract(pd2, 1);

        // Greedy matching with exact diagonal projection for unmatched points
        double total = 0;
        std::vector<bool> used2(p2.size(), false);
        for (size_t i = 0; i < p1.size(); ++i) {
            // Maximum cost is destroying the feature (projecting to diagonal)
            double best = std::pow(p1[i].second - p1[i].first, 2) / 2.0;
            int best_j = -1;
            for (size_t j = 0; j < p2.size(); ++j) {
                if (used2[j]) continue;
                double db = p1[i].first - p2[j].first;
                double dd = p1[i].second - p2[j].second;
                double d2 = db * db + dd * dd;
                if (d2 < best) { best = d2; best_j = static_cast<int>(j); }
            }
            if (best_j != -1) used2[best_j] = true;
            total += best;
        }
        // Any remaining unmatched features in p2 must be destroyed
        for (size_t j = 0; j < p2.size(); ++j) {
            if (!used2[j]) {
                total += std::pow(p2[j].second - p2[j].first, 2) / 2.0;
            }
        }
        return std::sqrt(total);
    }

    /// Update all TDA outputs in state
    void update(SimulationState& state, const PersistenceEngine& persistence) {
        const auto& pd = persistence.diagram();
        const auto& prev = persistence.previous_diagram();

        // TRI
        state.tda_risk_index = compute_tri(pd);

        // Wasserstein distance
        if (!prev.empty())
            state.wasserstein_dist = wasserstein_2(pd, prev);

        // Landscape norms — use H₁ if available, fall back to H₀
        auto L1 = landscape(pd, 1, 0);  // First H₁ landscape
        if (L1.squaredNorm() < 1e-20) {
            // No H₁ features — use H₀ landscape instead
            L1 = landscape(pd, 0, 0);
        }
        state.landscape_l1 = lp_norm(L1, 1.0);
        state.landscape_l2 = lp_norm(L1, 2.0);
    }
};

} // namespace sovereign
