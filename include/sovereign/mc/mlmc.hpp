#pragma once
/// @file mlmc.hpp
/// @brief Multilevel Monte Carlo (Giles, OPRE 2008).
///
/// Ŷ = Ŷ₀ + Σ_{l=1}^L Ŷ_l where Ŷ_l estimates E[P̂_l - P̂_{l-1}]
/// Optimal N_l ∝ √(V_l · h_l), complexity O(ε⁻²(log ε)²)

#include <sovereign/config.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <vector>
#include <numeric>
#include <iostream>

namespace sovereign {

struct MLMCLevel {
    int    level = 0;
    int    n_samples = 0;
    double h = 0;           ///< Timestep at this level
    double mean = 0;        ///< Sample mean of P_l - P_{l-1}
    double variance = 0;    ///< Sample variance
    double cost = 0;        ///< Computational cost
};

class MLMCEngine {
    const MLMCConfig& cfg_;

    /// Payoff functional type: takes (path_fine, path_coarse, level) → correction
    using PayoffFn = std::function<double(const Eigen::VectorXd&,
                                          const Eigen::VectorXd&, int)>;

public:
    explicit MLMCEngine(const MLMCConfig& cfg) : cfg_(cfg) {}

    /// Run MLMC estimation.
    /// simulate_fn(level, dt, seed) → pair of (fine_value, coarse_value)
    /// Returns estimated expectation and RMSE.
    struct MLMCResult {
        double estimate = 0;
        double rmse = 0;
        std::vector<MLMCLevel> levels;
        int total_samples = 0;
    };

    MLMCResult run(
        std::function<std::pair<double,double>(int level, double dt_fine, uint64_t seed)> simulate_fn,
        double T) const
    {
        MLMCResult result;
        int L = cfg_.n_levels;
        int M = cfg_.geometric_factor;

        // Phase 1: Initial sampling to estimate variance per level
        std::vector<MLMCLevel> levels(L);
        for (int l = 0; l < L; ++l) {
            levels[l].level = l;
            levels[l].h = T * std::pow(M, -l);
            int n_init = std::max(100, cfg_.base_samples / (1 << l));
            levels[l].n_samples = n_init;

            double sum = 0, sum2 = 0;
            for (int s = 0; s < n_init; ++s) {
                auto [fine, coarse] = simulate_fn(l, levels[l].h, s);
                double Y = (l == 0) ? fine : (fine - coarse);
                sum += Y;
                sum2 += Y * Y;
            }
            levels[l].mean = sum / n_init;
            // FIX #36: Unbiased variance estimator (/(n-1), not /n).
            // Biased variance underestimates V_l, causing MLMC to underallocate
            // high-variance levels and increase RMSE above target.
            levels[l].variance = (n_init > 1)
                ? (sum2 / n_init - levels[l].mean * levels[l].mean) * n_init / (n_init - 1)
                : 0.0;
            levels[l].cost = n_init * std::pow(M, l);
        }

        // Phase 2: Optimal sample allocation (Giles Theorem 3.1)
        // N_l ∝ √(V_l / C_l) Σ_k √(V_k · C_k)
        // Cost C_l ∝ 1/h_l => N_l ∝ √(V_l · h_l) Σ_k √(V_k / h_k)
        double eps = cfg_.target_rmse;
        double sum_sqrt_V_div_h = 0;
        for (int l = 0; l < L; ++l)
            sum_sqrt_V_div_h += std::sqrt(levels[l].variance / levels[l].h);

        for (int l = 0; l < L; ++l) {
            double N_opt = (2.0 / (eps * eps)) * sum_sqrt_V_div_h
                         * std::sqrt(levels[l].variance * levels[l].h);
            int N_target = std::max(static_cast<int>(std::ceil(N_opt)),
                                     levels[l].n_samples);

            // Additional samples needed
            int n_extra = N_target - levels[l].n_samples;
            if (n_extra > 0) {
                // FIX #35: Use Welford-style online update instead of
                // reconstructing sum from mean*n (catastrophic cancellation).
                double running_mean = levels[l].mean;
                double M2 = levels[l].variance * (levels[l].n_samples - 1);
                int n_total = levels[l].n_samples;

                for (int s = 0; s < n_extra; ++s) {
                    auto [fine, coarse] = simulate_fn(
                        l, levels[l].h, levels[l].n_samples + s);
                    double Y = (l == 0) ? fine : (fine - coarse);
                    n_total++;
                    double delta = Y - running_mean;
                    running_mean += delta / n_total;
                    double delta2 = Y - running_mean;
                    M2 += delta * delta2;
                }
                levels[l].n_samples = N_target;
                levels[l].mean = running_mean;
                levels[l].variance = (n_total > 1) ? M2 / (n_total - 1) : 0.0;
            }
        }

        // Phase 3: Combine estimates
        result.estimate = 0;
        result.rmse = 0;
        result.total_samples = 0;
        for (int l = 0; l < L; ++l) {
            result.estimate += levels[l].mean;
            result.rmse += levels[l].variance / levels[l].n_samples;
            result.total_samples += levels[l].n_samples;
        }
        result.rmse = std::sqrt(result.rmse);
        result.levels = levels;
        return result;
    }

    /// Print MLMC diagnostics
    static void print_diagnostics(const MLMCResult& r) {
        std::cout << "MLMC Estimate: " << r.estimate
                  << " ± " << r.rmse << "\n";
        std::cout << "Total samples: " << r.total_samples << "\n";
        for (const auto& l : r.levels) {
            std::cout << "  Level " << l.level
                      << ": N=" << l.n_samples
                      << " mean=" << l.mean
                      << " var=" << l.variance
                      << " h=" << l.h << "\n";
        }
    }
};

} // namespace sovereign
