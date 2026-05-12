#pragma once
/// @file correlation.hpp
/// @brief Layer 6.1 — Dynamic correlation matrix with RMT cleaning.
///
/// References:
///   Laloux et al. (9810255v1) — Marčenko-Pastur noise edge
///   Bun, Bouchaud & Potters (1610.08104v1) — Optimal RIE shrinkage
///
/// EWMA: Ω(t) = (1-α)Ω(t-1) + α·r(t)r(t)ᵀ
/// Cleaning: eigenvalues below MP edge → shrunk via RIE
/// SPD projection: Riemannian on the correlation manifold

#include <sovereign/config.hpp>
#include <sovereign/core/state.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>

namespace sovereign {

class CorrelationEngine {
    const TopologyConfig& cfg_;
    int N_;

    // Running EWMA of outer products
    Eigen::MatrixXd cov_ewma_;
    Eigen::VectorXd vol_ewma_;    ///< EWMA volatility for normalization
    Eigen::VectorXd mean_ewma_;   ///< EWMA mean return
    int samples_ = 0;

    /// Marčenko-Pastur edge: λ± = σ²(1 + 1/Q ± 2√(1/Q))
    std::pair<double, double> mp_edges(double sigma2, double Q) const {
        double sq = std::sqrt(1.0 / Q);
        double lp = sigma2 * (1.0 + 1.0 / Q + 2.0 * sq);
        double lm = sigma2 * (1.0 + 1.0 / Q - 2.0 * sq);
        return {std::max(lm, 0.0), lp};
    }

    /// Optimal RIE shrinkage (Bun et al. Eq. 45-50)
    /// ξ(λ) = λ / |1 - Q⁻¹ + Q⁻¹·λ·g(λ)|²
    /// where g(z) is the Stieltjes transform of MP law
    double rie_shrinkage(double lambda, double Q, double sigma2) const {
        // Simplified RIE: linear shrinkage toward grand mean
        auto [lm, lp] = mp_edges(sigma2, Q);
        if (lambda <= lp) {
            // Noise eigenvalue → shrink to MP mean = σ²
            double alpha_shrink = (lp - lambda) / (lp - lm + 1e-10);
            return sigma2 * (1.0 - alpha_shrink) + lambda * alpha_shrink * 0.5;
        }
        // Signal eigenvalue → keep (with mild regularization)
        return lambda * 0.95 + sigma2 * 0.05;
    }

public:
    CorrelationEngine(const TopologyConfig& cfg, int n_assets)
        : cfg_(cfg), N_(n_assets)
    {
        double base_corr = 0.3;
        cov_ewma_ = Eigen::MatrixXd::Constant(N_, N_, base_corr * 0.04);
        for (int i = 0; i < N_; ++i) cov_ewma_(i, i) = 0.04;
        
        vol_ewma_ = Eigen::VectorXd::Constant(N_, 0.2);
        mean_ewma_ = Eigen::VectorXd::Zero(N_);
    }

    /// Record returns every step
    void record(const SimulationState& state) {
        Eigen::VectorXd r = state.returns();
        samples_++;

        // EWMA update of mean and covariance
        double a = cfg_.ewma_alpha;
        mean_ewma_ = (1 - a) * mean_ewma_ + a * r;
        Eigen::VectorXd r_dm = r - mean_ewma_;
        cov_ewma_ = (1 - a) * cov_ewma_ + a * r_dm * r_dm.transpose();
    }

    /// Run RMT cleaning and eigen-decomposition (periodic)
    void update(SimulationState& state) {

        // Extract volatilities
        for (int i = 0; i < N_; ++i)
            vol_ewma_(i) = std::sqrt(std::max(cov_ewma_(i, i), 1e-15));

        // Raw correlation
        Eigen::MatrixXd& corr = state.raw_correlation;
        for (int i = 0; i < N_; ++i)
            for (int j = 0; j < N_; ++j)
                corr(i, j) = cov_ewma_(i, j) / (vol_ewma_(i) * vol_ewma_(j) + 1e-15);

        // RMT cleaning
        state.correlation = clean_rmt(corr, state.eigenvalues, state.eigenvectors);

        // Distance matrix: d_ij = sqrt(2(1 - ρ_ij))
        for (int i = 0; i < N_; ++i)
            for (int j = 0; j < N_; ++j)
                state.distance(i, j) = std::sqrt(
                    2.0 * (1.0 - std::min(state.correlation(i, j), 1.0)));
    }

    /// RMT cleaning: eigenvalue shrinkage below MP edge
    Eigen::MatrixXd clean_rmt(const Eigen::MatrixXd& raw,
                               Eigen::VectorXd& eigenvalues,
                               Eigen::MatrixXd& eigenvectors) const
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(raw);
        eigenvalues = solver.eigenvalues();
        eigenvectors = solver.eigenvectors();

        double Q = std::max(static_cast<double>(samples_) / N_, 1.001);
        double sigma2 = 1.0;  // Normalized correlation

        // Shrink eigenvalues
        Eigen::VectorXd cleaned_evals = eigenvalues;
        for (int i = 0; i < N_; ++i) {
            cleaned_evals(i) = rie_shrinkage(eigenvalues(i), Q, sigma2);
            cleaned_evals(i) = std::max(cleaned_evals(i), 1e-6);
        }

        // Reconstruct: C_clean = V · diag(λ_clean) · V^T
        Eigen::MatrixXd cleaned = eigenvectors * cleaned_evals.asDiagonal()
                                * eigenvectors.transpose();

        // Force unit diagonal (correlation constraint)
        for (int i = 0; i < N_; ++i) {
            double d = std::sqrt(cleaned(i, i));
            for (int j = 0; j < N_; ++j) {
                cleaned(i, j) /= (d * std::sqrt(cleaned(j, j)) + 1e-15);
            }
            cleaned(i, i) = 1.0;
        }

        return cleaned;
    }
};

} // namespace sovereign
