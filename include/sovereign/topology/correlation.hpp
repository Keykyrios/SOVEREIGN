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

    /// Trace-preserving MP eigenvalue clipping (Bouchaud & Potters).
    /// Noise eigenvalues (below lambda_+) are replaced with their mean.
    /// Signal eigenvalues are kept. Total trace is preserved.
    void clip_eigenvalues(Eigen::VectorXd& evals, double Q, double sigma2) const {
        auto [lm, lp] = mp_edges(sigma2, Q);
        int n_noise = 0;
        double noise_trace = 0, signal_trace = 0;
        for (int i = 0; i < evals.size(); ++i) {
            if (evals(i) <= lp) {
                noise_trace += evals(i);
                n_noise++;
            } else {
                signal_trace += evals(i);
            }
        }
        // Trace-preserving MP clipping
        double noise_replacement = (n_noise > 0) ? noise_trace / n_noise : sigma2;
        for (int i = 0; i < evals.size(); ++i) {
            if (evals(i) <= lp) {
                evals(i) = noise_replacement;
            }
        }
    }

public:
    CorrelationEngine(const TopologyConfig& cfg, int n_assets)
        : cfg_(cfg), N_(n_assets)
    {
        // Initialize covariance to near-zero identity.
        // DO NOT seed with fake uniform correlation!
        // The EWMA will build up the TRUE covariance from actual returns.
        cov_ewma_ = Eigen::MatrixXd::Identity(N_, N_) * 1e-10;
        vol_ewma_ = Eigen::VectorXd::Constant(N_, 1e-5);
        mean_ewma_ = Eigen::VectorXd::Zero(N_);
    }

    /// Record returns every step
    void record(const SimulationState& state) {
        Eigen::VectorXd r = state.returns();
        samples_++;

        // Adaptive warm-up alpha:
        // For the first 2*N samples, use simple running average (1/t)
        // so the matrix converges from actual data, not from a fake seed.
        // After warm-up, switch to the configured EWMA alpha.
        int warmup = 2 * N_;  // need at least 2N samples for rank-N covariance
        double a = (samples_ <= warmup)
                 ? 1.0 / static_cast<double>(samples_)
                 : cfg_.ewma_alpha;

        mean_ewma_ = (1 - a) * mean_ewma_ + a * r;
        Eigen::VectorXd r_dm = r - mean_ewma_;
        cov_ewma_ = (1 - a) * cov_ewma_ + a * r_dm * r_dm.transpose();
    }

    /// Run RMT cleaning and eigen-decomposition (periodic)
    void update(SimulationState& state) {

        // Extract volatilities — hard floor to prevent NaN cascade
        constexpr double VOL_FLOOR = 1e-8;
        for (int i = 0; i < N_; ++i)
            vol_ewma_(i) = std::sqrt(std::max(cov_ewma_(i, i), VOL_FLOOR * VOL_FLOOR));

        // Raw correlation — bypass if vol is dead
        Eigen::MatrixXd& corr = state.raw_correlation;
        for (int i = 0; i < N_; ++i) {
            for (int j = 0; j < N_; ++j) {
                if (vol_ewma_(i) < VOL_FLOOR || vol_ewma_(j) < VOL_FLOOR) {
                    corr(i, j) = (i == j) ? 1.0 : 0.0;  // Identity for dead assets
                } else {
                    corr(i, j) = cov_ewma_(i, j) / (vol_ewma_(i) * vol_ewma_(j));
                    corr(i, j) = std::clamp(corr(i, j), -1.0, 1.0);
                }
            }
        }

        // RMT cleaning
        state.correlation = clean_rmt(corr, state.eigenvalues, state.eigenvectors);

        // Distance matrix: d_ij = arccos(ρ_ij) (Riemannian geodesic on hypersphere)
        // This preserves the metric triangle inequality correctly when summing distances.
        for (int i = 0; i < N_; ++i)
            for (int j = 0; j < N_; ++j) {
                double rho = std::clamp(state.correlation(i, j), -1.0, 1.0);
                state.distance(i, j) = std::acos(rho);
            }
    }

    /// RMT cleaning: eigenvalue shrinkage below MP edge
    Eigen::MatrixXd clean_rmt(const Eigen::MatrixXd& raw,
                               Eigen::VectorXd& eigenvalues,
                               Eigen::MatrixXd& eigenvectors) const
    {
        // WARMUP BYPASS: Let the EWMA accumulate enough cross-asset signal
        // before unleashing the MP cleaner. Without this, the cleaner
        // activates on pure noise and cleans everything back to Identity.
        if (samples_ < 2 * N_) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(raw);
            eigenvalues = solver.eigenvalues();
            eigenvectors = solver.eigenvectors();
            return raw;  // Return raw correlation — no cleaning yet
        }

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(raw);
        eigenvalues = solver.eigenvalues();
        eigenvectors = solver.eigenvectors();

        // Use EFFECTIVE sample count for EWMA, not raw samples_.
        // EWMA has finite memory: N_eff ≈ 2/α. Using raw samples_ (which → ∞)
        // collapses the MP noise edge to 1.0 and nukes valid eigen-signals.
        double N_eff = std::min(2.0 / cfg_.ewma_alpha, static_cast<double>(samples_));
        double Q = std::max(N_eff / N_, 1.001);
        // Estimate noise variance from the bottom half of the spectrum
        // SelfAdjointEigenSolver returns eigenvalues in ASCENDING order,
        // so indices [0, half) are the smallest (noise) eigenvalues.
        double sigma2 = 0.0;
        int half = N_ / 2;
        for (int i = 0; i < half; ++i) {
            sigma2 += eigenvalues(i);
        }
        sigma2 /= std::max(half, 1);

        // Trace-preserving MP clipping
        Eigen::VectorXd cleaned_evals = eigenvalues;
        clip_eigenvalues(cleaned_evals, Q, sigma2);
        for (int i = 0; i < N_; ++i)
            cleaned_evals(i) = std::max(cleaned_evals(i), 1e-6);

        // Do NOT overwrite state eigenvalues with cleaned_evals.
        // We want the dashboard to show the TRUE distribution (which is heavy-tailed),
        // not the flat-lined MP cleaned spectrum!

        // Reconstruct: C_clean = V · diag(λ_clean) · V^T
        Eigen::MatrixXd cleaned = eigenvectors * cleaned_evals.asDiagonal()
                                * eigenvectors.transpose();

        // Higham's Alternating Projections for nearest correlation matrix
        // Preserves the clipped spectrum much better than scalar diagonal division
        Eigen::MatrixXd Y = cleaned;
        Eigen::MatrixXd S = Eigen::MatrixXd::Zero(N_, N_);
        for (int iter = 0; iter < cfg_.spd_max_iter; ++iter) {
            Eigen::MatrixXd R = Y - S;
            
            // P_S: Project onto positive semi-definite cone
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(R);
            Eigen::VectorXd evals = es.eigenvalues();
            Eigen::MatrixXd evecs = es.eigenvectors();
            for (int i = 0; i < N_; ++i) evals(i) = std::max(evals(i), 1e-8);
            Eigen::MatrixXd X = evecs * evals.asDiagonal() * evecs.transpose();
            
            S = X - R;
            Y = X;
            
            // P_U: Project onto unit diagonal (correlation constraint)
            for (int i = 0; i < N_; ++i) Y(i, i) = 1.0;
        }
        cleaned = Y;

        return cleaned;
    }
};

} // namespace sovereign
