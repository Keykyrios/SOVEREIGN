#pragma once
/// @file stability.hpp
/// @brief Dykstra's alternating projection for Hawkes spectral radius < 1.
///
/// Reference: Lima & Choi (1805.09570v3) — renormalization factors
///
/// The branching matrix Φ_{ij} = ∫₀^∞ φ_{ij}(t)dt must satisfy
/// ρ(Φ) < 1 for stationarity. We project onto the feasible set
/// {Φ : ρ(Φ) ≤ ρ_max} via alternating projections on the α parameters.

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <algorithm>

namespace sovereign {

class HawkesStabilityProjector {
    double rho_max_;
    int    max_iter_;

public:
    HawkesStabilityProjector(double rho_max = 0.95, int max_iter = 100)
        : rho_max_(rho_max), max_iter_(max_iter) {}

    /// Compute spectral radius of branching matrix
    static double spectral_radius(const Eigen::MatrixXd& Phi) {
        Eigen::EigenSolver<Eigen::MatrixXd> solver(Phi, false);
        return solver.eigenvalues().cwiseAbs().maxCoeff();
    }

    /// Project branching matrix so ρ(Φ) ≤ ρ_max.
    /// Modifies alpha_matrix in-place (the excitation strengths).
    /// beta_matrix and epsilon are needed to reconstruct Φ from α.
    ///
    /// Returns: number of iterations used, final spectral radius
    std::pair<int, double> project(
        Eigen::MatrixXd& alpha_matrix,
        const Eigen::MatrixXd& beta_matrix,
        const Eigen::MatrixXd& epsilon_matrix) const
    {
        int N = alpha_matrix.rows();
        Eigen::MatrixXd Phi(N, N);

        for (int iter = 0; iter < max_iter_; ++iter) {
            // Build branching matrix: Φ_{ij} = α_{ij} · β_{ij} / ε_{ij}
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    Phi(i, j) = alpha_matrix(i, j) * beta_matrix(i, j)
                              / (epsilon_matrix(i, j) + 1e-10);

            double rho = spectral_radius(Phi);
            if (rho <= rho_max_) return {iter, rho};

            // Rescale: α ← α · (ρ_max / ρ)
            double scale = rho_max_ / rho;
            alpha_matrix *= scale;
        }

        // Final check
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                Phi(i, j) = alpha_matrix(i, j) * beta_matrix(i, j)
                          / (epsilon_matrix(i, j) + 1e-10);
        return {max_iter_, spectral_radius(Phi)};
    }

    /// Dykstra's alternating projection between:
    ///   C₁ = {Φ : ρ(Φ) ≤ ρ_max}  (spectral norm ball)
    ///   C₂ = {Φ : Φ_{ij} ≥ 0}     (non-negativity)
    /// with increment corrections for proper convergence.
    std::pair<int, double> dykstra_project(Eigen::MatrixXd& alpha_matrix,
        const Eigen::MatrixXd& beta_matrix,
        const Eigen::MatrixXd& epsilon_matrix) const
    {
        int N = alpha_matrix.rows();
        Eigen::MatrixXd p_inc = Eigen::MatrixXd::Zero(N, N);
        Eigen::MatrixXd q_inc = Eigen::MatrixXd::Zero(N, N);

        for (int iter = 0; iter < max_iter_; ++iter) {
            Eigen::MatrixXd old_y = alpha_matrix + p_inc;
            Eigen::MatrixXd y = old_y;

            // Project onto C₁ (spectral radius constraint)
            Eigen::MatrixXd Phi(N, N);
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    Phi(i, j) = y(i, j) * beta_matrix(i, j)
                              / (epsilon_matrix(i, j) + 1e-10);

            double rho = spectral_radius(Phi);
            if (rho > rho_max_) {
                // Proper projection: eigendecompose, clip eigenvalues, reconstruct
                Eigen::EigenSolver<Eigen::MatrixXd> solver(Phi);
                Eigen::MatrixXcd V = solver.eigenvectors();
                Eigen::VectorXcd evals = solver.eigenvalues();

                // Clip eigenvalue magnitudes to rho_max
                for (int k = 0; k < N; ++k) {
                    double mag = std::abs(evals(k));
                    if (mag > rho_max_) {
                        evals(k) *= rho_max_ / mag;
                    }
                }

                // Reconstruct: Phi_new = V · diag(evals_clipped) · V^{-1}
                Eigen::MatrixXcd Phi_new = V * evals.asDiagonal() * V.inverse();
                // Extract real part and map back to alpha
                for (int i = 0; i < N; ++i)
                    for (int j = 0; j < N; ++j) {
                        double phi_new = std::max(Phi_new(i, j).real(), 0.0);
                        y(i, j) = phi_new * (epsilon_matrix(i, j) + 1e-10)
                                / (beta_matrix(i, j) + 1e-10);
                    }
            }
            p_inc = old_y - y;
            alpha_matrix = y;

            // Project onto C₂ (non-negativity)
            Eigen::MatrixXd old_z = alpha_matrix + q_inc;
            Eigen::MatrixXd z = old_z.cwiseMax(0.0);
            q_inc = old_z - z;
            alpha_matrix = z;

            // Check convergence
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    Phi(i, j) = alpha_matrix(i, j) * beta_matrix(i, j)
                              / (epsilon_matrix(i, j) + 1e-10);
            rho = spectral_radius(Phi);
            if (rho <= rho_max_ && alpha_matrix.minCoeff() >= 0)
                return {iter, rho};
        }

        Eigen::MatrixXd Phi(N, N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                Phi(i, j) = alpha_matrix(i, j) * beta_matrix(i, j)
                          / (epsilon_matrix(i, j) + 1e-10);
        return {max_iter_, spectral_radius(Phi)};
    }
};

} // namespace sovereign
