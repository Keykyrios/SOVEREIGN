#pragma once
/// @file random.hpp
/// @brief Thread-safe RNG management, fBm generation, Sobol QMC.

#include <Eigen/Dense>
#include <random>
#include <cstdint>
#include <cmath>
#include <vector>
#include <numeric>

namespace sovereign {

/// Xoshiro256** — fast, high-quality PRNG (period 2^256-1)
/// alignas(64) prevents false sharing when packed in std::vector for OpenMP
class alignas(64) Xoshiro256 {
    uint64_t s_[4];
    double spare_ = 0;
    bool has_spare_ = false;

    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    static uint64_t splitmix64(uint64_t& state) {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

public:
    explicit Xoshiro256(uint64_t seed = 42) { this->seed(seed); }

    void seed(uint64_t seed) {
        uint64_t s = seed;
        for (auto& si : s_) si = splitmix64(s);
    }

    uint64_t next() {
        uint64_t result = rotl(s_[1] * 5, 7) * 9;
        uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0]; s_[3] ^= s_[1]; s_[1] ^= s_[2]; s_[0] ^= s_[3];
        s_[2] ^= t; s_[3] = rotl(s_[3], 45);
        return result;
    }

    /// Uniform [0,1)
    double uniform() {
        return (next() >> 11) * 0x1.0p-53;
    }

    /// Standard normal via Box-Muller with cached spare (correct + fast)
    double normal() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        // Use 1-uniform() to get (0,1] — log is always finite
        double u1 = 1.0 - uniform(), u2 = uniform();
        double r = std::sqrt(-2.0 * std::log(u1));
        double theta = 2.0 * M_PI * u2;
        spare_ = r * std::sin(theta);
        has_spare_ = true;
        return r * std::cos(theta);
    }

    /// Fill vector with iid N(0,1)
    void fill_normal(Eigen::VectorXd& v) {
        for (int i = 0; i < v.size(); ++i) v(i) = normal();
    }

    /// Fill matrix with iid N(0,1)
    void fill_normal(Eigen::MatrixXd& m) {
        for (int j = 0; j < m.cols(); ++j)
            for (int i = 0; i < m.rows(); ++i)
                m(i, j) = normal();
    }

    /// Exponential(rate)
    double exponential(double rate) {
        return -std::log(1.0 - uniform()) / rate;  // (0,1] domain for log
    }

    /// Jump 2^128 steps — produces non-overlapping subsequences for
    /// parallel threads. Standard polynomial from Vigna's reference impl.
    void jump() {
        static const uint64_t JUMP[] = {
            0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL,
            0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL
        };
        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int i = 0; i < 4; ++i) {
            for (int b = 0; b < 64; ++b) {
                if (JUMP[i] & (1ULL << b)) {
                    s0 ^= s_[0]; s1 ^= s_[1];
                    s2 ^= s_[2]; s3 ^= s_[3];
                }
                next();
            }
        }
        s_[0] = s0; s_[1] = s1; s_[2] = s2; s_[3] = s3;
    }

    /// Create a thread-local RNG with guaranteed non-overlapping sequence
    Xoshiro256 fork() {
        Xoshiro256 child = *this;
        child.has_spare_ = false;  // Kill spare contamination
        child.spare_ = 0;
        has_spare_ = false;  // Reset parent spare too — stale after jump
        spare_ = 0;
        jump();  // Advance parent past child's 2^128 block
        return child;
    }

    const uint64_t* state() const { return s_; }
};

/// Cholesky-correlated normal vector: z = L * w where w ~ N(0,I)
inline Eigen::VectorXd correlated_normals(
    const Eigen::LLT<Eigen::MatrixXd>& chol, Xoshiro256& rng, int N)
{
    Eigen::VectorXd w(N);
    rng.fill_normal(w);
    return chol.matrixL() * w;
}

/// Generate fractional Brownian motion increments via Cholesky on
/// the exact covariance: Cov(B^H_s, B^H_t) = 0.5(|s|^{2H}+|t|^{2H}-|t-s|^{2H})
class FBMGenerator {
    int n_;
    double H_;
    double dt_;
    Eigen::LLT<Eigen::MatrixXd> chol_;

public:
    FBMGenerator() : n_(0), H_(0.1), dt_(1e-4) {}

    void init(int n_steps, double H, double dt) {
        n_ = n_steps; H_ = H; dt_ = dt;
        // Build covariance matrix of fBm values at t_1,...,t_n
        Eigen::MatrixXd C(n_, n_);
        double twoH = 2.0 * H_;
        for (int i = 0; i < n_; ++i) {
            double ti = (i + 1) * dt_;
            for (int j = 0; j <= i; ++j) {
                double tj = (j + 1) * dt_;
                double diff = std::abs(ti - tj);
                C(i, j) = 0.5 * (std::pow(ti, twoH) + std::pow(tj, twoH)
                                  - std::pow(diff, twoH));
                C(j, i) = C(i, j);
            }
        }
        chol_.compute(C);
    }

    /// Generate one fBm path [n_steps]
    Eigen::VectorXd generate(Xoshiro256& rng) const {
        Eigen::VectorXd w(n_);
        rng.fill_normal(w);
        return chol_.matrixL() * w;
    }

    /// Generate fBm increments (differences of consecutive values)
    Eigen::VectorXd generate_increments(Xoshiro256& rng) const {
        Eigen::VectorXd path = generate(rng);
        Eigen::VectorXd inc(n_);
        inc(0) = path(0);
        for (int i = 1; i < n_; ++i) inc(i) = path(i) - path(i - 1);
        return inc;
    }
};

/// Volterra fBm via Bennedsen hybrid scheme (1507.03004v4)
/// Ŵ^H(t_n) = sqrt(2H) Σ_{j=0}^{n-1} ∫_{t_j}^{t_{j+1}} (t_n-s)^{H-1/2} dW(s)
class VolterraFBM {
    int kappa_;      ///< Number of "power" cells near singularity
    double H_;
    double gamma_;   ///< γ = 1/2 - H
    double dt_;

    /// Precomputed b_j coefficients (Riemann sum weights)
    std::vector<double> b_coeff_;

public:
    VolterraFBM() : kappa_(1), H_(0.1), gamma_(0.4), dt_(1e-4) {}

    void init(double H, double dt, int kappa = 1) {
        H_ = H;
        gamma_ = 0.5 - H;
        dt_ = dt;
        kappa_ = kappa;
    }

    double cached_H() const { return H_; }

    /// Evaluate the Volterra integral at step n using past W increments.
    /// dW[j] = W(t_{j+1}) - W(t_j), j = 0,...,n-1
    double evaluate(int n, const Eigen::VectorXd& dW) const {
        if (n <= 0) return 0.0;
        double sqrt2H = std::sqrt(2.0 * H_);
        double val = 0.0;

        for (int j = 0; j < n; ++j) {
            int lag = n - j;  // t_n - t_j in units of dt

            double weight;
            if (lag <= kappa_) {
                // Power cell: average of (t-s)^{-γ} over [t_j, t_{j+1}]
                double a = (lag - 1) * dt_;
                double b = lag * dt_;
                double exp1 = 1.0 - gamma_;
                weight = ((std::pow(b, exp1) - std::pow(a, exp1)) / exp1) / dt_;
            } else {
                // Riemann: (t_n - t_{j+1/2})^{-γ}
                double midlag = (lag - 0.5) * dt_;
                weight = std::pow(midlag, -gamma_);
            }
            val += sqrt2H * weight * dW(j);
        }
        return val;
    }
};

} // namespace sovereign
