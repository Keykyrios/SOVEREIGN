#pragma once
/// @file state.hpp
/// @brief Layer 0 — Full simulation state tensor S_i(t).

#include <sovereign/config.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace sovereign {

struct LOBState {
    Eigen::VectorXi bid_volumes;
    Eigen::VectorXi ask_volumes;
    Eigen::VectorXd bid_prices;
    Eigen::VectorXd ask_prices;
    double best_bid = 0, best_ask = 0, mid_price = 0, spread = 0;
    int best_bid_idx = 0, best_ask_idx = 0;

    void resize(int M) {
        bid_volumes.setZero(M); ask_volumes.setZero(M);
        bid_prices.setZero(M);  ask_prices.setZero(M);
    }
    double imbalance() const {
        double b = bid_volumes.head(5).cast<double>().sum();
        double a = ask_volumes.head(5).cast<double>().sum();
        return (b - a) / (b + a + 1e-15);
    }
    double microprice() const {
        double b = bid_volumes(0), a = ask_volumes(0);
        return (a * best_bid + b * best_ask) / (a + b + 1e-15);
    }
};

struct alignas(64) AssetState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id = 0; double t = 0;
    // Layer 1
    double price = 100, log_price = 0, variance = 0.04, volatility = 0.2;
    double hurst = 0.1, jump_component = 0, lob_impact = 0;
    int regime = 2;
    // Layer 2
    Eigen::VectorXd hawkes_intensity;
    // Layer 3
    LOBState lob;
    // Layer 4
    double mm_inventory = 0, mm_spread = 0.01, mm_value = 0;
    // Layer 5
    double surplus = 1000, ruin_prob = 0, gerber_shiu = 0;
    // Returns
    double return_1 = 0, cum_return = 0;

    void init(int i, double p0, int M, int hdim) {
        id = i; price = p0; log_price = std::log(p0);
        lob.resize(M); hawkes_intensity.setZero(hdim);
    }
};

struct alignas(64) SimulationState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double t = 0; int step = 0;
    std::vector<AssetState, Eigen::aligned_allocator<AssetState>> assets;
    // Layer 6: Cross-asset
    Eigen::MatrixXd correlation, raw_correlation, distance;
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;
    double fiedler_value = 1; double clustering_coeff = 0;
    Eigen::VectorXd betweenness, degree;
    // Layer 7: TDA
    double tda_risk_index = 0, wasserstein_dist = 0;
    double landscape_l1 = 0, landscape_l2 = 0;
    // Layer 5: Aggregate
    Eigen::VectorXd ruin_vector;
    int total_events = 0; double wall_clock_s = 0;

    void init(const SimulationConfig& cfg) {
        const int N = cfg.universe.n_assets;
        const int M = cfg.lob.n_levels;
        const int hd = cfg.hawkes.n_order_types * cfg.hawkes.n_depth_levels;
        assets.resize(N);
        for (int i = 0; i < N; ++i) {
            assets[i].init(i, cfg.universe.initial_prices[i], M, hd);
            assets[i].surplus = cfg.ruin.initial_surplus;
        }
        correlation = Eigen::MatrixXd::Identity(N, N);
        raw_correlation = correlation;
        distance = Eigen::MatrixXd::Zero(N, N);
        eigenvalues = Eigen::VectorXd::Ones(N);
        eigenvectors = Eigen::MatrixXd::Identity(N, N);
        betweenness = Eigen::VectorXd::Zero(N);
        degree = Eigen::VectorXd::Zero(N);
        ruin_vector = Eigen::VectorXd::Zero(N);
    }
    Eigen::VectorXd prices() const {
        Eigen::VectorXd p(assets.size());
        for (size_t i = 0; i < assets.size(); ++i) p(i) = assets[i].price;
        return p;
    }
    Eigen::VectorXd returns() const {
        Eigen::VectorXd r(assets.size());
        for (size_t i = 0; i < assets.size(); ++i) r(i) = assets[i].return_1;
        return r;
    }
};

} // namespace sovereign
