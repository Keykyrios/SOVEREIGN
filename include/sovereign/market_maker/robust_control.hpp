#pragma once
/// @file robust_control.hpp
/// @brief Layer 4 — MM optimization under κ-ambiguity (Maenhout).
///
/// δ* = (γ+θ)·σ²·|I|/2 + spread_floor + ruin_penalty

#include <sovereign/config.hpp>
#include <sovereign/core/random.hpp>
#include <sovereign/core/state.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

namespace sovereign {

struct MMAgent {
    int id = 0;
    double inventory = 0, wealth = 0, value_fn = 0;
    double bid_spread = 0.01, ask_spread = 0.01;
};

class MarketMakerEngine {
    const MarketMakerConfig& cfg_;
    int N_, M_;
    std::vector<std::vector<MMAgent>> agents_;

    // HJB grid per asset: V(inventory, price_dev)
    struct HJBGrid {
        Eigen::MatrixXd V, delta;
        int ni, np; double inv_max;
        void init(int n, double imax) {
            ni = np = n; inv_max = imax;
            V.setZero(ni, np); delta.setConstant(ni, np, 0.01);
        }
    };
    std::vector<HJBGrid> grids_;

    double robust_spread(double inv, double vol, double ruin) const {
        double g = cfg_.gamma + cfg_.theta;
        return std::max(cfg_.spread_floor,
            g * vol * vol * std::abs(inv) * 0.5 + 2.0 * ruin);
    }

public:
    MarketMakerEngine(const MarketMakerConfig& cfg, int n_assets)
        : cfg_(cfg), N_(n_assets), M_(cfg.n_market_makers)
    {
        agents_.resize(N_);
        grids_.resize(N_);
        for (int i = 0; i < N_; ++i) {
            agents_[i].resize(M_);
            for (int m = 0; m < M_; ++m) agents_[i][m].id = m;
            grids_[i].init(cfg_.hjb_grid_points, cfg_.inventory_limit);
        }
    }

    /// Howard's policy iteration step on HJB
    void solve_hjb_step(int asset, double vol, double dt) {
        auto& g = grids_[asset];
        double di = 2.0 * g.inv_max / (g.ni - 1);
        double dp = 0.2 / (g.np - 1);
        Eigen::MatrixXd Vn = g.V;

        for (int ii = 1; ii < g.ni - 1; ++ii) {
            double inv = -g.inv_max + ii * di;
            for (int pi = 1; pi < g.np - 1; ++pi) {
                double geff = cfg_.gamma + cfg_.theta;
                double d2V = (g.V(ii, pi+1) - 2*g.V(ii,pi) + g.V(ii, pi-1)) / (dp*dp);
                double delta_opt = robust_spread(inv, vol, 0);
                double lam = 10.0 * std::exp(-delta_opt / vol);
                double revenue = lam * delta_opt;
                double risk = -0.5 * geff * vol * vol * inv * inv;
                Vn(ii, pi) = g.V(ii, pi) + dt * (0.5*vol*vol*d2V + revenue + risk);
                g.delta(ii, pi) = delta_opt;
            }
        }
        g.V = Vn;
    }

    void step(SimulationState& state, double dt, Xoshiro256& rng) {
        for (int i = 0; i < N_; ++i) {
            auto& a = state.assets[i];
            double vol = a.volatility, ruin = a.ruin_prob;
            double tot_inv = 0, tot_spread = 0;

            for (int m = 0; m < M_; ++m) {
                auto& mm = agents_[i][m];
                double sp = robust_spread(mm.inventory, vol, ruin);
                double skew = cfg_.gamma * vol * vol * mm.inventory * dt;
                mm.bid_spread = std::max(sp * 0.5 - skew, cfg_.spread_floor * 0.5);
                mm.ask_spread = std::max(sp * 0.5 + skew, cfg_.spread_floor * 0.5);
                double fill = 5.0 * std::exp(-sp / vol) * dt;
                mm.inventory += fill * 0.1 * rng.normal();
                mm.inventory = std::clamp(mm.inventory,
                    -cfg_.inventory_limit, cfg_.inventory_limit);
                tot_inv += mm.inventory;
                tot_spread += mm.bid_spread + mm.ask_spread;
            }
            a.mm_spread = tot_spread / M_;
            a.mm_inventory = tot_inv;
        }
    }

    void periodic_hjb(SimulationState& s, double dt) {
        for (int i = 0; i < N_; ++i)
            solve_hjb_step(i, s.assets[i].volatility, dt * 100);
    }
};

} // namespace sovereign
