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
#include <mutex>
#include <future>
#include <atomic>
#include <thread>
#include <condition_variable>

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
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::MatrixXd V, delta;
        int ni, np; double inv_max;
        void init(int n, double imax) {
            ni = np = n; inv_max = imax;
            V.setZero(ni, np); delta.setConstant(ni, np, 0.01);
        }
    };
    std::vector<HJBGrid, Eigen::aligned_allocator<HJBGrid>> grids_;
    std::vector<HJBGrid, Eigen::aligned_allocator<HJBGrid>> grids_back_;
    mutable std::mutex hjb_mutex_;
    std::atomic<bool> hjb_busy_{false};
    
    // Dedicated background thread
    std::thread hjb_thread_;
    std::condition_variable hjb_cv_;
    std::mutex hjb_cv_mutex_;
    bool hjb_quit_ = false;
    bool hjb_pending_ = false;
    std::vector<double> hjb_vols_;
    double hjb_dt_ = 0;

    void hjb_worker() {
        while (true) {
            std::vector<double> vols;
            double dt;
            {
                std::unique_lock<std::mutex> lock(hjb_cv_mutex_);
                hjb_cv_.wait(lock, [this]() { return hjb_pending_ || hjb_quit_; });
                if (hjb_quit_) break;
                vols = hjb_vols_;
                dt = hjb_dt_;
                hjb_pending_ = false;
            }

            for (int i = 0; i < N_; ++i) {
                HJBGrid& g = grids_back_[i];
                double vol = vols[i];
                double dp = 0.01;
                double di = 2.0 * g.inv_max / (g.ni - 1);
                
                double vol2 = vol * vol + 1e-12;
                double max_drift = std::max(cfg_.gamma * g.inv_max * vol, 1e-12);
                double dt_cfl_diff = dp * dp / vol2 * 0.4;
                double dt_cfl_adv  = dp / max_drift * 0.4;
                double dt_cfl = std::min(dt_cfl_diff, dt_cfl_adv);
                int n_substeps = std::max(1, static_cast<int>(std::ceil(dt * 100 / dt_cfl)));
                n_substeps = std::min(n_substeps, 200);
                double dt_sub = (dt * 100) / n_substeps;

                for (int sub = 0; sub < n_substeps; ++sub) {
                    Eigen::MatrixXd Vn = g.V;
                    for (int ii = 1; ii < g.ni - 1; ++ii) {
                        double inv = -g.inv_max + ii * di;
                        for (int pi = 1; pi < g.np - 1; ++pi) {
                            double geff = cfg_.gamma + cfg_.theta;
                            double d2V = (g.V(ii, pi+1) - 2*g.V(ii,pi) + g.V(ii, pi-1)) / (dp*dp);
                            double drift_sign = -geff * inv;
                            double dV_di;
                            if (drift_sign >= 0) dV_di = (g.V(std::min(ii+1, g.ni-1), pi) - g.V(ii, pi)) / di;
                            else dV_di = (g.V(ii, pi) - g.V(std::max(ii-1, 0), pi)) / di;
                            double delta_opt = std::max(vol + geff * std::abs(inv) * vol, cfg_.spread_floor);
                            delta_opt += std::min(std::abs(dV_di) * 0.1, vol * 10.0);
                            g.delta(ii, pi) = delta_opt;
                            
                            double kappa = std::max(vol, 0.01);
                            double lam = 10.0 * std::exp(-delta_opt / kappa);
                            double revenue = lam * delta_opt;
                            double risk = -0.5 * geff * vol2 * inv * inv;
                            double discount = -0.5 * g.V(ii, pi);
                            Vn(ii, pi) = g.V(ii, pi) + dt_sub * (
                                0.5 * vol2 * d2V + drift_sign * dV_di
                                + revenue + risk + discount
                            );
                            Vn(ii, pi) = std::clamp(Vn(ii, pi), -1e4, 1e4);
                        }
                    }
                    g.V = Vn;
                }
            }
            
            {
                std::lock_guard<std::mutex> lock(hjb_mutex_);
                grids_ = grids_back_;
            }
            hjb_busy_ = false;
        }
    }

    /// Spread from HJB grid (if solved) or closed-form fallback
    double robust_spread(int asset, double inv, double vol, double ruin) const {
        const auto& g = grids_[asset];
        // Map inventory to grid index
        double di = 2.0 * g.inv_max / (g.ni - 1);
        int ii = static_cast<int>((inv + g.inv_max) / di);
        ii = std::clamp(ii, 1, g.ni - 2);
        int pi = g.np / 2;  // center of price deviation grid

        double grid_delta = 0;
        {
            // Try to lock, but if we can't, just use closed-form to not block tick loop
            std::unique_lock<std::mutex> lock(hjb_mutex_, std::try_to_lock);
            if (lock.owns_lock()) {
                grid_delta = g.delta(ii, pi);
            } else {
                grid_delta = 0;
            }
        }
        if (grid_delta > 1e-6) {
            // Use HJB-solved optimal spread
            return std::max(cfg_.spread_floor, grid_delta + 2.0 * ruin);
        }
        // Fallback: closed-form
        double geff = cfg_.gamma + cfg_.theta;
        return std::max(cfg_.spread_floor,
            geff * vol * vol * std::abs(inv) * 0.5 + 2.0 * ruin);
    }

public:
    MarketMakerEngine(const MarketMakerConfig& cfg, int n_assets)
        : cfg_(cfg), N_(n_assets), M_(cfg.n_market_makers)
    {
        agents_.resize(N_);
        grids_.resize(N_);
        grids_back_.resize(N_);
        for (int i = 0; i < N_; ++i) {
            agents_[i].resize(M_);
            for (int m = 0; m < M_; ++m) agents_[i][m].id = m;
            grids_[i].init(30, cfg_.inventory_limit);
            grids_back_[i].init(30, cfg_.inventory_limit);
        }
        hjb_thread_ = std::thread(&MarketMakerEngine::hjb_worker, this);
    }

    ~MarketMakerEngine() {
        {
            std::lock_guard<std::mutex> lock(hjb_cv_mutex_);
            hjb_quit_ = true;
        }
        hjb_cv_.notify_one();
        if (hjb_thread_.joinable()) {
            hjb_thread_.join();
        }
    }

    /// Howard's policy iteration step on HJB with CFL enforcement.
    /// CFL condition: dt <= dp^2 / sigma^2 prevents FTCS divergence.
    void solve_hjb_step(int asset, double vol, double dt_total) {
        auto& g = grids_[asset];
        double di = 2.0 * g.inv_max / (g.ni - 1);
        double dp = 0.2 / (g.np - 1);

        // CFL stability: dt_safe <= min(dp^2/vol^2, dp/|max_drift|)
        // Advection-diffusion combined CFL
        double vol2 = vol * vol + 1e-12;
        double max_drift = std::max(cfg_.gamma * g.inv_max * vol, 1e-12);
        double dt_cfl_diff = dp * dp / vol2 * 0.4;
        double dt_cfl_adv  = dp / max_drift * 0.4;
        double dt_cfl = std::min(dt_cfl_diff, dt_cfl_adv);
        int n_substeps = std::max(1, static_cast<int>(std::ceil(dt_total / dt_cfl)));
        // Hard cap: prevents hanging on extreme vol spikes
        n_substeps = std::min(n_substeps, 200);
        double dt = dt_total / n_substeps;

        for (int sub = 0; sub < n_substeps; ++sub) {
            Eigen::MatrixXd Vn = g.V;
            for (int ii = 1; ii < g.ni - 1; ++ii) {
                double inv = -g.inv_max + ii * di;
                for (int pi = 1; pi < g.np - 1; ++pi) {
                    double geff = cfg_.gamma + cfg_.theta;
                    double d2V = (g.V(ii, pi+1) - 2*g.V(ii,pi) + g.V(ii, pi-1)) / (dp*dp);

                    // Policy improvement: upwind differencing on inventory drift
                    double drift_sign = -geff * inv;
                    double dV_di;
                    if (drift_sign >= 0) {
                        dV_di = (g.V(std::min(ii+1, g.ni-1), pi) - g.V(ii, pi)) / di;
                    } else {
                        dV_di = (g.V(ii, pi) - g.V(std::max(ii-1, 0), pi)) / di;
                    }
                    double delta_opt = std::max(vol + geff * std::abs(inv) * vol,
                                               cfg_.spread_floor);
                    // Let delta_opt float naturally with volatility and gradient
                    delta_opt += std::min(std::abs(dV_di) * 0.1, vol * 10.0);

                    double kappa = std::max(vol, 0.01);
                    double lam = 10.0 * std::exp(-delta_opt / kappa);
                    
                    // Phase 5: Maker/Taker Rebates. Inject exchange rebate into utility
                    double rebate = 0.0002;
                    double revenue = lam * (delta_opt + rebate);
                    
                    double risk = -0.5 * geff * vol * vol * inv * inv;
                    // Discount factor prevents V from growing unboundedly
                    double discount = -0.5 * g.V(ii, pi);
                    Vn(ii, pi) = g.V(ii, pi) + dt * (0.5*vol*vol*d2V + drift_sign * dV_di + revenue + risk + discount);
                    g.delta(ii, pi) = delta_opt;
                }
            }
            g.V = Vn;
            // Clamp V INSIDE the substep loop to prevent runaway explosion
            g.V = g.V.cwiseMax(-1e4).cwiseMin(1e4);

            // Neumann boundary: dV/dI = 0 at inventory limits
            // No trading at limits, so value is flat
            for (int pi = 0; pi < g.np; ++pi) {
                g.V(0, pi) = g.V(1, pi);
                g.V(g.ni - 1, pi) = g.V(g.ni - 2, pi);
            }
        }
    }

    void step(SimulationState& state, double dt, Xoshiro256& rng) {
        for (int i = 0; i < N_; ++i) {
            auto& a = state.assets[i];
            double vol = a.volatility, ruin = a.ruin_prob;
            double tot_inv = 0, tot_spread = 0;

            for (int m = 0; m < M_; ++m) {
                auto& mm = agents_[i][m];
                double sp = robust_spread(i, mm.inventory, vol, ruin);
                // Skew is a price offset (spread), not a rate. Do not multiply by dt!
                double skew = cfg_.gamma * vol * vol * mm.inventory;
                // Asymmetric penalty: if ruin is high, penalize being short more heavily
                if (ruin > 0.01 && mm.inventory < 0) skew *= 2.0; 
                mm.bid_spread = std::max(sp * 0.5 - skew, cfg_.spread_floor * 0.5);
                mm.ask_spread = std::max(sp * 0.5 + skew, cfg_.spread_floor * 0.5);

                // Avellaneda-Stoikov fill model:
                // λ(δ) = A · exp(-k · δ) where k ≈ 1.0 keeps exponent O(1)
                // Fill PROBABILITY per tick is λ·dt (Bernoulli)
                // Add network latency penalty: MM isn't God, they have queue position risk
                double network_latency_penalty = 0.05 * std::abs(mm.inventory); // The more you hold, the worse your queue pos
                double A = 100.0;    // base arrival rate (fills per unit time)
                double k = 1.0;      // price sensitivity of fill rate
                double rate_bid = A * std::exp(-k * mm.bid_spread - network_latency_penalty);
                double rate_ask = A * std::exp(-k * mm.ask_spread - network_latency_penalty);
                // Exact Poisson probability to prevent flatlining at 1.0 for large dt
                double prob_bid = 1.0 - std::exp(-rate_bid * dt);
                double prob_ask = 1.0 - std::exp(-rate_ask * dt);

                double fill_size = 1.0;
                bool bid_hit = (rng.uniform() < prob_bid);
                bool ask_hit = (rng.uniform() < prob_ask);

                // Phase 4: Self-Match Prevention (SMP)
                // Prevent synthetic wash trades where MM crosses its own spread in high-vol avalanches
                if (bid_hit && ask_hit) {
                    if (rng.uniform() < 0.5) bid_hit = false;
                    else ask_hit = false;
                }

                mm.inventory += (bid_hit ? fill_size : 0.0) - (ask_hit ? fill_size : 0.0);
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
        if (hjb_busy_.exchange(true)) return; // Already running

        {
            std::lock_guard<std::mutex> lock(hjb_cv_mutex_);
            hjb_vols_.resize(N_);
            for(int i=0; i<N_; ++i) hjb_vols_[i] = s.assets[i].volatility;
            hjb_dt_ = dt;
            hjb_pending_ = true;
        }
        hjb_cv_.notify_one();
    }
};

} // namespace sovereign
