#pragma once
/// @file lob.hpp
/// @brief Layer 3 — Full-depth limit order book with queue dynamics.
///
/// 2M+1 price levels, FIFO queues, iceberg detection, cancellation
/// proportional to distance from best quote.

#include <sovereign/config.hpp>
#include <sovereign/core/clock.hpp>
#include <sovereign/core/random.hpp>
#include <sovereign/core/state.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

namespace sovereign {

class OrderBook {
    const LOBConfig& cfg_;
    int M_;              ///< Levels per side
    double tick_;
    double mid_price_;

    // Volumes at each level
    Eigen::VectorXd bid_vol_;  ///< [M] from best bid outward
    Eigen::VectorXd ask_vol_;  ///< [M] from best ask outward

    // Hidden (iceberg) volumes
    Eigen::VectorXd bid_hidden_;
    Eigen::VectorXd ask_hidden_;

public:
    OrderBook() : cfg_(*static_cast<const LOBConfig*>(nullptr)), M_(0), tick_(0.01), mid_price_(100) {}

    explicit OrderBook(const LOBConfig& cfg, double initial_price)
        : cfg_(cfg), M_(cfg.n_levels), tick_(cfg.tick_size), mid_price_(initial_price)
    {
        bid_vol_.setZero(M_);
        ask_vol_.setZero(M_);
        bid_hidden_.setZero(M_);
        ask_hidden_.setZero(M_);
        initialize_book();
    }

    void initialize_book() {
        // Concave LOB shape: more volume near mid, less at extremes
        for (int d = 0; d < M_; ++d) {
            double depth_factor = cfg_.initial_depth * std::exp(-0.01 * d);
            bid_vol_(d) = depth_factor;
            ask_vol_(d) = depth_factor;
        }
    }

    /// Process a Hawkes-generated event
    void process_event(const Event& ev, LOBState& lob_state, Xoshiro256& rng) {
        switch (ev.event_type) {
            case 0: process_market_order(ev, rng); break;
            case 1: process_limit_order(ev); break;
            case 2: process_cancel(ev, rng); break;
            case 3: process_modify(ev, rng); break;
            case 4: process_hidden(ev, rng); break;
        }
        update_lob_state(lob_state);
    }

    /// Market order: walk the book consuming liquidity
    void process_market_order(const Event& ev, Xoshiro256& rng) {
        double remaining = ev.size;
        auto& book = ev.is_buy ? ask_vol_ : bid_vol_;
        auto& hidden = ev.is_buy ? ask_hidden_ : bid_hidden_;

        for (int d = 0; d < M_ && remaining > 0; ++d) {
            // Reveal iceberg
            if (hidden(d) > 0 && book(d) < remaining) {
                book(d) += hidden(d);
                hidden(d) = 0;
            }

            double fill = std::min(remaining, book(d));
            book(d) -= fill;
            remaining -= fill;

            if (fill > 0) {
                // Price impact: mid-price moves
                double impact = tick_ * (d + 1) * fill / (cfg_.initial_depth + 1e-10);
                mid_price_ += ev.is_buy ? impact : -impact;
            }
        }
    }

    /// Limit order: add volume at specified depth
    void process_limit_order(const Event& ev) {
        int d = std::min(ev.depth_level, M_ - 1);
        auto& book = ev.is_buy ? bid_vol_ : ask_vol_;
        book(d) += ev.size;
    }

    /// Cancellation: remove volume, rate proportional to distance
    void process_cancel(const Event& ev, Xoshiro256& rng) {
        auto& book = ev.is_buy ? bid_vol_ : ask_vol_;
        // Cancel at random level weighted by distance decay
        for (int d = 0; d < M_; ++d) {
            double cancel_prob = cfg_.cancel_rate_base
                               * std::exp(-cfg_.cancel_distance_decay * d);
            if (rng.uniform() < cancel_prob && book(d) > 0) {
                double cancel_amount = std::min(book(d), ev.size * 0.5);
                book(d) -= cancel_amount;
                break;
            }
        }
    }

    /// Modify: shift order to different level
    void process_modify(const Event& ev, Xoshiro256& rng) {
        auto& book = ev.is_buy ? bid_vol_ : ask_vol_;
        int from = std::min(ev.depth_level, M_ - 1);
        int to = std::min(from + static_cast<int>(rng.normal() * 2), M_ - 1);
        to = std::max(0, to);
        if (from != to && book(from) > 0) {
            double amount = std::min(book(from), ev.size);
            book(from) -= amount;
            book(to) += amount;
        }
    }

    /// Hidden iceberg order
    void process_hidden(const Event& ev, Xoshiro256& rng) {
        if (rng.uniform() < cfg_.iceberg_probability) {
            int d = std::min(ev.depth_level, M_ - 1);
            auto& hidden = ev.is_buy ? bid_hidden_ : ask_hidden_;
            hidden(d) += ev.size * 2.0;  // Hidden portion is larger
        }
    }

    /// Passive cancellations: natural decay each timestep
    void passive_decay(double dt, Xoshiro256& rng) {
        for (int d = 0; d < M_; ++d) {
            double decay_rate = cfg_.cancel_rate_base
                              * std::exp(-cfg_.cancel_distance_decay * d) * dt;
            if (rng.uniform() < decay_rate) {
                bid_vol_(d) *= (1.0 - 0.01);
                ask_vol_(d) *= (1.0 - 0.01);
            }
        }
        // Natural replenishment at outer levels
        for (int d = M_ / 2; d < M_; ++d) {
            double replenish = 0.1 * cfg_.initial_depth * std::exp(-0.01 * d) * dt;
            bid_vol_(d) += replenish;
            ask_vol_(d) += replenish;
        }
    }

    /// Sync LOBState struct from internal state
    void update_lob_state(LOBState& s) const {
        s.bid_volumes = bid_vol_;
        s.ask_volumes = ask_vol_;
        s.mid_price = mid_price_;
        s.spread = tick_;  // Minimum spread

        // Find best bid/ask (first level with volume > 0)
        s.best_bid = mid_price_ - 0.5 * tick_;
        s.best_ask = mid_price_ + 0.5 * tick_;
        for (int d = 0; d < M_; ++d) {
            s.bid_prices(d) = mid_price_ - (d + 0.5) * tick_;
            s.ask_prices(d) = mid_price_ + (d + 0.5) * tick_;
        }
        s.spread = s.best_ask - s.best_bid;
    }

    double mid_price() const { return mid_price_; }
    void set_mid_price(double p) { mid_price_ = p; }
    const Eigen::VectorXd& bid_volumes() const { return bid_vol_; }
    const Eigen::VectorXd& ask_volumes() const { return ask_vol_; }
};

/// Manages N order books
class LOBEngine {
    const LOBConfig& cfg_;
    int N_;
    std::vector<OrderBook> books_;

public:
    LOBEngine(const LOBConfig& cfg, int n_assets,
              const std::vector<double>& initial_prices)
        : cfg_(cfg), N_(n_assets)
    {
        books_.reserve(N_);
        for (int i = 0; i < N_; ++i)
            books_.emplace_back(cfg_, initial_prices[i]);
    }

    /// Process all events from the clock
    void process_events(const std::vector<Event>& events,
                        SimulationState& state, Xoshiro256& rng)
    {
        for (const auto& ev : events) {
            if (ev.asset_id >= 0 && ev.asset_id < N_) {
                books_[ev.asset_id].process_event(
                    ev, state.assets[ev.asset_id].lob, rng);
            }
        }
    }

    /// Passive dynamics (decay + replenishment)
    void step(SimulationState& state, double dt, Xoshiro256& rng) {
        for (int i = 0; i < N_; ++i) {
            books_[i].set_mid_price(state.assets[i].price);
            books_[i].passive_decay(dt, rng);
            books_[i].update_lob_state(state.assets[i].lob);
        }
    }

    /// Compute LOB-induced impact integral for each asset
    void compute_impact(SimulationState& state) const {
        for (int i = 0; i < N_; ++i) {
            const auto& lob = state.assets[i].lob;
            // ∫ [D(p) - S(p)] · K(p - P) dp
            double impact = 0;
            for (int d = 0; d < std::min(20, (int)lob.bid_volumes.size()); ++d) {
                double dist = (d + 1) * cfg_.tick_size;
                double kernel = std::exp(-dist * dist / (2.0 * 0.1 * 0.1));
                impact += (lob.bid_volumes(d) - lob.ask_volumes(d)) * kernel;
            }
            state.assets[i].lob_impact = impact * 1e-4;
        }
    }

    OrderBook& book(int i) { return books_[i]; }
};

} // namespace sovereign
