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
    const LOBConfig* cfg_ptr_;  ///< Pointer avoids UB in default ctor
    const LOBConfig& cfg() const { return *cfg_ptr_; }
    int M_;              ///< Levels per side
    double tick_;
    double mid_price_;

    // Volumes at each level
    Eigen::VectorXi bid_vol_;
    Eigen::VectorXi ask_vol_;

    // Hidden (iceberg) volumes
    Eigen::VectorXi bid_hidden_;
    Eigen::VectorXi ask_hidden_;

    // Iceberg revelation latency queues (Phase 5)
    Eigen::VectorXi bid_pending_reveal_;
    Eigen::VectorXi ask_pending_reveal_;

    // O(1) best level tracking
    int best_bid_level_ = 0;
    int best_ask_level_ = 0;

public:
    OrderBook() : cfg_ptr_(nullptr), M_(0), tick_(0.01), mid_price_(100) {}

    explicit OrderBook(const LOBConfig& cfg, double initial_price)
        : cfg_ptr_(&cfg), M_(cfg.n_levels), tick_(cfg.tick_size)
    {
        // Phase 5: Regulation NMS Sub-Penny enforcement
        mid_price_ = std::round(initial_price / tick_) * tick_;
        bid_vol_.setZero(M_);
        ask_vol_.setZero(M_);
        bid_hidden_.setZero(M_);
        ask_hidden_.setZero(M_);
        bid_pending_reveal_.setZero(M_);
        ask_pending_reveal_.setZero(M_);
        initialize_book();
    }

    void initialize_book() {
        for (int d = 0; d < M_; ++d) {
            double depth_factor = cfg().initial_depth * std::exp(-0.01 * d);
            int lots = std::max(1, static_cast<int>(std::round(depth_factor)));
            bid_vol_(d) = lots;
            ask_vol_(d) = lots;
        }
        best_bid_level_ = 0;
        best_ask_level_ = 0;
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

    /// Market order: walk the book consuming liquidity.
    /// Mid-price is structurally derived from book state, not floating.
    void process_market_order(const Event& ev, Xoshiro256& /*rng*/) {
        int remaining = std::max(1, static_cast<int>(std::round(ev.size)));
        auto& book = ev.is_buy ? ask_vol_ : bid_vol_;
        auto& hidden = ev.is_buy ? ask_hidden_ : bid_hidden_;
        auto& pending = ev.is_buy ? ask_pending_reveal_ : bid_pending_reveal_;

        for (int d = 0; d < M_ && remaining > 0; ++d) {
            // Phase 5: Prevent instantaneous iceberg revelation (HFT latency)
            if (hidden(d) > 0 && book(d) <= remaining) {
                // Don't add to book instantly; push to pending queue
                pending(d) += hidden(d);
                hidden(d) = 0;
            }

            int fill = std::min(remaining, book(d));
            book(d) -= fill;
            remaining -= fill;
        }

        // O(1) best level update after volume consumed
        update_best_levels();
    }

    /// O(M) complete scan for best levels (called when levels may shift arbitrarily)
    void update_best_levels() {
        best_bid_level_ = 0;
        while (best_bid_level_ < M_ - 1 && bid_vol_(best_bid_level_) <= 0)
            best_bid_level_++;
        best_ask_level_ = 0;
        while (best_ask_level_ < M_ - 1 && ask_vol_(best_ask_level_) <= 0)
            best_ask_level_++;
    }

    /// Limit order: add volume at specified depth
    void process_limit_order(const Event& ev) {
        int d = std::min(ev.depth_level, M_ - 1);
        auto& book = ev.is_buy ? bid_vol_ : ask_vol_;
        int lots = std::max(1, static_cast<int>(std::round(ev.size)));
        book(d) += lots;
        // O(1) best level tighten
        if (ev.is_buy && d < best_bid_level_) best_bid_level_ = d;
        if (!ev.is_buy && d < best_ask_level_) best_ask_level_ = d;
    }

    /// Cancellation: remove volume, rate proportional to distance
    void process_cancel(const Event& ev, Xoshiro256& rng) {
        auto& book = ev.is_buy ? bid_vol_ : ask_vol_;
        for (int d = 0; d < M_; ++d) {
            double cancel_prob = cfg().cancel_rate_base
                               * std::exp(-cfg().cancel_distance_decay * d);
            if (rng.uniform() < cancel_prob && book(d) > 0) {
                int cancel_amount = std::max(1, static_cast<int>(std::round(ev.size * 0.5)));
                cancel_amount = std::min(book(d), cancel_amount);
                book(d) -= cancel_amount;
                if (book(d) <= 0) update_best_levels();
                break;
            }
        }
    }

    /// Modify: shift order to different level
    void process_modify(const Event& ev, Xoshiro256& rng) {
        auto& book = ev.is_buy ? bid_vol_ : ask_vol_;
        int from = std::min(ev.depth_level, M_ - 1);
        int shift = std::clamp(static_cast<int>(rng.normal() * 2), -3, 3);
        int to = std::clamp(from + shift, 0, M_ - 1);
        if (from != to && book(from) > 0) {
            int amount = std::min(book(from), std::max(1, static_cast<int>(std::round(ev.size))));
            book(from) -= amount;
            book(to) += amount;
            update_best_levels();
        }
    }

    /// Hidden iceberg order
    void process_hidden(const Event& ev, Xoshiro256& rng) {
        if (rng.uniform() < cfg().iceberg_probability) {
            int d = std::min(ev.depth_level, M_ - 1);
            auto& hidden = ev.is_buy ? bid_hidden_ : ask_hidden_;
            int lots = std::max(1, static_cast<int>(std::round(ev.size * 2.0)));
            hidden(d) += lots;
        }
    }

    /// Passive cancellations: natural decay each timestep and asymmetric replenishment
    void passive_decay(double dt, double macro_price, Xoshiro256& rng) {
        double drift_speed = 1.0;
        mid_price_ += drift_speed * (macro_price - mid_price_) * dt;
        mid_price_ = std::round(mid_price_ / tick_) * tick_; // Regulation NMS sub-penny tick enforcement

        for (int d = 0; d < M_; ++d) {
            double decay_rate = cfg().cancel_rate_base
                              * std::exp(-cfg().cancel_distance_decay * d) * dt;
            if (rng.uniform() < (1.0 - std::exp(-decay_rate))) {
                if (bid_vol_(d) > 0) bid_vol_(d) -= 1;
                if (ask_vol_(d) > 0) ask_vol_(d) -= 1;
            }
            
            // Phase 5: Delayed iceberg revelation (5us latency gap equivalent)
            // 1 - exp(-rate * dt). For a 5us delay, rate = 200,000.
            if (bid_pending_reveal_(d) > 0 && rng.uniform() < (1.0 - std::exp(-200000.0 * dt))) {
                bid_vol_(d) += bid_pending_reveal_(d);
                bid_pending_reveal_(d) = 0;
            }
            if (ask_pending_reveal_(d) > 0 && rng.uniform() < (1.0 - std::exp(-200000.0 * dt))) {
                ask_vol_(d) += ask_pending_reveal_(d);
                ask_pending_reveal_(d) = 0;
            }
        }

        double bias = (macro_price - mid_price_) / std::max(mid_price_, 1e-6);
        bias = std::clamp(bias * 100.0, -0.8, 0.8);

        for (int d = 0; d < M_; ++d) {
            double replenish = 0.1 * cfg().initial_depth * std::exp(-0.01 * d) * dt;
            if (rng.uniform() < replenish * (1.0 + bias)) bid_vol_(d) += 1;
            if (rng.uniform() < replenish * (1.0 - bias)) ask_vol_(d) += 1;
        }
        update_best_levels();
    }

    /// Sync LOBState struct from internal state
    void update_lob_state(LOBState& s) const {
        s.bid_volumes = bid_vol_;
        s.ask_volumes = ask_vol_;
        s.mid_price = mid_price_;

        // Use ACTUAL cached best levels, not hardcoded 1-tick spread
        s.best_bid = mid_price_ - (best_bid_level_ + 0.5) * tick_;
        s.best_ask = mid_price_ + (best_ask_level_ + 0.5) * tick_;
        s.spread = s.best_ask - s.best_bid;

        for (int d = 0; d < M_; ++d) {
            s.bid_prices(d) = mid_price_ - (d + 0.5) * tick_;
            s.ask_prices(d) = mid_price_ + (d + 0.5) * tick_;
        }
    }

    double mid_price() const { return mid_price_; }
    void set_mid_price(double p) { mid_price_ = p; }
    const Eigen::VectorXi& bid_volumes() const { return bid_vol_; }
    const Eigen::VectorXi& ask_volumes() const { return ask_vol_; }
};

/// Manages N order books
class LOBEngine {
    const LOBConfig& cfg_;
    int N_;
    std::vector<OrderBook> books_;
    std::vector<Xoshiro256> thread_rngs_;

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
    void init_thread_rngs(Xoshiro256& rng) {
        thread_rngs_.resize(N_);
        for (int i = 0; i < N_; ++i) thread_rngs_[i] = rng.fork();
    }

    void step(SimulationState& state, double dt, Xoshiro256& /*rng*/) {
        #pragma omp parallel for
        for (int i = 0; i < N_; ++i) {
            books_[i].passive_decay(dt, state.assets[i].price, thread_rngs_[i]);
            books_[i].update_lob_state(state.assets[i].lob);
        }
    }

    /// Compute LOB-induced impact integral for each asset
    void compute_impact(SimulationState& state) const {
        #pragma omp parallel for
        for (int i = 0; i < N_; ++i) {
            const auto& lob = state.assets[i].lob;
            // ∫ [D(p) - S(p)] · K(p - P) dp
            double impact = 0;
            for (int d = 0; d < std::min(20, (int)lob.bid_volumes.size()); ++d) {
                double dist = (d + 1) * cfg_.tick_size;
                // Bouchaud square-root / Kyle's Lambda concave impact propagator
                // Impact grows as sqrt(imbalance) instead of a linear/Gaussian blur
                double imbalance = static_cast<double>(lob.bid_volumes(d) - lob.ask_volumes(d));
                double sign = imbalance > 0 ? 1.0 : (imbalance < 0 ? -1.0 : 0.0);
                double depth_weight = 1.0 / (1.0 + dist / cfg_.tick_size); // Decays with depth
                impact += sign * std::sqrt(std::abs(imbalance)) * depth_weight;
            }
            // Scale by dt: impact is a rate (per unit time), not per-tick.
            // Without this, impact accumulates N_steps times faster than intended.
            state.assets[i].lob_impact = impact * 1e-4;
        }
    }

    OrderBook& book(int i) { return books_[i]; }
};

} // namespace sovereign
