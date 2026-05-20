#pragma once
/// @file engine.hpp
/// @brief Main simulation orchestrator — couples all 8 layers.
///
/// Tick loop:
///   1. Rough vol + CGMY jumps (Layer 1)
///   2. Hawkes event generation (Layer 2)
///   3. LOB processing (Layer 3)
///   4. Market maker update (Layer 4)
///   5. Ruin dynamics (Layer 5)
///   6. Correlation + topology (Layer 6) — every K steps
///   7. Persistent homology (Layer 7) — every K steps
///   8. Contagion diffusion (Layer 6.4)
///   9. Feedback: Γ→Hawkes, Γ→MM, topology→Hawkes

#include <sovereign/config.hpp>
#include <sovereign/core/state.hpp>
#include <sovereign/core/clock.hpp>
#include <sovereign/core/random.hpp>
#include <sovereign/price/rough_vol.hpp>
#include <sovereign/price/levy_jumps.hpp>
#include <sovereign/price/regime.hpp>
#include <sovereign/hawkes/multivariate.hpp>
#include <sovereign/orderbook/lob.hpp>
#include <sovereign/market_maker/robust_control.hpp>
#include <sovereign/ruin/gerber_shiu.hpp>
#include <sovereign/topology/correlation.hpp>
#include <sovereign/topology/graphs.hpp>
#include <sovereign/topology/spectral.hpp>
#include <sovereign/topology/contagion.hpp>
#include <sovereign/tda/persistence.hpp>
#include <sovereign/tda/landscapes.hpp>
#include <sovereign/mc/mlmc.hpp>
#include <sovereign/viz/telemetry.hpp>

#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <functional>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

namespace sovereign {

/// Callback for external monitoring (dashboard, logging, etc.)
using StepCallback = std::function<void(const SimulationState&, int step)>;

class Engine {
    SimulationConfig cfg_;
    SimulationState  state_;
    SimulationClock  clock_;
    Xoshiro256       rng_;

    // Layer engines
    RoughVolEngine      rough_vol_;
    CGMYEngine          levy_;
    RegimeEngine        regime_;
    HawkesEngine        hawkes_;
    LOBEngine           lob_;
    MarketMakerEngine   mm_;
    RuinEngine          ruin_;
    CorrelationEngine   correlation_;
    GraphEngine         graphs_;
    SpectralEngine      spectral_;
    ContagionEngine     contagion_;
    PersistenceEngine   persistence_;
    LandscapeEngine     landscape_;
    TelemetryWriter     telemetry_;

    // Cross-asset Cholesky (updated when correlation changes)
    Eigen::LLT<Eigen::MatrixXd> corr_chol_;

    // Timing
    int topology_interval_ = 50;   ///< Update topology every N steps
    int hjb_interval_ = 100;       ///< Update HJB every N steps

    // Pre-allocated event drain buffer (zero-alloc per tick)
    std::vector<Event> event_buffer_;

    // Async topology/TDA worker with double-buffered snapshot
    std::thread topo_thread_;
    std::mutex topo_mutex_;
    std::condition_variable topo_cv_;
    std::atomic<bool> topo_pending_{false};
    std::atomic<bool> topo_shutdown_{false};
    std::atomic<bool> topo_worker_busy_{false};
    // Double-buffered snapshot: swap instead of deep copy
    SimulationState topo_snapshot_a_, topo_snapshot_b_;
    SimulationState* topo_front_ = &topo_snapshot_a_;
    SimulationState* topo_back_  = &topo_snapshot_b_;
    // Results returned from worker
    struct TopoResult {
        Eigen::MatrixXd correlation;
        double fiedler_value = 0;
        Eigen::VectorXd betweenness;
        double clustering_coeff = 0;
        double tda_risk_index = 0;
        double wasserstein_dist = 0;
        double landscape_l1 = 0;
        double landscape_l2 = 0;
        Eigen::VectorXd eigenvalues;
        Eigen::MatrixXd eigenvectors;
        Eigen::MatrixXd distance;
        bool ready = false;
    };
    TopoResult topo_result_;

    // Pre-allocated scratch for hawkes aggregation
    Eigen::VectorXd hawkes_agg_;

    StepCallback callback_;

public:
    explicit Engine(const SimulationConfig& cfg)
        : cfg_(cfg),
          clock_(cfg.dt, cfg.T),
          rng_(cfg.seed),
          rough_vol_(cfg.rough_vol, cfg.universe.n_assets),
          levy_(cfg.levy, cfg.universe.n_assets),
          regime_(cfg.regime, cfg.universe.n_assets),
          hawkes_(cfg.hawkes, cfg.universe.n_assets),
          lob_(cfg.lob, cfg.universe.n_assets, cfg.universe.initial_prices),
          mm_(cfg.market_maker, cfg.universe.n_assets),
          ruin_(cfg.ruin, cfg.universe.n_assets),
          correlation_(cfg.topology, cfg.universe.n_assets),
          graphs_(cfg.universe.n_assets, cfg.topology.pmfg_enabled),
          spectral_(cfg.universe.n_assets),
          contagion_(cfg.topology, cfg.universe.n_assets),
          persistence_(cfg.tda, cfg.universe.n_assets),
          landscape_(cfg.tda),
          telemetry_("sim_state.json", cfg.universe.n_assets, 10)
    {
        cfg_.finalize();
        state_.init(cfg_);
        corr_chol_.compute(state_.correlation);
        event_buffer_.reserve(4096);
        hawkes_agg_.resize(cfg.universe.n_assets);

        // Kill Eigen internal threading to prevent OMP oversubscription
        Eigen::setNbThreads(1);

        // Init double-buffer snapshots
        topo_snapshot_a_.init(cfg_);
        topo_snapshot_b_.init(cfg_);

        // Fork persistent per-thread RNGs once (not per-tick)
        levy_.init_thread_rngs(rng_);
        lob_.init_thread_rngs(rng_);

        // Launch async topology worker
        topo_thread_ = std::thread([this]() { topology_worker(); });
    }

    ~Engine() {
        topo_shutdown_ = true;
        topo_cv_.notify_one();
        if (topo_thread_.joinable()) topo_thread_.join();
    }

    /// Background thread: runs topology + TDA off the main tick loop
    void topology_worker() {
        while (true) {
            std::unique_lock<std::mutex> lock(topo_mutex_);
            topo_cv_.wait(lock, [this]{ return topo_pending_ || topo_shutdown_; });
            if (topo_shutdown_) return;
            topo_pending_ = false;
            topo_worker_busy_ = true;

            // Work on back-buffer directly (no deep copy)
            SimulationState& snap = *topo_back_;
            lock.unlock();

            // Heavy computation off main thread
            correlation_.update(snap);
            graphs_.update(snap);
            spectral_.update(snap, graphs_);
            persistence_.compute(snap.distance);
            landscape_.update(snap, persistence_);

            // Package results
            std::lock_guard<std::mutex> rlock(topo_mutex_);
            topo_result_.correlation = snap.correlation;
            topo_result_.fiedler_value = snap.fiedler_value;
            topo_result_.betweenness = snap.betweenness;
            topo_result_.clustering_coeff = snap.clustering_coeff;
            topo_result_.tda_risk_index = snap.tda_risk_index;
            topo_result_.wasserstein_dist = snap.wasserstein_dist;
            topo_result_.landscape_l1 = snap.landscape_l1;
            topo_result_.landscape_l2 = snap.landscape_l2;
            topo_result_.eigenvalues = snap.eigenvalues;
            topo_result_.eigenvectors = snap.eigenvectors;
            topo_result_.distance = snap.distance;
            topo_result_.ready = true;
            topo_worker_busy_ = false;
        }
    }

    void set_callback(StepCallback cb) { callback_ = std::move(cb); }

    /// Run the full simulation
    void run() {
        auto wall_start = std::chrono::high_resolution_clock::now();
        int n_steps = cfg_.n_steps();
        int N = cfg_.universe.n_assets;

        std::cout << "═══════════════════════════════════════════════\n"
                  << "  SOVEREIGN Engine v" << "0.1.0" << "\n"
                  << "  Assets: " << N << " | Steps: " << n_steps
                  << " | dt: " << cfg_.dt << "\n"
                  << "  Hawkes branching ratio: "
                  << hawkes_.total_branching_ratio() << "\n"
                  << "═══════════════════════════════════════════════\n";

        for (int s = 0; s < n_steps && !clock_.done(); ++s) {
            state_.step = s;
            state_.t = s * cfg_.dt;

            // ── Layer 1: Price dynamics ─────────────────────────
            Eigen::VectorXd old_prices = state_.prices();
            rough_vol_.step(state_, cfg_.dt, corr_chol_, rng_);
            levy_.step(state_, cfg_.dt, rng_);

            // ── Layer 1.2b: Regime switching ────────────────────
            hawkes_.aggregate_intensities(state_.t, hawkes_agg_);
            regime_.step(state_, cfg_.dt, hawkes_agg_, rng_);

            // ── Layer 2: Hawkes event generation ────────────────
            hawkes_.generate_events(clock_, state_, cfg_.dt, rng_);

            // ── Layer 3: LOB processing ─────────────────────
            clock_.drain_events(event_buffer_);
            lob_.process_events(event_buffer_, state_, rng_);
            lob_.step(state_, cfg_.dt, rng_);
            lob_.compute_impact(state_);

            // Wire LOB impact into price dynamics and unify returns
            for (int i = 0; i < N; ++i) {
                // LOB impact is a RATE — scale by dt for time-consistency.
                // Without this, impact accumulates 50000x faster than intended.
                state_.assets[i].log_price += state_.assets[i].lob_impact * cfg_.dt;
                
                // CRITICAL FIX: Clamp log_price to prevent std::exp yielding INFINITY,
                // which crashes JSON serialization.
                state_.assets[i].log_price = std::clamp(state_.assets[i].log_price, -10.0, 15.0);
                
                state_.assets[i].price = std::exp(state_.assets[i].log_price);
                
                // LOG returns: scale-invariant for covariance estimation
                double old_p = old_prices(i);
                double new_p = state_.assets[i].price;
                double ret = std::log(new_p / std::max(old_p, 1e-10));
                state_.assets[i].return_1 = ret;
                state_.assets[i].cum_return += ret;
            }

            // ── Layer 4: Market maker + HJB ─────────────────
            mm_.step(state_, cfg_.dt, rng_);
            if (s % hjb_interval_ == 0 && s > 0) {
                mm_.periodic_hjb(state_, cfg_.dt);
            }

            // ── Layer 5: Ruin dynamics ──────────────────────
            ruin_.step(state_, cfg_.dt, rng_);

            // Update ruin_vector with local ruin_prob before contagion
            for (int i = 0; i < N; ++i) {
                state_.ruin_vector(i) = std::max(state_.ruin_vector(i), state_.assets[i].ruin_prob);
            }

            // ── Layer 6: Topology (async dispatch) ────────────
            correlation_.record(state_);

            if (s % topology_interval_ == 0 && s > 0) {
                bool dispatched = false;
                {
                    std::lock_guard<std::mutex> lock(topo_mutex_);
                    if (!topo_worker_busy_) {
                        // Copy into front buffer and swap — all under the lock
                        // to eliminate the race between busy-check and dispatch.
                        // For N=50, this copies ~20KB of matrices — negligible.
                        topo_front_->correlation = state_.correlation;
                        topo_front_->distance = state_.distance;
                        topo_front_->ruin_vector = state_.ruin_vector;
                        topo_front_->eigenvalues = state_.eigenvalues;
                        topo_front_->eigenvectors = state_.eigenvectors;
                        std::swap(topo_front_, topo_back_);
                        topo_pending_ = true;
                        dispatched = true;
                    }
                }
                if (dispatched) topo_cv_.notify_one();
            }

            // ── Callback / Telemetry (BEFORE topo ingestion for consistency) ──
            if (s % 10 == 0) telemetry_.write(state_);
            if (callback_ && s % 100 == 0) {
                callback_(state_, s);
            }

            // Ingest results from background worker (non-blocking)
            {
                std::lock_guard<std::mutex> lock(topo_mutex_);
                if (topo_result_.ready) {
                    state_.correlation = topo_result_.correlation;
                    state_.fiedler_value = topo_result_.fiedler_value;
                    state_.betweenness = topo_result_.betweenness;
                    state_.clustering_coeff = topo_result_.clustering_coeff;
                    state_.tda_risk_index = topo_result_.tda_risk_index;
                    state_.wasserstein_dist = topo_result_.wasserstein_dist;
                    state_.landscape_l1 = topo_result_.landscape_l1;
                    state_.landscape_l2 = topo_result_.landscape_l2;
                    state_.eigenvalues = topo_result_.eigenvalues;
                    state_.eigenvectors = topo_result_.eigenvectors;
                    state_.distance = topo_result_.distance;
                    corr_chol_.compute(state_.correlation);
                    hawkes_.modulate_by_graph(graphs_.graph_distances());
                    contagion_.invalidate_cache();
                    topo_result_.ready = false;
                }
            }

            // ── Layer 6.4: Contagion diffusion (every step) ─────
            contagion_.step(state_, cfg_.dt);

            // ── Feedback loops ──────────────────────────────────
            // Gamma -> Hawkes: mild boost to event intensity when ruin elevated
            // (ruin=1 -> 1.4x baseline, not 2.4x — prevents runaway cascade)
            for (int i = 0; i < N; ++i) {
                double ruin_boost = 1.0 + 0.5 * std::max(state_.assets[i].ruin_prob - 0.5, 0.0);
                hawkes_.set_baseline_modulation(i, ruin_boost);
                // DO NOT force regime — regime switching is purely stochastic
                // via the Hawkes intensity in regime_.step(). Forcing regime
                // based on ruin_prob inverts the feedback: low regime -> less
                // jumps -> LESS mean-reversion -> price explosion -> more ruin.
            }

            clock_.tick();

            // Progress
            if (s % (n_steps / 20 + 1) == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - wall_start).count();
                double pct = 100.0 * s / n_steps;
                std::cout << "\r  [" << std::fixed;
                std::cout.precision(1);
                std::cout << pct << "%] step=" << s
                          << " t=" << state_.t
                          << " events=" << state_.total_events
                          << " fiedler=" << state_.fiedler_value
                          << " TRI=" << state_.tda_risk_index
                          << " elapsed=" << elapsed << "s"
                          << std::flush;
            }
        }

        auto wall_end = std::chrono::high_resolution_clock::now();
        state_.wall_clock_s = std::chrono::duration<double>(wall_end - wall_start).count();

        std::cout << "\n═══════════════════════════════════════════════\n"
                  << "  Simulation complete: " << state_.wall_clock_s << "s\n"
                  << "  Total events: " << state_.total_events << "\n"
                  << "  Final TRI: " << state_.tda_risk_index << "\n"
                  << "  Final Fiedler: " << state_.fiedler_value << "\n"
                  << "═══════════════════════════════════════════════\n";
    }

    const SimulationState& state() const { return state_; }
    SimulationState& state() { return state_; }
    const SimulationConfig& config() const { return cfg_; }
};

} // namespace sovereign
