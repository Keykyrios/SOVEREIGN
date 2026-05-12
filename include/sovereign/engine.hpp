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
            rough_vol_.step(state_, cfg_.dt, corr_chol_, rng_);
            levy_.step(state_, cfg_.dt, rng_);

            // ── Layer 1.2b: Regime switching ────────────────────
            Eigen::VectorXd hawkes_agg = hawkes_.aggregate_intensities(state_.t);
            regime_.step(state_, cfg_.dt, hawkes_agg, rng_);

            // ── Layer 2: Hawkes event generation ────────────────
            hawkes_.generate_events(clock_, state_, cfg_.dt, rng_);

            // ── Layer 3: LOB processing ─────────────────────────
            auto events = clock_.drain_events();
            lob_.process_events(events, state_, rng_);
            lob_.step(state_, cfg_.dt, rng_);
            lob_.compute_impact(state_);

            // ── Layer 4: Market maker ───────────────────────────
            mm_.step(state_, cfg_.dt, rng_);

            // ── Layer 5: Ruin dynamics ──────────────────────────
            ruin_.step(state_, cfg_.dt, rng_);

            // ── Layer 6: Topology (periodic) ────────────────────
            correlation_.record(state_);

            if (s % topology_interval_ == 0 && s > 0) {
                correlation_.update(state_);
                corr_chol_.compute(state_.correlation);

                graphs_.update(state_);
                spectral_.update(state_, graphs_);

                // Modulate Hawkes by graph distance
                hawkes_.modulate_by_graph(graphs_.graph_distances());
            }

            // ── Layer 6.4: Contagion diffusion (every step) ─────
            contagion_.step(state_, cfg_.dt);

            // ── Layer 7: TDA (periodic, expensive) ──────────────
            if (s % (topology_interval_ * 5) == 0 && s > 0) {
                persistence_.compute(state_.distance);
                landscape_.update(state_, persistence_);
            }

            // ── Feedback loops ──────────────────────────────────
            // Γ → Layer 2: boost Hawkes when ruin is high
            // (handled inside hawkes via aggregate_intensities modulation)
            // Γ → Layer 4: spread widening (inside MM robust_spread)
            // Γ → Layer 1: regime switch trigger
            for (int i = 0; i < N; ++i) {
                if (state_.assets[i].ruin_prob > 0.8 && state_.assets[i].regime > 0) {
                    state_.assets[i].regime = std::max(0, state_.assets[i].regime - 1);
                }
            }

            // ── Callback / Telemetry ─────────────────────────────
            telemetry_.write(state_);
            if (callback_ && s % 100 == 0) {
                callback_(state_, s);
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
