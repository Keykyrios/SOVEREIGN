#pragma once
/// @file telemetry.hpp
/// @brief Writes full simulation state snapshots to JSON for the dashboard.
/// All data comes directly from SimulationState — nothing synthesized.

#include <sovereign/core/state.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <cstdio>

namespace sovereign {

class TelemetryWriter {
    std::string path_;
    int flush_interval_;
    int n_assets_;
    int lob_export_levels_;  ///< How many LOB levels to export (keep manageable)

public:
    TelemetryWriter(const std::string& path, int n_assets,
                    int flush_interval = 10, int lob_levels = 20)
        : path_(path), flush_interval_(flush_interval),
          n_assets_(n_assets), lob_export_levels_(lob_levels)
    {
        std::ofstream f(path_);
        f << "{}";
    }

    void write(const SimulationState& state) {
        if (state.step % flush_interval_ != 0) return;

        nlohmann::json snap;

        // ── Simulation clock ─────────────────────────────────────────────
        snap["step"]         = state.step;
        snap["t"]            = state.t;
        snap["total_events"] = state.total_events;
        snap["wall_clock_s"] = state.wall_clock_s;

        // ── Per-asset state (full) ────────────────────────────────────────
        for (int i = 0; i < n_assets_; ++i) {
            const auto& a = state.assets[i];

            auto& ai = snap["assets"][i];

            // Layer 1: Price process
            ai["id"]        = a.id;
            ai["price"]     = a.price;
            ai["log_price"] = a.log_price;
            ai["vol"]       = a.volatility;
            ai["variance"]  = a.variance;
            ai["hurst"]     = a.hurst;
            ai["jump"]      = a.jump_component;
            ai["lob_impact"]= a.lob_impact;
            ai["regime"]    = a.regime;
            ai["return1"]   = a.return_1;
            ai["cum_ret"]   = a.cum_return;

            // Layer 2: Hawkes intensities (all K*D values)
            auto& hi = ai["hawkes_intensity"];
            for (int k = 0; k < (int)a.hawkes_intensity.size(); ++k)
                hi[k] = a.hawkes_intensity(k);

            // Layer 3: LOB (top lob_export_levels_ levels)
            int L = std::min(lob_export_levels_,
                             (int)a.lob.bid_volumes.size());
            auto& lob = ai["lob"];
            for (int d = 0; d < L; ++d) {
                lob["bid_vol"][d]   = a.lob.bid_volumes(d);
                lob["ask_vol"][d]   = a.lob.ask_volumes(d);
                lob["bid_price"][d] = a.lob.bid_prices(d);
                lob["ask_price"][d] = a.lob.ask_prices(d);
            }
            lob["best_bid"]  = a.lob.best_bid;
            lob["best_ask"]  = a.lob.best_ask;
            lob["mid_price"] = a.lob.mid_price;
            lob["spread"]    = a.lob.spread;
            lob["imbalance"] = a.lob.imbalance();
            lob["microprice"]= a.lob.microprice();

            // Layer 4: Market maker
            ai["mm_spread"]    = a.mm_spread;
            ai["mm_inventory"] = a.mm_inventory;
            ai["mm_value"]     = a.mm_value;

            // Layer 5: Ruin
            ai["surplus"]    = a.surplus;
            ai["ruin"]       = a.ruin_prob;
            ai["gerber_shiu"]= a.gerber_shiu;
        }

        // ── Cross-asset: Layer 6 ─────────────────────────────────────────
        // Full correlation matrix (flattened row-major)
        {
            auto& c = snap["correlation"];
            int idx = 0;
            for (int i = 0; i < n_assets_; ++i)
                for (int j = 0; j < n_assets_; ++j)
                    c[idx++] = state.correlation(i, j);
        }
        {
            auto& c = snap["raw_correlation"];
            int idx = 0;
            for (int i = 0; i < n_assets_; ++i)
                for (int j = 0; j < n_assets_; ++j)
                    c[idx++] = state.raw_correlation(i, j);
        }
        // Eigenvalues
        for (int i = 0; i < n_assets_; ++i)
            snap["eigenvalues"][i] = state.eigenvalues(i);

        // Graph metrics
        snap["fiedler"]    = state.fiedler_value;
        snap["clustering"] = state.clustering_coeff;
        for (int i = 0; i < n_assets_; ++i) {
            snap["betweenness"][i] = state.betweenness(i);
            snap["degree"][i]      = state.degree(i);
        }

        // ── Cross-asset: Layer 7 TDA ─────────────────────────────────────
        snap["tri"]         = state.tda_risk_index;
        snap["wasserstein"] = state.wasserstein_dist;
        snap["l1"]          = state.landscape_l1;
        snap["l2"]          = state.landscape_l2;

        // ── Ruin vector ──────────────────────────────────────────────────
        for (int i = 0; i < n_assets_; ++i)
            snap["ruin_vector"][i] = state.ruin_vector(i);

        // ── Distance matrix ──────────────────────────────────────────────
        {
            auto& d = snap["distance"];
            int idx = 0;
            for (int i = 0; i < n_assets_; ++i)
                for (int j = 0; j < n_assets_; ++j)
                    d[idx++] = state.distance(i, j);
        }

        // Direct write — simple and works on all platforms
        std::ofstream f(path_, std::ios::trunc);
        if (f.is_open()) {
            f << snap.dump(-1);
        }
    }
};

} // namespace sovereign
