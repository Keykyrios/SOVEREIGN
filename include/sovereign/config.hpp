#pragma once
/// @file config.hpp
/// @brief SOVEREIGN global configuration — every simulation parameter.
///
/// All magic numbers from the specification live here. The SimulationConfig
/// struct is read from JSON at startup and is immutable thereafter.

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace sovereign {

// ─── Layer 0: Universe ──────────────────────────────────────────────────

struct UniverseConfig {
    int n_assets = 50;                       ///< N ∈ [50, 200]
    std::vector<std::string> asset_names;    ///< Auto-generated if empty
    std::vector<double> initial_prices;      ///< Per-asset initial mid-price
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(UniverseConfig,
    n_assets, asset_names, initial_prices)

// ─── Layer 1.1: Rough Bergomi ───────────────────────────────────────────

struct RoughVolConfig {
    double hurst          = 0.10;   ///< H ≈ 0.1 (rough regime)
    double hurst_mean     = 0.10;   ///< H̄ for stochastic Hurst OU
    double hurst_kappa    = 2.0;    ///< κ_H mean-reversion speed
    double hurst_sigma    = 0.50;   ///< σ_H diffusion of H(t) (increased for visual dynamics)
    double eta            = 0.35;   ///< Vol-of-vol (rBergomi, calibrated range 0.2–0.5)
    double rho            = -0.90;  ///< Spot-vol correlation
    double xi_0           = 0.04;   ///< Forward variance ξ₀(t)
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(RoughVolConfig,
    hurst, hurst_mean, hurst_kappa, hurst_sigma, eta, rho, xi_0)

// ─── Layer 1.2: CGMY Lévy Jumps ────────────────────────────────────────

struct LevyConfig {
    double C      = 1.0;    ///< Activity rate
    double G      = 5.0;    ///< Negative jump decay
    double M      = 10.0;   ///< Positive jump decay
    double Y      = 1.5;    ///< Fine structure Y ∈ (1,2) → infinite variation
    double cir_kappa  = 1.0;    ///< CIR subordinator mean-reversion
    double cir_eta    = 1.0;    ///< CIR long-run mean
    double cir_lambda = 0.5;    ///< CIR vol-of-vol
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(LevyConfig,
    C, G, M, Y, cir_kappa, cir_eta, cir_lambda)

// ─── Layer 1.2b: Regime ─────────────────────────────────────────────────

struct RegimeConfig {
    int n_regimes     = 5;    ///< K = 5 hidden states
    int initial_regime = 2;   ///< Start in "normal"
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(RegimeConfig,
    n_regimes, initial_regime)

// ─── Layer 2: Hawkes ────────────────────────────────────────────────────

struct HawkesConfig {
    int    n_order_types       = 5;     ///< market, limit, cancel, modify, hidden
    int    n_depth_levels      = 10;    ///< Depth levels tracked
    double base_intensity      = 10.0;  ///< μ: baseline arrival
    double alpha_self          = 0.5;   ///< Self-excitation
    double alpha_cross         = 0.1;   ///< Cross-asset excitation
    double beta                = 1.0;   ///< Kernel time scale
    double epsilon             = 0.5;   ///< Power-law tail ∈ (0,1)
    double max_spectral_radius = 0.95;  ///< ρ(branching) < 1
    int    dykstra_max_iter    = 100;   ///< Projection iterations
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(HawkesConfig,
    n_order_types, n_depth_levels, base_intensity, alpha_self, alpha_cross,
    beta, epsilon, max_spectral_radius, dykstra_max_iter)

// ─── Layer 3: LOB ──────────────────────────────────────────────────────

struct LOBConfig {
    int    n_levels            = 500;    ///< M levels per side
    double tick_size           = 0.01;
    double initial_depth       = 100.0;
    double cancel_rate_base    = 0.1;
    double cancel_distance_decay = 0.5;
    double iceberg_probability = 0.05;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(LOBConfig,
    n_levels, tick_size, initial_depth, cancel_rate_base,
    cancel_distance_decay, iceberg_probability)

// ─── Layer 4: Market Maker ──────────────────────────────────────────────

struct MarketMakerConfig {
    int    n_market_makers  = 5;
    double gamma            = 2.0;      ///< CRRA risk aversion
    double theta            = 0.5;      ///< Ambiguity aversion (κ-ignorance) — base value
    double tri_alpha        = 0.1;      ///< TRI→θ coupling: θ(t) = θ₀·exp(α·TRI(t))
    double inventory_limit  = 1000.0;
    int    policy_iter_max  = 100;      ///< Howard's iteration cap
    int    hjb_grid_points  = 200;      ///< Per dimension
    double spread_floor     = 0.01;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(MarketMakerConfig,
    n_market_makers, gamma, theta, tri_alpha, inventory_limit,
    policy_iter_max, hjb_grid_points, spread_floor)

// ─── Layer 5: Ruin ─────────────────────────────────────────────────────

struct RuinConfig {
    double initial_surplus          = 1000.0;
    double premium_rate             = 1.0;
    double ruin_threshold           = 0.0;
    double delta_discount           = 0.05;
    int    restart_levels           = 5;
    double restart_threshold_factor = 0.5;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(RuinConfig,
    initial_surplus, premium_rate, ruin_threshold,
    delta_discount, restart_levels, restart_threshold_factor)

// ─── Layer 6: Topology ─────────────────────────────────────────────────

struct TopologyConfig {
    double ewma_alpha           = 0.005;  ///< Effective memory ~400 ticks (balanced)
    double rmt_ratio            = 2.0;   ///< Q = T/N for Marčenko-Pastur
    int    spd_max_iter         = 50;
    bool   pmfg_enabled         = true;
    double contagion_diffusivity = 0.1;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(TopologyConfig,
    ewma_alpha, rmt_ratio, spd_max_iter, pmfg_enabled, contagion_diffusivity)

// ─── Layer 7: TDA ──────────────────────────────────────────────────────

struct TDAConfig {
    int    max_dimension        = 2;     ///< H₀, H₁, H₂
    double max_filtration       = 2.0;   ///< ε_max for VR complex
    int    sliding_window       = 50;
    int    landscape_resolution = 200;
    double tri_weight_power     = 2.0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(TDAConfig,
    max_dimension, max_filtration, sliding_window,
    landscape_resolution, tri_weight_power)

// ─── Monte Carlo ────────────────────────────────────────────────────────

struct MLMCConfig {
    int    n_levels         = 5;
    int    base_samples     = 10000;
    int    geometric_factor = 4;       ///< M: timestep refinement
    double target_rmse      = 1e-3;
    double confidence       = 0.95;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(MLMCConfig,
    n_levels, base_samples, geometric_factor, target_rmse, confidence)

// ─── Master Config ──────────────────────────────────────────────────────

struct SimulationConfig {
    // Time
    double   T      = 1.0;      ///< Simulation horizon (years)
    double   dt     = 1e-4;     ///< Finest timestep
    uint64_t seed   = 42;

    // Sub-configs
    UniverseConfig    universe;
    RoughVolConfig    rough_vol;
    LevyConfig        levy;
    RegimeConfig      regime;
    HawkesConfig      hawkes;
    LOBConfig         lob;
    MarketMakerConfig market_maker;
    RuinConfig        ruin;
    TopologyConfig    topology;
    TDAConfig         tda;
    MLMCConfig        mlmc;

    /// Derived quantities (computed after load)
    int n_steps() const { return static_cast<int>(T / dt); }

    /// Auto-fill empty fields after deserialization
    void finalize() {
        if (universe.asset_names.empty()) {
            universe.asset_names.resize(universe.n_assets);
            for (int i = 0; i < universe.n_assets; ++i) {
                char buf[16];
                std::snprintf(buf, sizeof(buf), "ASSET_%03d", i);
                universe.asset_names[i] = buf;
            }
        }
        if (universe.initial_prices.empty()) {
            universe.initial_prices.assign(universe.n_assets, 100.0);
        }
    }

    /// Load from JSON file
    static SimulationConfig from_json(const std::string& path);

    /// Save to JSON file
    void to_json(const std::string& path) const;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(SimulationConfig,
    T, dt, seed, universe, rough_vol, levy, regime, hawkes, lob,
    market_maker, ruin, topology, tda, mlmc)

} // namespace sovereign
