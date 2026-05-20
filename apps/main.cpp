/// @file main.cpp
/// @brief SOVEREIGN entry point — parse config, run simulation, report.

#include <sovereign/engine.hpp>
#include <iostream>
#include <string>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

void print_usage() {
    std::cout << "Usage: sovereign [OPTIONS]\n"
              << "  --config <path>   Load config from JSON file\n"
              << "  --assets <N>      Number of assets (default: 50)\n"
              << "  --steps <N>       Override number of timesteps\n"
              << "  --dt <val>        Timestep size (default: 1e-4)\n"
              << "  --seed <val>      RNG seed (default: 42)\n"
              << "  --save-config <p> Save effective config to file\n"
              << "  --help            Show this message\n";
}

int main(int argc, char* argv[]) {
#if defined(__x86_64__) || defined(_M_X64)
    // Enable FTZ (Flush-To-Zero) and DAZ (Denormals-Are-Zero) to prevent
    // 100x latency stalls when Hawkes intensities decay to subnormal floats.
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
    
    sovereign::SimulationConfig cfg;

    // Parse CLI into temporaries — do NOT apply yet, config file loads first
    std::string config_path;
    std::string save_config_path;
    int cli_assets = -1;
    int cli_steps = -1;
    double cli_dt = -1;
    uint64_t cli_seed = 0;
    bool cli_seed_set = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (std::strcmp(argv[i], "--assets") == 0 && i + 1 < argc) {
            cli_assets = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            cli_steps = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--dt") == 0 && i + 1 < argc) {
            cli_dt = std::stod(argv[++i]);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cli_seed = std::stoull(argv[++i]);
            cli_seed_set = true;
        } else if (std::strcmp(argv[i], "--save-config") == 0 && i + 1 < argc) {
            save_config_path = argv[++i];
        } else if (std::strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        }
    }

    // Step 1: Load config from file FIRST (establishes baseline T, dt, etc.)
    if (!config_path.empty()) {
        cfg = sovereign::SimulationConfig::from_json(config_path);
    }

    // Step 2: Apply CLI overrides AFTER config load so they always win
    if (cli_assets > 0) cfg.universe.n_assets = cli_assets;
    if (cli_steps > 0)  cfg.dt = cfg.T / cli_steps;
    if (cli_dt > 0)     cfg.dt = cli_dt;
    if (cli_seed_set)   cfg.seed = cli_seed;

    cfg.finalize();

    // Save effective config if requested
    if (!save_config_path.empty()) {
        cfg.to_json(save_config_path);
        std::cout << "Config saved to: " << save_config_path << "\n";
    }

    std::cout << R"(
 ╔═══════════════════════════════════════════════════════════════╗
 ║  ███████╗ ██████╗ ██╗   ██╗███████╗██████╗ ███████╗██╗ ██████╗ ███╗   ██╗ ║
 ║  ██╔════╝██╔═══██╗██║   ██║██╔════╝██╔══██╗██╔════╝██║██╔════╝ ████╗  ██║ ║
 ║  ███████╗██║   ██║██║   ██║█████╗  ██████╔╝█████╗  ██║██║  ███╗██╔██╗ ██║ ║
 ║  ╚════██║██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗██╔══╝  ██║██║   ██║██║╚██╗██║ ║
 ║  ███████║╚██████╔╝ ╚████╔╝ ███████╗██║  ██║███████╗██║╚██████╔╝██║ ╚████║ ║
 ║  ╚══════╝ ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ║
 ║                                                                             ║
 ║  Stochastic Order-driven Volatility Engine with                             ║
 ║  Recursive Endogenous Instability, Generated Numerically                    ║
 ╚═══════════════════════════════════════════════════════════════╝
)" << "\n";

    // Set up step callback for monitoring
    auto callback = [](const sovereign::SimulationState& state, int step) {
        // Could write to file, send to dashboard, etc.
        if (state.fiedler_value < 0.1) {
            std::cout << "\n  ⚠ FRAGMENTATION WARNING at step " << step
                      << ": Fiedler=" << state.fiedler_value << "\n";
        }
        if (state.tda_risk_index > 10.0) {
            std::cout << "\n  ⚠ TRI SPIKE at step " << step
                      << ": TRI=" << state.tda_risk_index << "\n";
        }
        double max_ruin = state.ruin_vector.maxCoeff();
        if (max_ruin > 0.5) {
            std::cout << "\n  ⚠ RUIN ALERT at step " << step
                      << ": max(Γ)=" << max_ruin << "\n";
        }
    };

    // Build and run
    sovereign::Engine engine(cfg);
    engine.set_callback(callback);
    engine.run();

    // Final report
    const auto& state = engine.state();

    std::cout << "\n── Final State Summary ──────────────────────\n";

    // Price statistics
    auto prices = state.prices();
    double mean_price = prices.mean();
    double min_price = prices.minCoeff();
    double max_price = prices.maxCoeff();
    std::cout << "  Prices: mean=" << mean_price
              << " min=" << min_price
              << " max=" << max_price << "\n";

    // Ruin statistics
    double mean_ruin = state.ruin_vector.mean();
    double max_ruin = state.ruin_vector.maxCoeff();
    std::cout << "  Ruin: mean(Γ)=" << mean_ruin
              << " max(Γ)=" << max_ruin << "\n";

    // Topology
    std::cout << "  Topology: Fiedler=" << state.fiedler_value
              << " clustering=" << state.clustering_coeff << "\n";

    // TDA
    std::cout << "  TDA: TRI=" << state.tda_risk_index
              << " W₂=" << state.wasserstein_dist
              << " ||λ||₁=" << state.landscape_l1
              << " ||λ||₂=" << state.landscape_l2 << "\n";

    std::cout << "────────────────────────────────────────────\n";

    return 0;
}
