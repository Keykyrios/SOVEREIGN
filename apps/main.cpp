/// @file main.cpp
/// @brief SOVEREIGN entry point — parse config, run simulation, report.

#include <sovereign/engine.hpp>
#include <iostream>
#include <string>
#include <cstring>

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
    sovereign::SimulationConfig cfg;

    // Parse CLI
    std::string config_path;
    std::string save_config_path;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (std::strcmp(argv[i], "--assets") == 0 && i + 1 < argc) {
            cfg.universe.n_assets = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            int steps = std::stoi(argv[++i]);
            cfg.dt = cfg.T / steps;
        } else if (std::strcmp(argv[i], "--dt") == 0 && i + 1 < argc) {
            cfg.dt = std::stod(argv[++i]);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg.seed = std::stoull(argv[++i]);
        } else if (std::strcmp(argv[i], "--save-config") == 0 && i + 1 < argc) {
            save_config_path = argv[++i];
        } else if (std::strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        }
    }

    // Load config from file if specified
    if (!config_path.empty()) {
        cfg = sovereign::SimulationConfig::from_json(config_path);
    }
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
    int N = cfg.universe.n_assets;

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
