/// @file config.cpp
/// @brief SimulationConfig JSON I/O implementation.

#include <sovereign/config.hpp>
#include <fstream>
#include <iostream>

namespace sovereign {

SimulationConfig SimulationConfig::from_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Warning: config file '" << path << "' not found, using defaults.\n";
        SimulationConfig cfg;
        cfg.finalize();
        return cfg;
    }
    nlohmann::json j;
    try {
        f >> j;
        SimulationConfig cfg = j.get<SimulationConfig>();
        cfg.finalize();
        return cfg;
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "Fatal: JSON parse error in '" << path << "': " << e.what() << "\n";
        std::cerr << "Falling back to defaults.\n";
        SimulationConfig cfg;
        cfg.finalize();
        return cfg;
    }
}

void SimulationConfig::to_json(const std::string& path) const {
    nlohmann::json j = *this;
    std::ofstream f(path);
    f << j.dump(2);
}

} // namespace sovereign
