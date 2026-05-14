#pragma once
/// @file telemetry.hpp
/// @brief Writes full simulation state snapshots to JSON for the dashboard.
/// All data comes directly from SimulationState — nothing synthesized.

#include <sovereign/core/state.hpp>
#include <boost/asio.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>

namespace sovereign {

class TelemetryWriter {
    int flush_interval_;
    int n_assets_;
    int lob_export_levels_;
    std::future<void> write_future_;
    boost::asio::io_context io_context_;
    boost::asio::ip::udp::socket socket_;
    boost::asio::ip::udp::endpoint endpoint_;

public:
    TelemetryWriter(const std::string& path, int n_assets,
                    int flush_interval = 10, int lob_levels = 20)
        : flush_interval_(flush_interval),
          n_assets_(n_assets), lob_export_levels_(lob_levels),
          socket_(io_context_, boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0)),
          endpoint_(boost::asio::ip::address::from_string("127.0.0.1"), 8080)
    {
    }

    ~TelemetryWriter() {
        if (write_future_.valid()) write_future_.wait();
    }

    void write(const SimulationState& state) {
        if (state.step % flush_interval_ != 0) return;

        if (write_future_.valid() && write_future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            return;
        }

        // Direct string construction to avoid nlohmann::json AST overhead
        std::ostringstream ss;
        ss.precision(6);
        ss << std::fixed;

        ss << "{";
        ss << "\"step\":" << state.step << ",";
        ss << "\"t\":" << state.t << ",";
        ss << "\"total_events\":" << state.total_events << ",";
        ss << "\"wall_clock_s\":" << state.wall_clock_s << ",";

        ss << "\"assets\":[";
        for (int i = 0; i < n_assets_; ++i) {
            const auto& a = state.assets[i];
            ss << "{";
            ss << "\"id\":" << a.id << ",";
            ss << "\"price\":" << a.price << ",";
            ss << "\"log_price\":" << a.log_price << ",";
            ss << "\"vol\":" << a.volatility << ",";
            ss << "\"variance\":" << a.variance << ",";
            ss << "\"hurst\":" << a.hurst << ",";
            ss << "\"jump\":" << a.jump_component << ",";
            ss << "\"lob_impact\":" << a.lob_impact << ",";
            ss << "\"regime\":" << a.regime << ",";
            ss << "\"return1\":" << a.return_1 << ",";
            ss << "\"cum_ret\":" << a.cum_return << ",";

            ss << "\"hawkes_intensity\":[";
            for (int k = 0; k < (int)a.hawkes_intensity.size(); ++k) {
                ss << a.hawkes_intensity(k) << (k == a.hawkes_intensity.size()-1 ? "" : ",");
            }
            ss << "],";

            int L = std::min(lob_export_levels_, (int)a.lob.bid_volumes.size());
            ss << "\"lob\":{";
            auto print_vec = [&](const std::string& name, const auto& v) {
                ss << "\"" << name << "\":[";
                for (int d = 0; d < L; ++d) ss << v(d) << (d == L-1 ? "" : ",");
                ss << "],";
            };
            print_vec("bid_vol", a.lob.bid_volumes);
            print_vec("ask_vol", a.lob.ask_volumes);
            print_vec("bid_price", a.lob.bid_prices);
            print_vec("ask_price", a.lob.ask_prices);
            
            ss << "\"best_bid\":" << a.lob.best_bid << ",";
            ss << "\"best_ask\":" << a.lob.best_ask << ",";
            ss << "\"mid_price\":" << a.lob.mid_price << ",";
            ss << "\"spread\":" << a.lob.spread << ",";
            ss << "\"imbalance\":" << a.lob.imbalance() << ",";
            ss << "\"microprice\":" << a.lob.microprice();
            ss << "},";

            ss << "\"mm_spread\":" << a.mm_spread << ",";
            ss << "\"mm_inventory\":" << a.mm_inventory << ",";
            ss << "\"mm_value\":" << a.mm_value << ",";
            ss << "\"surplus\":" << a.surplus << ",";
            ss << "\"ruin\":" << a.ruin_prob << ",";
            ss << "\"gerber_shiu\":" << a.gerber_shiu;
            ss << "}" << (i == n_assets_-1 ? "" : ",");
        }
        ss << "],";

        auto print_mat = [&](const std::string& name, const Eigen::MatrixXd& m) {
            ss << "\"" << name << "\":[";
            int idx = 0; int total = m.rows() * m.cols();
            for (int i = 0; i < m.rows(); ++i) {
                for (int j = 0; j < m.cols(); ++j) {
                    ss << m(i, j) << (++idx == total ? "" : ",");
                }
            }
            ss << "],";
        };
        auto print_vec_field = [&](const std::string& name, const Eigen::VectorXd& v) {
            ss << "\"" << name << "\":[";
            for (int i = 0; i < v.size(); ++i) ss << v(i) << (i == v.size()-1 ? "" : ",");
            ss << "],";
        };

        print_mat("correlation", state.correlation);
        print_mat("raw_correlation", state.raw_correlation);
        print_vec_field("eigenvalues", state.eigenvalues);

        ss << "\"fiedler\":" << state.fiedler_value << ",";
        ss << "\"clustering\":" << state.clustering_coeff << ",";
        print_vec_field("betweenness", state.betweenness);
        print_vec_field("degree", state.degree);

        ss << "\"tri\":" << state.tda_risk_index << ",";
        ss << "\"wasserstein\":" << state.wasserstein_dist << ",";
        ss << "\"l1\":" << state.landscape_l1 << ",";
        ss << "\"l2\":" << state.landscape_l2 << ",";
        print_vec_field("ruin_vector", state.ruin_vector);
        print_mat("distance", state.distance);
        
        // Remove trailing comma from last print_mat
        ss.seekp(-1, std::ios_base::end);
        ss << "}";

        // Send over UDP asynchronously
        std::string payload = ss.str();
        write_future_ = std::async(std::launch::async, [this, payload]() {
            try {
                socket_.send_to(boost::asio::buffer(payload), endpoint_);
            } catch (...) {}
        });
    }
};

} // namespace sovereign
