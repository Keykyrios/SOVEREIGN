#pragma once
/// @file telemetry.hpp
/// @brief Writes full simulation state snapshots as length-prefixed JSON
///        over a persistent TCP stream to the dashboard.
///
/// Protocol: [4-byte little-endian length][JSON payload]
/// The dashboard (Python) acts as TCP server on 127.0.0.1:8080.
/// The engine connects as a client and streams frames.
/// If the dashboard isn't running, writes are silently skipped.

#include <sovereign/core/state.hpp>
#include <boost/asio.hpp>
#include <sstream>
#include <string>
#include <future>
#include <chrono>
#include <cstdint>

namespace sovereign {

class TelemetryWriter {
    int flush_interval_;
    int n_assets_;
    int lob_export_levels_;
    std::future<void> write_future_;

    // TCP connection state — only accessed from the async write thread
    // (serialized by the future gate: main thread waits for previous future
    // before launching a new one, so no concurrent socket access).
    boost::asio::io_context io_context_;
    boost::asio::ip::tcp::socket socket_;
    bool connected_ = false;

    void ensure_connected() {
        if (connected_) return;
        try {
            if (socket_.is_open()) {
                boost::system::error_code ec;
                socket_.close(ec);
            }
            socket_ = boost::asio::ip::tcp::socket(io_context_);
            boost::asio::ip::tcp::endpoint ep(
                boost::asio::ip::address::from_string("127.0.0.1"), 8080);
            socket_.connect(ep);
            connected_ = true;
        } catch (...) {
            connected_ = false;
        }
    }

    void send_frame(const std::string& payload) {
        try {
            uint32_t len = static_cast<uint32_t>(payload.size());
            boost::asio::write(socket_, boost::asio::buffer(&len, sizeof(len)));
            boost::asio::write(socket_, boost::asio::buffer(payload));
        } catch (...) {
            connected_ = false;
            boost::system::error_code ec;
            socket_.close(ec);
        }
    }

public:
    TelemetryWriter(const std::string& /*path*/, int n_assets,
                    int flush_interval = 10, int lob_levels = 20)
        : flush_interval_(flush_interval),
          n_assets_(n_assets), lob_export_levels_(lob_levels),
          socket_(io_context_)
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

        // Direct string construction — no nlohmann::json AST overhead
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
                ss << a.hawkes_intensity(k);
                if (k < (int)a.hawkes_intensity.size() - 1) ss << ",";
            }
            ss << "],";

            int L = std::min(lob_export_levels_, (int)a.lob.bid_volumes.size());
            ss << "\"lob\":{";
            // Generic vector printer — works for both VectorXi and VectorXd
            auto print_lob_vec = [&](const char* name, const auto& v, bool last = false) {
                ss << "\"" << name << "\":[";
                for (int d = 0; d < L; ++d) {
                    ss << v(d);
                    if (d < L - 1) ss << ",";
                }
                ss << "]";
                if (!last) ss << ",";
            };
            print_lob_vec("bid_vol", a.lob.bid_volumes);
            print_lob_vec("ask_vol", a.lob.ask_volumes);
            print_lob_vec("bid_price", a.lob.bid_prices);
            print_lob_vec("ask_price", a.lob.ask_prices, true);
            ss << ",";

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
            ss << "}";
            if (i < n_assets_ - 1) ss << ",";
        }
        ss << "],";

        // Matrix/vector helpers with explicit trailing-comma control
        auto print_mat = [&](const char* name, const Eigen::MatrixXd& m, bool last = false) {
            ss << "\"" << name << "\":[";
            int idx = 0;
            int total = static_cast<int>(m.rows() * m.cols());
            for (int i = 0; i < m.rows(); ++i) {
                for (int j = 0; j < m.cols(); ++j) {
                    ss << m(i, j);
                    if (++idx < total) ss << ",";
                }
            }
            ss << "]";
            if (!last) ss << ",";
        };
        auto print_vec = [&](const char* name, const Eigen::VectorXd& v, bool last = false) {
            ss << "\"" << name << "\":[";
            for (int i = 0; i < v.size(); ++i) {
                ss << v(i);
                if (i < v.size() - 1) ss << ",";
            }
            ss << "]";
            if (!last) ss << ",";
        };

        print_mat("correlation", state.correlation);
        print_mat("raw_correlation", state.raw_correlation);
        print_vec("eigenvalues", state.eigenvalues);

        ss << "\"fiedler\":" << state.fiedler_value << ",";
        ss << "\"clustering\":" << state.clustering_coeff << ",";
        print_vec("betweenness", state.betweenness);
        print_vec("degree", state.degree);

        ss << "\"tri\":" << state.tda_risk_index << ",";
        ss << "\"wasserstein\":" << state.wasserstein_dist << ",";
        ss << "\"l1\":" << state.landscape_l1 << ",";
        ss << "\"l2\":" << state.landscape_l2 << ",";
        print_vec("ruin_vector", state.ruin_vector);
        print_mat("distance", state.distance, true);  // last field — no trailing comma

        ss << "}";

        // Send over TCP asynchronously via length-prefixed frame
        std::string payload = ss.str();
        write_future_ = std::async(std::launch::async, [this, payload]() {
            ensure_connected();
            if (connected_) {
                send_frame(payload);
            }
        });
    }
};

} // namespace sovereign
