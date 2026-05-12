#pragma once
/// @file clock.hpp
/// @brief Simulation clock — hybrid synchronous + event-driven scheduling.

#include <functional>
#include <queue>
#include <vector>

namespace sovereign {

struct Event {
    double time;
    int    asset_id;
    int    event_type;   ///< 0=market, 1=limit, 2=cancel, 3=modify, 4=hidden
    int    depth_level;
    double size;
    bool   is_buy;

    bool operator>(const Event& o) const { return time > o.time; }
};

class SimulationClock {
    double t_       = 0.0;
    double dt_      = 1e-4;
    double t_end_   = 1.0;
    int    step_    = 0;

    /// Priority queue for Hawkes-driven events (earliest first)
    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> event_queue_;

public:
    SimulationClock() = default;
    SimulationClock(double dt, double T) : dt_(dt), t_end_(T) {}

    double t()       const { return t_; }
    double dt()      const { return dt_; }
    double t_end()   const { return t_end_; }
    int    step()    const { return step_; }
    bool   done()    const { return t_ >= t_end_; }

    /// Advance to next synchronous step
    void tick() { t_ += dt_; ++step_; }

    /// Schedule an asynchronous event
    void schedule(const Event& e) { event_queue_.push(e); }

    /// Check if there's a pending event before the next sync tick
    bool has_event_before_tick() const {
        return !event_queue_.empty() && event_queue_.top().time < t_ + dt_;
    }

    /// Pop the next event (must check has_event_before_tick first)
    Event pop_event() {
        Event e = event_queue_.top();
        event_queue_.pop();
        return e;
    }

    /// Number of pending events
    size_t pending_events() const { return event_queue_.size(); }

    /// Drain all events up to time t_now + dt
    std::vector<Event> drain_events() {
        std::vector<Event> events;
        double deadline = t_ + dt_;
        while (!event_queue_.empty() && event_queue_.top().time < deadline) {
            events.push_back(event_queue_.top());
            event_queue_.pop();
        }
        return events;
    }
};

} // namespace sovereign
