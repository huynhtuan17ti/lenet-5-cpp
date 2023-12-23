#pragma once
#include <chrono>

class Timer {
  using DT = std::chrono::milliseconds;
  using ClockT = std::chrono::steady_clock;
  using timep_t = typename ClockT::time_point;
  timep_t _start = ClockT::now(), _end = {};

 public:
  void tick() {
    _end = timep_t{};
    _start = ClockT::now();
  }

  void tock() { _end = ClockT::now(); }

  auto duration() const { return std::chrono::duration_cast<DT>(_end - _start); }
};
