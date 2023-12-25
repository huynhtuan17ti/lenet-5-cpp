#pragma once
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include "VariadicTable.h"

class Timer {
 private:
  using DT = std::chrono::milliseconds;
  using ClockT = std::chrono::steady_clock;
  using timep_t = typename ClockT::time_point;
  timep_t _start = ClockT::now(), _end = {};
  std::vector<int64_t> stats;

 public:
  void Tick() {
    _end = timep_t{};
    _start = ClockT::now();
  }

  void Tock() { _end = ClockT::now(); }

  auto Duration() const { return std::chrono::duration_cast<DT>(_end - _start); }

  void Record() {
    int64_t dur = Duration().count();
    stats.push_back(dur);
  }

  void Report() const {
    VariadicTable<std::string, int64_t> vt({"Network", "Elapsed Time (ms)"}, 10);
    for (size_t i = 0; i < stats.size(); ++i) {
      vt.addRow("Layer " + std::to_string(i + 1), stats[i]);
    }
    vt.print(std::cout);
  }
};
