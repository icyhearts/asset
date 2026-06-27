#include <algorithm>
#include <atomic>
#include <cassert>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

using std::size_t;
using std::vector;

bool single_thread_contain_target(const vector<int>& arr, int target) {
  std::unordered_map<int, int> seen;
  seen.reserve(arr.size() * 2);

  for (int x : arr) {
    auto it = seen.find(target - x);
    if (it != seen.end() && it->second > 0) {
      return true;
    }
    ++seen[x];
  }
  return false;
}

bool parallel_contain_target(const vector<int>& arr, int target) {
  const size_t n = arr.size();
  if (n < 2) {
    return false;
  }

  std::unordered_map<int, int> counts;
  counts.reserve(n * 2);
  for (int x : arr) {
    ++counts[x];
  }

  unsigned int thread_count = std::thread::hardware_concurrency();
  if (thread_count == 0) {
    thread_count = 2;
  }
  thread_count = std::min<unsigned int>(thread_count, static_cast<unsigned int>(n));

  std::atomic<bool> found{false};
  vector<std::thread> workers;
  workers.reserve(thread_count);

  const size_t chunk = (n + thread_count - 1) / thread_count;

  for (unsigned int tid = 0; tid < thread_count; ++tid) {
    const size_t begin = tid * chunk;
    const size_t end = std::min(n, begin + chunk);
    if (begin >= end) {
      break;
    }

    workers.emplace_back([&arr, &counts, &found, target, begin, end]() {
      for (size_t i = begin; i < end && !found.load(std::memory_order_relaxed); ++i) {
        const int x = arr[i];
        const int y = target - x;
        auto it = counts.find(y);
        if (it == counts.end()) {
          continue;
        }

        if (y != x || it->second >= 2) {
          found.store(true, std::memory_order_relaxed);
          return;
        }
      }
    });
  }

  for (auto& worker : workers) {
    worker.join();
  }

  return found.load(std::memory_order_relaxed);
}

void run_case(const vector<int>& arr, int target, bool expected, const char* name) {
  const bool single = single_thread_contain_target(arr, target);
  const bool parallel = parallel_contain_target(arr, target);

  std::cout << name << ": target=" << target
            << ", single=" << std::boolalpha << single
            << ", parallel=" << parallel << '\n';

  assert(single == expected);
  assert(parallel == single);
}

int main() {
  run_case({2, 7, 11, 15}, 9, true, "basic true");
  run_case({1, 2, 3, 4}, 8, false, "basic false");
  run_case({3, 3}, 6, true, "same value twice");
  run_case({3}, 6, false, "same value once");
  run_case({-10, 5, 20, 7, -3}, 2, true, "negative numbers");

  vector<int> large(1'000'000);
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> dist(-2'000'000, 2'000'000);
  for (int& x : large) {
    x = dist(rng);
  }
  large[123456] = 111'111'111;
  large[987654] = -22'222'222;
  run_case(large, 88'888'889, true, "large true");

  vector<int> no_match(200'000);
  for (size_t i = 0; i < no_match.size(); ++i) {
    no_match[i] = static_cast<int>(i * 2);
  }
  run_case(no_match, -1, false, "large false");

  std::cout << "All tests passed.\n";
  return 0;
}
