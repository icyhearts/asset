#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

using std::size_t;
using std::vector;

unsigned int worker_count_for(size_t n) {
  if (n == 0) {
    return 0;
  }

  unsigned int thread_count = std::thread::hardware_concurrency();
  if (thread_count == 0) {
    thread_count = 2;
  }
  return std::min<unsigned int>(thread_count, static_cast<unsigned int>(n));
}

size_t bucket_index(int value, size_t bucket_count) {
  return std::hash<int>{}(value) % bucket_count;
}

bool checked_complement(int x, int target, int& y) {
  const long long value = static_cast<long long>(target) - x;
  if (value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    return false;
  }

  y = static_cast<int>(value);
  return true;
}

bool single_thread_contain_target(const vector<int>& arr, int target) {
  std::unordered_map<int, int> seen;
  seen.reserve(arr.size() * 2);

  for (int x : arr) {
    int y = 0;
    if (!checked_complement(x, target, y)) {
      ++seen[x];
      continue;
    }

    auto it = seen.find(y);
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

  const unsigned int thread_count = worker_count_for(n);
  const size_t bucket_count = thread_count;
  const size_t chunk = (n + thread_count - 1) / thread_count;
  vector<std::thread> workers;
  workers.reserve(thread_count);

  vector<vector<vector<int>>> local_buckets(
      thread_count, vector<vector<int>>(bucket_count));

  // Phase 1: each worker scans N/T elements and writes only to its own buckets.
  for (unsigned int tid = 0; tid < thread_count; ++tid) {
    const size_t begin = tid * chunk;
    const size_t end = std::min(n, begin + chunk);
    if (begin >= end) {
      break;
    }

    workers.emplace_back([&arr, &local_buckets, bucket_count, tid, begin, end, chunk]() {
      const size_t reserve_per_bucket = chunk / bucket_count + 1;
      for (size_t b = 0; b < bucket_count; ++b) {
        local_buckets[tid][b].reserve(reserve_per_bucket);
      }

      for (size_t i = begin; i < end; ++i) {
        const int x = arr[i];
        local_buckets[tid][bucket_index(x, bucket_count)].push_back(x);
      }
    });
  }

  for (auto& worker : workers) {
    worker.join();
  }
  workers.clear();

  vector<std::unordered_map<int, int>> bucket_counts(bucket_count);

  // Phase 2: each bucket is owned by one worker, so no locks are needed.
  for (size_t bucket = 0; bucket < bucket_count; ++bucket) {
    workers.emplace_back([&local_buckets, &bucket_counts, thread_count, bucket]() {
      size_t total = 0;
      for (unsigned int tid = 0; tid < thread_count; ++tid) {
        total += local_buckets[tid][bucket].size();
      }

      auto& counts = bucket_counts[bucket];
      counts.reserve(total * 2 + 1);
      for (unsigned int tid = 0; tid < thread_count; ++tid) {
        for (int x : local_buckets[tid][bucket]) {
          ++counts[x];
        }
      }
    });
  }

  for (auto& worker : workers) {
    worker.join();
  }
  workers.clear();

  std::atomic<bool> found{false};

  // Phase 3: scan arr in parallel. bucket_counts is read-only in this phase.
  for (unsigned int tid = 0; tid < thread_count; ++tid) {
    const size_t begin = tid * chunk;
    const size_t end = std::min(n, begin + chunk);
    if (begin >= end) {
      break;
    }

    workers.emplace_back([&arr, &bucket_counts, &found, target, bucket_count, begin, end]() {
      for (size_t i = begin; i < end && !found.load(std::memory_order_relaxed); ++i) {
        const int x = arr[i];

        int y = 0;
        if (!checked_complement(x, target, y)) {
          continue;
        }

        const auto& counts = bucket_counts[bucket_index(y, bucket_count)];
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

template <class Func>
std::pair<bool, double> timed_call(Func&& func) {
  const auto start = std::chrono::steady_clock::now();
  const bool result = func();
  const auto stop = std::chrono::steady_clock::now();
  const std::chrono::duration<double, std::milli> elapsed = stop - start;
  return {result, elapsed.count()};
}

void run_case(const vector<int>& arr, int target, bool expected, const char* name) {
  const auto [single, single_ms] =
      timed_call([&]() { return single_thread_contain_target(arr, target); });
  const auto [parallel, parallel_ms] =
      timed_call([&]() { return parallel_contain_target(arr, target); });
  const double speedup = parallel_ms > 0.0 ? single_ms / parallel_ms : 0.0;

  std::cout << name << ": size=" << arr.size()
            << ", target=" << target
            << ", single=" << std::boolalpha << single
            << ", parallel=" << parallel
            << std::fixed << std::setprecision(3)
            << ", single_ms=" << single_ms
            << ", parallel_ms=" << parallel_ms
            << ", speedup=" << speedup << "x\n";

  assert(single == expected);
  assert(parallel == single);
}

int main() {
  run_case({2, 7, 11, 15}, 9, true, "basic true");
  run_case({1, 2, 3, 4}, 8, false, "basic false");
  run_case({3, 3}, 6, true, "same value twice");
  run_case({3}, 6, false, "same value once");
  run_case({-10, 5, 20, 7, -3}, 2, true, "negative numbers");
  run_case({std::numeric_limits<int>::max(), -1}, std::numeric_limits<int>::max() - 1,
           true, "int max true");
  run_case({std::numeric_limits<int>::min(), 1}, std::numeric_limits<int>::max(),
           false, "overflow complement false");

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
  unsigned int thread_count = std::thread::hardware_concurrency();
  std::cout << "thread_count:" << thread_count << "\n";
  return 0;
}
