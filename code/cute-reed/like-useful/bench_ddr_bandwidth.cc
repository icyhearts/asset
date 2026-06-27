#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <emmintrin.h>
#endif

#if defined(__linux__)
#include <sys/mman.h>
#endif

using std::size_t;

namespace {

constexpr size_t kCacheLineBytes = 64;
constexpr size_t kMiB = 1024ull * 1024ull;

volatile uint64_t g_sink = 0;

struct Buffer {
  uint8_t* data = nullptr;
  size_t bytes = 0;

  explicit Buffer(size_t bytes_) : bytes(bytes_) {
    void* p = nullptr;
    if (posix_memalign(&p, kCacheLineBytes, bytes) != 0) {
      throw std::bad_alloc();
    }
    data = static_cast<uint8_t*>(p);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    madvise(data, bytes, MADV_HUGEPAGE);
#endif
  }

  Buffer(Buffer const&) = delete;
  Buffer& operator=(Buffer const&) = delete;

  ~Buffer() {
    std::free(data);
  }
};

unsigned int worker_count_for(size_t n, unsigned int requested) {
  if (n == 0) {
    return 0;
  }

  unsigned int threads = requested;
  if (threads == 0) {
    threads = std::thread::hardware_concurrency();
  }
  if (threads == 0) {
    threads = 1;
  }
  return std::min<unsigned int>(threads, static_cast<unsigned int>(n));
}

size_t parse_size_token(std::string text) {
  if (text.empty()) {
    return 0;
  }

  char suffix = text.back();
  size_t multiplier = kMiB;
  if (suffix == 'k' || suffix == 'K' || suffix == 'm' || suffix == 'M' ||
      suffix == 'g' || suffix == 'G') {
    text.pop_back();
    if (suffix == 'k' || suffix == 'K') {
      multiplier = 1024ull;
    } else if (suffix == 'm' || suffix == 'M') {
      multiplier = kMiB;
    } else {
      multiplier = 1024ull * kMiB;
    }
  }

  return std::stoull(text) * multiplier;
}

size_t parse_cache_size(std::string text) {
  if (text.empty()) {
    return 0;
  }

  char suffix = text.back();
  size_t multiplier = 1;
  if (suffix == 'K' || suffix == 'k' || suffix == 'M' || suffix == 'm') {
    text.pop_back();
    multiplier = (suffix == 'K' || suffix == 'k') ? 1024ull : kMiB;
  }

  return std::stoull(text) * multiplier;
}

size_t detect_llc_bytes() {
  // Linux exposes cache sizes per CPU. index3 is usually LLC on x86 servers.
  std::ifstream input("/sys/devices/system/cpu/cpu0/cache/index3/size");
  std::string text;
  if (input >> text) {
    return parse_cache_size(text);
  }
  return 0;
}

template <class Func>
void parallel_for(size_t work_items, unsigned int threads, Func&& func) {
  const size_t chunk = (work_items + threads - 1) / threads;
  std::vector<std::thread> workers;
  workers.reserve(threads);

  for (unsigned int tid = 0; tid < threads; ++tid) {
    const size_t begin = tid * chunk;
    const size_t end = std::min(work_items, begin + chunk);
    if (begin >= end) {
      break;
    }
    workers.emplace_back([&, tid, begin, end]() {
      func(tid, begin, end);
    });
  }

  for (auto& worker : workers) {
    worker.join();
  }
}

void initialize_buffer(Buffer& buffer, unsigned int threads, uint64_t seed) {
  auto* ptr = reinterpret_cast<uint64_t*>(buffer.data);
  const size_t elems = buffer.bytes / sizeof(uint64_t);

  parallel_for(elems, threads, [&](unsigned int tid, size_t begin, size_t end) {
    uint64_t value = seed + 0x9e3779b97f4a7c15ull * (tid + 1);
    for (size_t i = begin; i < end; ++i) {
      value = value * 2862933555777941757ull + 3037000493ull;
      ptr[i] = value;
    }
  });
}

void flush_cache_lines(void* data, size_t bytes) {
#if defined(__x86_64__) || defined(__i386__)
  auto* p = static_cast<char*>(data);
  auto* end = p + bytes;
  for (; p < end; p += kCacheLineBytes) {
    _mm_clflush(p);
  }
  _mm_mfence();
#else
  (void)data;
  (void)bytes;
#endif
}

void stream_store_u64(uint64_t* ptr, uint64_t value) {
#if defined(__x86_64__) || defined(__i386__)
  _mm_stream_si64(reinterpret_cast<long long*>(ptr), static_cast<long long>(value));
#else
  *ptr = value;
#endif
}

void stream_store_fence() {
#if defined(__x86_64__) || defined(__i386__)
  _mm_sfence();
#endif
}

double seconds_since(std::chrono::steady_clock::time_point start,
                     std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double>(stop - start).count();
}

double best_read_seconds(Buffer& src, unsigned int threads, int iterations) {
  auto* ptr = reinterpret_cast<uint64_t*>(src.data);
  const size_t elems = src.bytes / sizeof(uint64_t);
  double best = std::numeric_limits<double>::max();

  for (int iter = 0; iter < iterations; ++iter) {
    flush_cache_lines(src.data, src.bytes);

    std::vector<uint64_t> partials(threads, 0);
    const auto start = std::chrono::steady_clock::now();
    parallel_for(elems, threads, [&](unsigned int tid, size_t begin, size_t end) {
      uint64_t sum = 0;
      for (size_t i = begin; i < end; ++i) {
        sum += ptr[i];
      }
      partials[tid] = sum;
    });
    const auto stop = std::chrono::steady_clock::now();

    g_sink ^= std::accumulate(partials.begin(), partials.end(), uint64_t{0});
    best = std::min(best, seconds_since(start, stop));
  }

  return best;
}

double best_write_seconds(Buffer& dst, unsigned int threads, int iterations) {
  auto* ptr = reinterpret_cast<uint64_t*>(dst.data);
  const size_t elems = dst.bytes / sizeof(uint64_t);
  double best = std::numeric_limits<double>::max();

  for (int iter = 0; iter < iterations; ++iter) {
    flush_cache_lines(dst.data, dst.bytes);

    const auto start = std::chrono::steady_clock::now();
    parallel_for(elems, threads, [&](unsigned int tid, size_t begin, size_t end) {
      uint64_t value = 0x123456789abcdef0ull + tid + iter;
      for (size_t i = begin; i < end; ++i) {
        stream_store_u64(ptr + i, value + i);
      }
      stream_store_fence();
    });
    const auto stop = std::chrono::steady_clock::now();

    best = std::min(best, seconds_since(start, stop));
  }

  return best;
}

double best_copy_seconds(Buffer& src, Buffer& dst, unsigned int threads, int iterations) {
  auto* src_ptr = reinterpret_cast<uint64_t*>(src.data);
  auto* dst_ptr = reinterpret_cast<uint64_t*>(dst.data);
  const size_t elems = src.bytes / sizeof(uint64_t);
  double best = std::numeric_limits<double>::max();

  for (int iter = 0; iter < iterations; ++iter) {
    flush_cache_lines(src.data, src.bytes);
    flush_cache_lines(dst.data, dst.bytes);

    const auto start = std::chrono::steady_clock::now();
    parallel_for(elems, threads, [&](unsigned int, size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        stream_store_u64(dst_ptr + i, src_ptr[i]);
      }
      stream_store_fence();
    });
    const auto stop = std::chrono::steady_clock::now();

    best = std::min(best, seconds_since(start, stop));
  }

  return best;
}

void print_result(char const* name, size_t bytes, double seconds) {
  const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
  const double bandwidth = gib / seconds;
  std::cout << std::left << std::setw(18) << name
            << " bytes=" << std::right << std::setw(12) << bytes
            << " best_s=" << std::fixed << std::setprecision(6) << seconds
            << " bandwidth=" << std::setprecision(2) << bandwidth << " GiB/s\n";
}

void print_usage(char const* argv0) {
  std::cout << "Usage: " << argv0 << " [size] [threads] [iterations]\n"
            << "  size examples: 512M, 2G, 1024M. Default: max(512M, 4*LLC).\n"
            << "  threads=0 means hardware_concurrency(). Default: 0.\n"
            << "  iterations default: 3. Best time is reported.\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && (std::strcmp(argv[1], "-h") == 0 ||
                   std::strcmp(argv[1], "--help") == 0)) {
    print_usage(argv[0]);
    return 0;
  }

  const size_t llc_bytes = detect_llc_bytes();
  size_t bytes = std::max<size_t>(512ull * kMiB, llc_bytes * 4);
  unsigned int requested_threads = 0;
  int iterations = 3;

  if (argc > 1) {
    bytes = parse_size_token(argv[1]);
  }
  if (argc > 2) {
    requested_threads = static_cast<unsigned int>(std::stoul(argv[2]));
  }
  if (argc > 3) {
    iterations = std::max(1, std::stoi(argv[3]));
  }

  bytes = (bytes / sizeof(uint64_t)) * sizeof(uint64_t);
  if (bytes == 0) {
    std::cerr << "buffer size must be at least 8 bytes\n";
    return 1;
  }

  const unsigned int threads = worker_count_for(bytes / sizeof(uint64_t), requested_threads);

  std::cout << "DDR bandwidth benchmark\n"
            << "  buffer_size = " << bytes / static_cast<double>(kMiB) << " MiB\n"
            << "  threads     = " << threads << "\n"
            << "  iterations  = " << iterations << "\n"
            << "  detected LLC= " << llc_bytes / static_cast<double>(kMiB) << " MiB\n"
#if defined(__x86_64__) || defined(__i386__)
            << "  cache flush = clflush before each timed pass\n"
            << "  write path  = non-temporal stores\n";
#else
            << "  cache flush = unavailable on this architecture\n"
            << "  write path  = normal stores\n";
#endif

  try {
    Buffer src(bytes);
    Buffer dst(bytes);

    initialize_buffer(src, threads, 0x1234);
    initialize_buffer(dst, threads, 0x5678);

    const double read_s = best_read_seconds(src, threads, iterations);
    const double write_s = best_write_seconds(dst, threads, iterations);
    const double copy_s = best_copy_seconds(src, dst, threads, iterations);

    print_result("read", bytes, read_s);
    print_result("nt_write", bytes, write_s);
    print_result("read+nt_write", bytes * 2, copy_s);

    std::cout << "sink=" << g_sink << "\n";
  } catch (std::bad_alloc const&) {
    std::cerr << "failed to allocate buffers. Try a smaller size, for example: "
              << argv[0] << " 256M 8 3\n";
    return 1;
  } catch (std::exception const& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
