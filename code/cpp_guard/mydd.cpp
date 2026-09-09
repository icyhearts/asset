#include <charconv>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <sys/stat.h>
#include <unistd.h>

namespace {

namespace fs = std::filesystem;

constexpr std::uintmax_t kBytesPerMiB = 1024U * 1024U;

struct Options {
  std::uintmax_t max_file_size_bytes = 0;
  bool truncate_unconditionally = false;
};

struct Statistics {
  std::uintmax_t regular_files = 0;
  std::uintmax_t truncated_files = 0;
  std::uintmax_t failures = 0;
};

void print_usage(const char *program) {
  std::cerr << "Usage: " << program << " <max_file_size_mib>\n"
            << "  Recursively scan the current directory. Regular files larger than\n"
            << "  the limit are truncated to zero bytes. A limit of 0 truncates\n"
            << "  every regular file. Symbolic links and non-regular files are skipped.\n";
}

std::optional<Options> parse_options(std::string_view value) {
  if (value.empty()) {
    return std::nullopt;
  }

  std::uintmax_t max_mib = 0;
  const char *begin = value.data();
  const char *end = begin + value.size();
  const auto result = std::from_chars(begin, end, max_mib, 10);
  if (result.ec != std::errc{} || result.ptr != end) {
    return std::nullopt;
  }
  if (max_mib > std::numeric_limits<std::uintmax_t>::max() / kBytesPerMiB) {
    return std::nullopt;
  }

  return Options{max_mib * kBytesPerMiB, max_mib == 0};
}

void report_errno(const fs::path &path, const char *operation) {
  const int error_number = errno;
  std::cerr << "[WARNING] " << operation << " '" << path.string()
            << "' failed: " << std::strerror(error_number) << '\n';
}

bool should_truncate(const struct stat &file_stat, const Options &options) {
  if (options.truncate_unconditionally) {
    return true;
  }

  // A regular file cannot have a negative size. Keep the check explicit so a
  // malformed or raced stat result is never converted to a huge unsigned size.
  if (file_stat.st_size < 0) {
    return false;
  }
  return static_cast<std::uintmax_t>(file_stat.st_size) >
         options.max_file_size_bytes;
}

bool truncate_file(const fs::path &path, const Options &options,
                   Statistics &statistics) {
  // Use lstat first so files that are below the limit do not require write
  // permission merely to be inspected. It also makes symbolic-link handling
  // explicit; the later O_NOFOLLOW open protects the write side of the race.
  struct stat path_stat {};
  if (::lstat(path.c_str(), &path_stat) != 0) {
    report_errno(path, "lstat");
    ++statistics.failures;
    return false;
  }
  if (!S_ISREG(path_stat.st_mode)) {
    return true;
  }

  ++statistics.regular_files;
  if (!should_truncate(path_stat, options)) {
    return true;
  }

  int open_flags = O_WRONLY | O_CLOEXEC;
#ifdef O_NOFOLLOW
  open_flags |= O_NOFOLLOW;
#endif

  const int file_descriptor = ::open(path.c_str(), open_flags);
  if (file_descriptor < 0) {
    report_errno(path, "open");
    ++statistics.failures;
    return false;
  }

  struct stat file_stat {};
  if (::fstat(file_descriptor, &file_stat) != 0) {
    report_errno(path, "fstat");
    ++statistics.failures;
    (void)::close(file_descriptor);
    return false;
  }

  if (!S_ISREG(file_stat.st_mode)) {
    // The directory entry may have been replaced after the filesystem scan.
    (void)::close(file_descriptor);
    return true;
  }

  if (!should_truncate(file_stat, options)) {
    if (::close(file_descriptor) != 0) {
      report_errno(path, "close");
      ++statistics.failures;
      return false;
    }
    return true;
  }

  const auto old_size = file_stat.st_size;
  if (::ftruncate(file_descriptor, 0) != 0) {
    report_errno(path, "ftruncate");
    ++statistics.failures;
    (void)::close(file_descriptor);
    return false;
  }

  if (::close(file_descriptor) != 0) {
    report_errno(path, "close");
    ++statistics.failures;
    return false;
  }

  ++statistics.truncated_files;
  std::cout << "truncated '" << path.string() << "' (" << old_size
            << " bytes)\n";
  return true;
}

int scan_current_directory(const Options &options, Statistics &statistics) {
  std::error_code error_code;
  fs::recursive_directory_iterator iterator(
      fs::path{"."}, fs::directory_options::skip_permission_denied,
      error_code);
  if (error_code) {
    std::cerr << "[ERROR] cannot traverse current directory: "
              << error_code.message() << '\n';
    return 1;
  }

  const fs::recursive_directory_iterator end;
  while (iterator != end) {
    const fs::path path = iterator->path();
    std::error_code status_error;
    const fs::file_status status = iterator->symlink_status(status_error);
    if (status_error) {
      std::cerr << "[WARNING] cannot stat '" << path.string() << "': "
                << status_error.message() << '\n';
      ++statistics.failures;
    } else if (status.type() == fs::file_type::regular) {
      (void)truncate_file(path, options, statistics);
    }

    iterator.increment(error_code);
    if (error_code) {
      std::cerr << "[WARNING] cannot continue traversal: "
                << error_code.message() << '\n';
      ++statistics.failures;
      error_code.clear();
    }
  }

  return statistics.failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    print_usage(argv[0]);
    return 2;
  }

  const auto options = parse_options(argv[1]);
  if (!options.has_value()) {
    std::cerr << "[ERROR] max_file_size_mib must be a non-negative decimal "
                 "integer that fits in uintmax_t\n";
    print_usage(argv[0]);
    return 2;
  }

  Statistics statistics;
  const int scan_status = scan_current_directory(*options, statistics);
  std::cout << "scanned " << statistics.regular_files << " regular file(s), "
            << "truncated " << statistics.truncated_files << " file(s)";
  if (statistics.failures != 0) {
    std::cout << ", " << statistics.failures << " failure(s)";
  }
  std::cout << '\n';
  return scan_status;
}
