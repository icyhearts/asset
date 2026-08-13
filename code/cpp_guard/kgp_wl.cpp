#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <pwd.h>
#include <set>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <unordered_set>
#include <vector>

namespace {

std::string trim(const std::string &value) {
  const auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
    return std::isspace(ch) != 0;
  });
  const auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
    return std::isspace(ch) != 0;
  }).base();

  if (begin >= end) {
    return {};
  }
  return std::string(begin, end);
}

std::unordered_set<std::string> parse_whitelist(const std::string &value) {
  std::unordered_set<std::string> whitelist;
  std::size_t begin = 0;

  while (begin <= value.size()) {
    const auto end = value.find(',', begin);
    const auto username = trim(value.substr(begin, end - begin));
    if (!username.empty()) {
      whitelist.insert(username);
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }

  return whitelist;
}

std::optional<int> parse_signal_number(const std::string &value) {
  if (value.empty()) {
    return std::nullopt;
  }

  int signal_number = 0;
  const auto *begin = value.data();
  const auto *end = begin + value.size();
  const auto result = std::from_chars(begin, end, signal_number);
  if (result.ec != std::errc{} || result.ptr != end || signal_number <= 0 ||
      signal_number > SIGRTMAX) {
    return std::nullopt;
  }
  return signal_number;
}

std::optional<pid_t> parse_pid(const std::string &value) {
  if (value.empty()) {
    return std::nullopt;
  }

  long long parsed = 0;
  const auto *begin = value.data();
  const auto *end = begin + value.size();
  const auto result = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc{} || result.ptr != end || parsed <= 0 ||
      parsed > std::numeric_limits<pid_t>::max()) {
    return std::nullopt;
  }
  return static_cast<pid_t>(parsed);
}

std::optional<std::set<pid_t>> query_compute_pids() {
  constexpr const char *command =
      "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits";
  FILE *pipe = ::popen(command, "r");
  if (pipe == nullptr) {
    std::cerr << "[ERROR] failed to start nvidia-smi: " << std::strerror(errno) << '\n';
    return std::nullopt;
  }

  std::string output;
  char buffer[4096];
  while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }
  const bool read_failed = std::ferror(pipe) != 0;
  const int status = ::pclose(pipe);

  if (read_failed) {
    std::cerr << "[ERROR] failed to read nvidia-smi output\n";
    return std::nullopt;
  }
  if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::cerr << "[ERROR] nvidia-smi compute-app query failed\n";
    return std::nullopt;
  }

  std::set<pid_t> pids;
  std::size_t begin = 0;
  while (begin <= output.size()) {
    const auto end = output.find('\n', begin);
    auto line = output.substr(begin, end - begin);
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (const auto pid = parse_pid(trim(line)); pid.has_value()) {
      pids.insert(*pid);
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }

  return pids;
}

std::optional<uid_t> query_process_uid(pid_t pid) {
  const std::string proc_path = "/proc/" + std::to_string(pid);
  struct stat proc_stat {};
  if (::stat(proc_path.c_str(), &proc_stat) != 0) {
    return std::nullopt;
  }
  return proc_stat.st_uid;
}

std::optional<std::string> query_username(uid_t uid) {
  long suggested_size = ::sysconf(_SC_GETPW_R_SIZE_MAX);
  std::size_t buffer_size = suggested_size > 0 ? static_cast<std::size_t>(suggested_size)
                                                : 16384;
  constexpr std::size_t max_buffer_size = 1024 * 1024;

  while (buffer_size <= max_buffer_size) {
    std::vector<char> buffer(buffer_size);
    struct passwd password {};
    struct passwd *result = nullptr;
    const int error = ::getpwuid_r(uid, &password, buffer.data(), buffer.size(), &result);
    if (error == 0) {
      if (result == nullptr || result->pw_name == nullptr || result->pw_name[0] == '\0') {
        return std::nullopt;
      }
      return std::string(result->pw_name);
    }
    if (error != ERANGE) {
      return std::nullopt;
    }
    buffer_size *= 2;
  }

  return std::nullopt;
}

void print_usage(const char *program) {
  std::cerr << "Usage:\n"
            << "  WL=user1,user2 " << program << " <signal_number>\n"
            << "  WL=user1,user2 " << program << " --loop <signal_number>\n";
}

bool run_once(const std::unordered_set<std::string> &whitelist, int signal_number) {
  const auto pids = query_compute_pids();
  if (!pids.has_value()) {
    return false;
  }

  bool succeeded = true;
  for (pid_t pid : *pids) {
    const auto uid = query_process_uid(pid);
    if (!uid.has_value()) {
      std::cerr << "[WARN] cannot determine owner of PID " << pid << "; skip\n";
      succeeded = false;
      continue;
    }

    const auto username = query_username(*uid);
    if (!username.has_value()) {
      std::cerr << "[WARN] cannot resolve username for PID " << pid << " (UID " << *uid
                << "); skip\n";
      succeeded = false;
      continue;
    }
    if (whitelist.find(*username) != whitelist.end()) {
      continue;
    }

    const auto current_uid = query_process_uid(pid);
    if (!current_uid.has_value()) {
      std::cerr << "[WARN] PID " << pid << " exited before signal delivery\n";
      continue;
    }
    if (*current_uid != *uid) {
      std::cerr << "[WARN] owner of PID " << pid << " changed before signal delivery; skip\n";
      succeeded = false;
      continue;
    }

    if (::kill(pid, signal_number) != 0) {
      std::cerr << "[WARN] failed to send signal " << signal_number << " to PID " << pid
                << " owned by " << *username << ": " << std::strerror(errno) << '\n';
      succeeded = false;
      continue;
    }

    std::cout << "[SIGNAL] PID " << pid << " owned by " << *username << ": sent signal "
              << signal_number << std::endl;
  }

  return succeeded;
}

}  // namespace

int main(int argc, char **argv) {
  const char *raw_whitelist = std::getenv("WL");
  if (raw_whitelist == nullptr || trim(raw_whitelist).empty()) {
    std::cerr << "[ERROR] WL must be a non-empty comma-separated username whitelist\n";
    return 1;
  }

  const auto whitelist = parse_whitelist(raw_whitelist);
  if (whitelist.empty()) {
    std::cerr << "[ERROR] WL must contain at least one username\n";
    return 1;
  }

  bool loop = false;
  const char *signal_argument = nullptr;
  if (argc == 2) {
    signal_argument = argv[1];
  } else if (argc == 3 && std::string(argv[1]) == "--loop") {
    loop = true;
    signal_argument = argv[2];
  } else {
    print_usage(argc > 0 ? argv[0] : "kgp_wl");
    return 1;
  }

  const auto signal_number = parse_signal_number(signal_argument);
  if (!signal_number.has_value()) {
    std::cerr << "[ERROR] signal_number must be an integer from 1 through " << SIGRTMAX
              << '\n';
    return 1;
  }

  if (!loop) {
    return run_once(whitelist, *signal_number) ? 0 : 1;
  }

  while (true) {
    if (!run_once(whitelist, *signal_number)) {
      return 1;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}
