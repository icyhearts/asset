#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
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
#include <utility>
#include <vector>

namespace {

constexpr const char *kConfigPath = "/tmp/.kgp_wl_config.txt";

struct GuardConfig {
  std::unordered_set<std::string> whitelist;
  std::vector<int> signal_list;
};

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

std::optional<std::vector<int>> parse_signal_list(const std::string &value) {
  std::vector<int> signal_list;
  std::size_t begin = 0;

  while (begin <= value.size()) {
    const auto end = value.find(',', begin);
    const auto token = trim(value.substr(begin, end - begin));
    const auto signal_number = parse_signal_number(token);
    if (!signal_number.has_value()) {
      return std::nullopt;
    }
    signal_list.push_back(*signal_number);
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }

  if (signal_list.empty()) {
    return std::nullopt;
  }
  return signal_list;
}

std::optional<GuardConfig> load_config() {
  std::ifstream config_file(kConfigPath);
  if (!config_file.is_open()) {
    std::cerr << "[ERROR] cannot open config file " << kConfigPath << ": "
              << std::strerror(errno) << '\n';
    return std::nullopt;
  }

  std::optional<std::string> raw_whitelist;
  std::optional<std::string> raw_signal_list;
  bool valid = true;
  std::size_t line_number = 0;
  std::string line;
  while (std::getline(config_file, line)) {
    ++line_number;
    const auto normalized = trim(line);
    if (normalized.empty() || normalized.front() == '#') {
      continue;
    }

    const auto separator = normalized.find('=');
    if (separator == std::string::npos) {
      std::cerr << "[ERROR] " << kConfigPath << ':' << line_number
                << ": expected key=value\n";
      valid = false;
      continue;
    }

    const auto key = trim(normalized.substr(0, separator));
    const auto value = trim(normalized.substr(separator + 1));
    if (key == "WL") {
      if (raw_whitelist.has_value()) {
        std::cerr << "[ERROR] " << kConfigPath << ':' << line_number
                  << ": duplicate WL entry\n";
        valid = false;
      } else {
        raw_whitelist = value;
      }
    } else if (key == "sig_list") {
      if (raw_signal_list.has_value()) {
        std::cerr << "[ERROR] " << kConfigPath << ':' << line_number
                  << ": duplicate sig_list entry\n";
        valid = false;
      } else {
        raw_signal_list = value;
      }
    } else {
      std::cerr << "[ERROR] " << kConfigPath << ':' << line_number
                << ": unknown key '" << key << "'\n";
      valid = false;
    }
  }

  const bool read_failed = config_file.bad();
  config_file.close();
  if (read_failed) {
    std::cerr << "[ERROR] failed while reading config file " << kConfigPath << '\n';
    return std::nullopt;
  }
  if (!valid) {
    return std::nullopt;
  }
  if (!raw_whitelist.has_value()) {
    std::cerr << "[ERROR] " << kConfigPath << ": missing WL entry\n";
    return std::nullopt;
  }
  if (!raw_signal_list.has_value()) {
    std::cerr << "[ERROR] " << kConfigPath << ": missing sig_list entry\n";
    return std::nullopt;
  }

  auto whitelist = parse_whitelist(*raw_whitelist);
  if (whitelist.empty()) {
    std::cerr << "[ERROR] " << kConfigPath
              << ": WL must contain at least one username\n";
    return std::nullopt;
  }

  auto signal_list = parse_signal_list(*raw_signal_list);
  if (!signal_list.has_value()) {
    std::cerr << "[ERROR] " << kConfigPath
              << ": sig_list must be a comma-separated list of integers from 1 through "
              << SIGRTMAX << '\n';
    return std::nullopt;
  }

  return GuardConfig{std::move(whitelist), std::move(*signal_list)};
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

std::optional<std::string> read_proc_file(pid_t pid, const char *entry) {
  const std::string path = "/proc/" + std::to_string(pid) + '/' + entry;
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    return std::nullopt;
  }

  std::string contents((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  if (file.bad()) {
    return std::nullopt;
  }
  return contents;
}

std::optional<std::string> read_proc_link(pid_t pid, const char *entry) {
  const std::string path = "/proc/" + std::to_string(pid) + '/' + entry;
  std::vector<char> buffer(256);
  constexpr std::size_t max_link_size = 1024 * 1024;

  while (buffer.size() <= max_link_size) {
    const ssize_t length = ::readlink(path.c_str(), buffer.data(), buffer.size());
    if (length < 0) {
      return std::nullopt;
    }
    if (static_cast<std::size_t>(length) < buffer.size()) {
      return std::string(buffer.data(), static_cast<std::size_t>(length));
    }
    buffer.resize(buffer.size() * 2);
  }

  return std::nullopt;
}

std::optional<pid_t> parse_parent_pid(const std::string &status) {
  std::size_t begin = 0;
  while (begin <= status.size()) {
    const auto end = status.find('\n', begin);
    const auto line = status.substr(begin, end - begin);
    constexpr std::string_view ppid_prefix = "PPid:";
    if (line.find(ppid_prefix) == 0) {
      return parse_pid(trim(line.substr(ppid_prefix.size())));
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return std::nullopt;
}

std::vector<std::string> parse_proc_cmdline(const std::string &cmdline) {
  std::vector<std::string> arguments;
  std::size_t begin = 0;
  while (begin < cmdline.size()) {
    const auto end = cmdline.find('\0', begin);
    const auto argument_end = end == std::string::npos ? cmdline.size() : end;
    if (argument_end > begin) {
      arguments.emplace_back(cmdline.substr(begin, argument_end - begin));
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return arguments;
}

std::string quote_for_log(const std::string &value) {
  constexpr char hex_digits[] = "0123456789abcdef";
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (const unsigned char ch : value) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (std::isprint(ch) != 0) {
          escaped.push_back(static_cast<char>(ch));
        } else {
          escaped += "\\x";
          escaped.push_back(hex_digits[ch >> 4]);
          escaped.push_back(hex_digits[ch & 0x0f]);
        }
        break;
    }
  }
  escaped.push_back('"');
  return escaped;
}

std::string join_command_arguments(const std::vector<std::string> &arguments) {
  if (arguments.empty()) {
    return "<none>";
  }

  std::string result;
  for (const auto &argument : arguments) {
    if (!result.empty()) {
      result.push_back(' ');
    }
    result += quote_for_log(argument);
  }
  return result;
}

std::optional<std::string> query_process_start_time(pid_t pid) {
  const auto stat = read_proc_file(pid, "stat");
  if (!stat.has_value()) {
    return std::nullopt;
  }

  const auto last_rparen = stat->rfind(')');
  if (last_rparen == std::string::npos || last_rparen + 2 >= stat->size()) {
    return std::nullopt;
  }

  // Fields after ')': state(0) ppid(1) pgrp(2) session(3) tty_nr(4) tpgid(5)
  //   flags(6) minflt(7) cminflt(8) majflt(9) cmajflt(10) utime(11) stime(12)
  //   cutime(13) cstime(14) priority(15) nice(16) num_threads(17) itrealvalue(18)
  //   starttime(19)
  std::size_t pos = last_rparen + 2;
  for (int i = 0; i < 19; ++i) {
    pos = stat->find(' ', pos);
    if (pos == std::string::npos) {
      return std::nullopt;
    }
    ++pos;
  }
  const auto end = stat->find(' ', pos);
  const auto field = stat->substr(pos, end == std::string::npos ? std::string::npos : end - pos);

  long long start_ticks = 0;
  const auto *begin = field.data();
  const auto *field_end = begin + field.size();
  if (std::from_chars(begin, field_end, start_ticks).ec != std::errc{}) {
    return std::nullopt;
  }

  const long ticks_per_sec = ::sysconf(_SC_CLK_TCK);
  if (ticks_per_sec <= 0) {
    return std::nullopt;
  }

  // Read boot time from /proc/stat btime field
  std::time_t boot_epoch = 0;
  {
    std::ifstream proc_stat("/proc/stat");
    std::string line;
    while (std::getline(proc_stat, line)) {
      if (line.compare(0, 6, "btime ") == 0) {
        long long bt = 0;
        std::from_chars(line.data() + 6, line.data() + line.size(), bt);
        boot_epoch = static_cast<std::time_t>(bt);
        break;
      }
    }
    if (boot_epoch == 0) {
      return std::nullopt;
    }
  }

  const std::time_t start_epoch =
      boot_epoch + static_cast<std::time_t>(start_ticks / ticks_per_sec);

  struct tm tm_buf {};
  if (::localtime_r(&start_epoch, &tm_buf) == nullptr) {
    return std::nullopt;
  }

  char buf[64];
  if (std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S%z", &tm_buf) == 0) {
    return std::nullopt;
  }
  return std::string(buf);
}

struct ProcessInfo {
  std::string parent_pid = "<unavailable>";
  std::string command = "<unavailable>";
  std::string command_args = "<unavailable>";
  std::string work_directory = "<unavailable>";
  std::string start_time = "<unavailable>";
};

ProcessInfo query_process_info(pid_t pid) {
  ProcessInfo info;

  if (const auto status = read_proc_file(pid, "status"); status.has_value()) {
    if (const auto parent_pid = parse_parent_pid(*status); parent_pid.has_value()) {
      info.parent_pid = std::to_string(*parent_pid);
    }
  }

  std::vector<std::string> arguments;
  if (const auto cmdline = read_proc_file(pid, "cmdline"); cmdline.has_value()) {
    arguments = parse_proc_cmdline(*cmdline);
    if (!arguments.empty()) {
      info.command_args = join_command_arguments(arguments);
    } else {
      info.command_args = "<none>";
    }
  }

  if (const auto comm = read_proc_file(pid, "comm"); comm.has_value()) {
    const auto command = trim(*comm);
    if (!command.empty()) {
      info.command = command;
    }
  } else if (!arguments.empty()) {
    info.command = arguments.front();
  }

  if (const auto work_directory = read_proc_link(pid, "cwd");
      work_directory.has_value()) {
    info.work_directory = *work_directory;
  }

  if (const auto start_time = query_process_start_time(pid);
      start_time.has_value()) {
    info.start_time = *start_time;
  }

  return info;
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

bool run_once(const std::unordered_set<std::string> &whitelist,
              const std::vector<int> &signal_list, std::size_t &next_signal_index) {
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

    const int signal_number = signal_list[next_signal_index];
    const auto process_info = query_process_info(pid);
    std::cout << "[PROCESS] pid=" << pid << " ppid=" << process_info.parent_pid
              << " owner=" << *username << " command="
              << quote_for_log(process_info.command)
              << " command_args=" << process_info.command_args
              << " work_directory=" << quote_for_log(process_info.work_directory)
              << " start_time=" << process_info.start_time
              << " signal=" << signal_number << std::endl;
    if (::kill(pid, signal_number) != 0) {
      std::cerr << "[WARN] failed to send signal " << signal_number << " to PID " << pid
                << " (ppid=" << process_info.parent_pid
                << " command=" << quote_for_log(process_info.command)
                << " args=" << process_info.command_args
                << " cwd=" << quote_for_log(process_info.work_directory)
                << " start_time=" << process_info.start_time
                << ") owned by " << *username << ": " << std::strerror(errno) << '\n';
      succeeded = false;
      continue;
    }

    std::cout << "[SIGNAL] PID " << pid << " (ppid=" << process_info.parent_pid
              << " command=" << quote_for_log(process_info.command)
              << " args=" << process_info.command_args
              << " cwd=" << quote_for_log(process_info.work_directory)
              << " start_time=" << process_info.start_time
              << ") owned by " << *username << ": sent signal "
              << signal_number << std::endl;
    next_signal_index = (next_signal_index + 1) % signal_list.size();
  }

  return succeeded;
}

}  // namespace

int main() {
  const auto config = load_config();
  if (!config.has_value()) {
    return 1;
  }

  std::cout << "[CONFIG] loaded " << kConfigPath << ": " << config->whitelist.size()
            << " whitelisted users, " << config->signal_list.size() << " signals\n";

  std::size_t next_signal_index = 0;
  while (true) {
    if (!run_once(config->whitelist, config->signal_list, next_signal_index)) {
      std::cerr << "[WARN] scan completed with errors; retrying\n";
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}
