#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct CommandResult {
  int status = 0;
  std::string output;
};

std::string trim(const std::string &value) {
  auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
    return std::isspace(ch) != 0;
  });
  auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
    return std::isspace(ch) != 0;
  }).base();

  if (begin >= end) {
    return {};
  }
  return std::string(begin, end);
}

std::vector<std::string> split(const std::string &value, char delimiter) {
  std::vector<std::string> fields;
  std::stringstream stream(value);
  std::string field;
  while (std::getline(stream, field, delimiter)) {
    fields.push_back(field);
  }
  if (!value.empty() && value.back() == delimiter) {
    fields.emplace_back();
  }
  return fields;
}

std::vector<std::string> lines(const std::string &value) {
  std::vector<std::string> result;
  std::stringstream stream(value);
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    result.push_back(line);
  }
  return result;
}

std::string shell_quote(const std::string &value) {
  std::string quoted = "'";
  for (char ch : value) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted += ch;
    }
  }
  quoted += "'";
  return quoted;
}

std::filesystem::path make_temp_path() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  std::random_device random_device;
  std::uniform_int_distribution<unsigned long long> distribution;

  std::filesystem::path path;
  do {
    std::ostringstream name;
    name << "kgp_" << now << "_" << distribution(random_device) << ".out";
    path = std::filesystem::temp_directory_path() / name.str();
  } while (std::filesystem::exists(path));

  return path;
}

CommandResult run_and_capture(const std::string &command, bool capture_stderr) {
  const auto output_path = make_temp_path();
  std::string full_command = command + " > " + shell_quote(output_path.string());
  full_command += capture_stderr ? " 2>&1" : " 2>/dev/null";

  CommandResult result;
  result.status = std::system(full_command.c_str());

  std::ifstream input(output_path);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  result.output = buffer.str();

  std::error_code error_code;
  std::filesystem::remove(output_path, error_code);

  return result;
}

bool run_command_quiet(const std::string &command) {
  const std::string full_command = command + " >/dev/null 2>&1";
  return std::system(full_command.c_str()) == 0;
}

std::optional<int> parse_positive_int(const std::string &value) {
  if (value.empty()) {
    return std::nullopt;
  }

  int parsed = 0;
  for (char ch : value) {
    if (std::isdigit(static_cast<unsigned char>(ch)) == 0) {
      return std::nullopt;
    }
    const int digit = ch - '0';
    if (parsed > (2147483647 - digit) / 10) {
      return std::nullopt;
    }
    parsed = parsed * 10 + digit;
  }

  if (parsed <= 0) {
    return std::nullopt;
  }
  return parsed;
}

bool is_positive_integer(const std::string &value) {
  return parse_positive_int(value).has_value();
}

std::string basename(const std::string &path) {
  const auto pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

void print_usage(const std::string &program) {
  std::cerr << "Usage:\n"
            << "  " << program << " [--once] <gpu_id_list> <protected_user> <sig_num>\n"
            << "  " << program << " --loop <gpu_id_list> <protected_user> <sig_num>\n"
            << "Example:\n"
            << "  " << program << " \"1,3,5\" like 11\n"
            << "  " << program << " --loop \"1,3,5\" like 11\n";
}

std::map<std::string, std::string> query_gpu_uuid_by_index() {
  std::map<std::string, std::string> result;
  const auto command_result = run_and_capture(
      "nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits", true);

  if (command_result.status != 0) {
    std::cerr << "[ERROR] nvidia-smi GPU query failed\n";
    if (!command_result.output.empty()) {
      std::cerr << command_result.output;
    }
    return result;
  }

  for (const auto &line : lines(command_result.output)) {
    const auto fields = split(line, ',');
    if (fields.size() < 2) {
      continue;
    }
    const auto index = trim(fields[0]);
    const auto uuid = trim(fields[1]);
    if (!index.empty() && !uuid.empty()) {
      result[index] = uuid;
    }
  }

  return result;
}

std::vector<std::pair<std::string, std::string>> query_compute_apps() {
  std::vector<std::pair<std::string, std::string>> result;
  const auto command_result = run_and_capture(
      "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits",
      true);

  if (command_result.status != 0) {
    std::cerr << "[ERROR] nvidia-smi compute-app query failed\n";
    if (!command_result.output.empty()) {
      std::cerr << command_result.output;
    }
    return result;
  }

  for (const auto &line : lines(command_result.output)) {
    const auto fields = split(line, ',');
    if (fields.size() < 2) {
      continue;
    }

    const auto uuid = trim(fields[0]);
    const auto pid = trim(fields[1]);
    if (!uuid.empty() && is_positive_integer(pid)) {
      result.emplace_back(uuid, pid);
    }
  }

  return result;
}

std::string query_owner(const std::string &pid) {
  const auto command_result = run_and_capture("ps -o user= -p " + pid, false);
  if (command_result.status != 0) {
    return {};
  }
  return trim(command_result.output);
}

bool send_signal(const std::string &pid, int signal_number) {
  return run_command_quiet("kill -" + std::to_string(signal_number) + " " + pid);
}

std::vector<std::string> parse_gpu_list(const std::string &gpu_list) {
  std::vector<std::string> result;
  for (const auto &field : split(gpu_list, ',')) {
    const auto gpu = trim(field);
    if (!gpu.empty()) {
      result.push_back(gpu);
    }
  }
  return result;
}

bool run_once(const std::string &gpu_list, const std::string &protected_user,
              int signal_number) {
  const auto gpu_ids = parse_gpu_list(gpu_list);
  const auto gpu_uuid_by_index = query_gpu_uuid_by_index();
  if (gpu_uuid_by_index.empty()) {
    return false;
  }

  const auto compute_apps = query_compute_apps();

  for (const auto &gpu_index : gpu_ids) {
    const auto uuid_it = gpu_uuid_by_index.find(gpu_index);
    if (uuid_it == gpu_uuid_by_index.end()) {
      continue;
    }

    const auto &gpu_uuid = uuid_it->second;
    for (const auto &[app_uuid, pid] : compute_apps) {
      if (app_uuid != gpu_uuid) {
        continue;
      }

      const auto owner = query_owner(pid);
      if (owner.empty() || owner == protected_user) {
        continue;
      }

      std::cout << "[KILL] GPU " << gpu_index << " PID " << pid << " owned by "
                << owner << '\n';
      if (!send_signal(pid, signal_number)) {
        std::cerr << "[WARN] failed to send signal " << signal_number
                  << " to PID " << pid << '\n';
      }
    }
  }

  return true;
}

}  // namespace

int main(int argc, char **argv) {
  enum class Mode { once, loop };

  const std::string program = basename(argc > 0 ? argv[0] : "kgp");
  Mode mode = program.find("loop") != std::string::npos ? Mode::loop : Mode::once;

  int first_arg = 1;
  if (argc > 1) {
    const std::string option = argv[1];
    if (option == "--loop") {
      mode = Mode::loop;
      first_arg = 2;
    } else if (option == "--once") {
      mode = Mode::once;
      first_arg = 2;
    } else if (option == "-h" || option == "--help") {
      print_usage(program);
      return 0;
    }
  }

  if (argc - first_arg != 3) {
    print_usage(program);
    return 1;
  }

  const std::string gpu_list = argv[first_arg];
  const std::string protected_user = argv[first_arg + 1];
  const auto signal_number = parse_positive_int(argv[first_arg + 2]);
  if (!signal_number.has_value()) {
    std::cerr << "[ERROR] sig_num must be a positive integer\n";
    return 1;
  }

  if (mode == Mode::once) {
    return run_once(gpu_list, protected_user, *signal_number) ? 0 : 1;
  }

  while (true) {
    if (!run_once(gpu_list, protected_user, *signal_number)) {
      return 1;
    }
    //std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}
