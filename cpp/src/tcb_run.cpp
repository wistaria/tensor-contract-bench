#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

#include "tcb/case_loader.hpp"
#include "tcb/reference_matmul.hpp"
#include "tcb/result_writer.hpp"

namespace {

struct Options {
  std::filesystem::path case_path;
  std::string backend;
  std::string einsum;
  std::optional<tcb::Index> n;
  std::optional<int> warmup;
  std::optional<int> repeat;
  std::filesystem::path output_path;
};

void print_usage(std::ostream &out) {
  out << "Usage: tcb-run --case PATH --backend cpp:reference --N VALUE "
         "--einsum EXPR --warmup VALUE --repeat VALUE --output PATH\n";
}

bool parse_int64(const std::string &text, tcb::Index &value) {
  try {
    std::size_t parsed = 0;
    const auto converted = std::stoll(text, &parsed);
    if (parsed != text.size()) {
      return false;
    }
    value = static_cast<tcb::Index>(converted);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool parse_int(const std::string &text, int &value) {
  try {
    std::size_t parsed = 0;
    const auto converted = std::stoi(text, &parsed);
    if (parsed != text.size()) {
      return false;
    }
    value = converted;
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool parse_options(int argc, char **argv, Options &options, std::ostream &err) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(std::cout);
      return false;
    }
    if (i + 1 >= argc) {
      err << "missing value for " << arg << '\n';
      return false;
    }
    const std::string value = argv[++i];
    if (arg == "--case") {
      options.case_path = value;
    } else if (arg == "--backend") {
      options.backend = value;
    } else if (arg == "--einsum") {
      options.einsum = value;
    } else if (arg == "--N") {
      tcb::Index n = 0;
      if (!parse_int64(value, n)) {
        err << "invalid --N value: " << value << '\n';
        return false;
      }
      options.n = n;
    } else if (arg == "--warmup") {
      int warmup = 0;
      if (!parse_int(value, warmup)) {
        err << "invalid --warmup value: " << value << '\n';
        return false;
      }
      options.warmup = warmup;
    } else if (arg == "--repeat") {
      int repeat = 0;
      if (!parse_int(value, repeat)) {
        err << "invalid --repeat value: " << value << '\n';
        return false;
      }
      options.repeat = repeat;
    } else if (arg == "--output") {
      options.output_path = value;
    } else {
      err << "unknown option: " << arg << '\n';
      return false;
    }
  }

  if (options.case_path.empty()) {
    err << "missing required --case\n";
    return false;
  }
  if (options.backend.empty()) {
    err << "missing required --backend\n";
    return false;
  }
  if (!options.n.has_value()) {
    err << "missing required --N\n";
    return false;
  }
  if (options.einsum.empty()) {
    err << "missing required --einsum\n";
    return false;
  }
  if (!options.warmup.has_value()) {
    err << "missing required --warmup\n";
    return false;
  }
  if (!options.repeat.has_value()) {
    err << "missing required --repeat\n";
    return false;
  }
  if (options.output_path.empty()) {
    err << "missing required --output\n";
    return false;
  }
  if (options.backend != "cpp:reference") {
    err << "unsupported backend: " << options.backend << '\n';
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  Options options;
  if (!parse_options(argc, argv, options, std::cerr)) {
    print_usage(std::cerr);
    return 2;
  }

  try {
    const auto benchmark_case = tcb::load_benchmark_case(options.case_path);
    const auto result =
        tcb::run_reference_matmul_square(benchmark_case, options.einsum, *options.n, *options.warmup, *options.repeat);
    tcb::write_result_jsonl(options.output_path, {result});
  } catch (const std::exception &error) {
    std::cerr << "tcb-run: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
