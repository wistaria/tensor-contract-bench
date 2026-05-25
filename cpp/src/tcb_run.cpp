#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "tcb/case_loader.hpp"
#include "tcb/reference_matmul.hpp"
#include "tcb/result_writer.hpp"

namespace {

struct Options {
  std::filesystem::path case_path;
  std::string backend;
  std::string einsum;
  bool all_einsums = false;
  std::optional<tcb::Index> n;
  bool all_n = false;
  std::optional<int> warmup;
  std::optional<int> repeat;
  std::filesystem::path output_path;
};

void print_usage(std::ostream &out) {
  out << "Usage: tcb-run --case PATH --backend cpp:reference --N VALUE "
         "--einsum EXPR --warmup VALUE --repeat VALUE --output PATH\n"
         "       tcb-run --case PATH --backend cpp:reference --all-N "
         "--all-einsums --warmup VALUE --repeat VALUE --output PATH\n";
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
    if (arg == "--all-N") {
      options.all_n = true;
      continue;
    }
    if (arg == "--all-einsums") {
      options.all_einsums = true;
      continue;
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
  if (options.n.has_value() == options.all_n) {
    err << "specify exactly one of --N or --all-N\n";
    return false;
  }
  if (options.einsum.empty() == options.all_einsums) {
    err << "specify exactly one of --einsum or --all-einsums\n";
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

std::vector<tcb::Index> selected_n_values(const tcb::BenchmarkCase &benchmark_case, const Options &options) {
  if (options.n.has_value()) {
    return {*options.n};
  }
  const auto found = benchmark_case.parameters.find("N");
  if (found == benchmark_case.parameters.end() || found->second.empty()) {
    throw tcb::ReferenceMatmulError("case does not define parameter N");
  }
  return found->second;
}

std::vector<std::string> selected_einsums(const tcb::BenchmarkCase &benchmark_case, const Options &options) {
  if (!options.einsum.empty()) {
    return {options.einsum};
  }
  if (benchmark_case.variants.contraction_permutations.mode != tcb::PermutationMode::Selected ||
      benchmark_case.variants.contraction_permutations.selected_einsums.empty()) {
    throw tcb::ReferenceMatmulError("case does not define selected contraction variants");
  }
  return benchmark_case.variants.contraction_permutations.selected_einsums;
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
    std::vector<tcb::BenchmarkResult> results;
    const auto n_values = selected_n_values(benchmark_case, options);
    const auto einsums = selected_einsums(benchmark_case, options);
    results.reserve(n_values.size() * einsums.size());
    for (const auto n : n_values) {
      for (const auto &einsum : einsums) {
        results.push_back(tcb::run_reference_matmul_square(benchmark_case, einsum, n, *options.warmup, *options.repeat));
      }
    }
    tcb::write_result_jsonl(options.output_path, results);
  } catch (const std::exception &error) {
    std::cerr << "tcb-run: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
