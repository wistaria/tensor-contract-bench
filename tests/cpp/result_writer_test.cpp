#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "tcb/result_writer.hpp"

namespace {

tcb::BenchmarkResult make_synthetic_result() {
  tcb::BenchmarkResult result;
  result.case_name = "matmul_square";
  result.einsum = "ik,kj->ij";
  result.backend = "cpp:reference";
  result.language = tcb::Language::Cpp;
  result.device = tcb::Device::Cpu;
  result.dtype = tcb::DType::Float64;
  result.parameters = {{"N", 16}};
  result.input_layouts = {
      {"A", {{"i", "k"}, {"i", "k"}, {16, 1}}},
      {"B", {{"k", "j"}, {"k", "j"}, {16, 1}}},
  };
  result.output_layout = {{"i", "j"}, {"i", "j"}, {16, 1}};
  result.timing = {
      1,
      3,
      {0.003, 0.002, 0.004},
      0.002,
      0.003,
      0.003,
      0.000816496580927726,
  };
  result.performance.flops = 8192;
  result.performance.gflops = 0.0027306666666666666;
  result.performance.bytes_estimated = 6144;
  result.performance.bandwidth_gbs_estimated = 0.002048;
  result.validation.status = tcb::ValidationStatus::Passed;
  result.validation.max_abs_error = 0.0;
  result.validation.max_rel_error = 0.0;
  result.system.cpu = "unknown";
  result.system.gpu = std::nullopt;
  result.system.num_threads = 1;
  result.system.compiler = "unknown";
  result.system.compiler_version = std::nullopt;
  result.system.library_version = {{"reference", "unknown"}};
  return result;
}

void test_write_to_stream() {
  std::ostringstream output;
  tcb::write_result_jsonl(output, make_synthetic_result());
  const auto line = output.str();
  assert(line.find("\"case_name\":\"matmul_square\"") != std::string::npos);
  assert(line.find("\"backend\":\"cpp:reference\"") != std::string::npos);
  assert(!line.empty());
  assert(line.back() == '\n');
}

void test_write_file_and_validate_schema() {
  const auto path = std::filesystem::temp_directory_path() / "tcb_synthetic_result.jsonl";
  tcb::write_result_jsonl(path, {make_synthetic_result()});

  {
    std::ifstream input(path);
    std::string line;
    std::getline(input, line);
    assert(line.find("\"timing\"") != std::string::npos);
  }

  const std::string command =
      "python3 tools/validate_schemas.py --kind result " + path.string() + " >/dev/null";
  const int status = std::system(command.c_str());
  std::filesystem::remove(path);
  assert(status == 0);
}

} // namespace

int main() {
  test_write_to_stream();
  test_write_file_and_validate_schema();
  return 0;
}
