#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "tcb/case_loader.hpp"
#include "tcb/reference_matmul.hpp"
#include "tcb/result_writer.hpp"

namespace {

void test_reference_matmul_square_writes_valid_result() {
  const auto benchmark_case = tcb::load_benchmark_case("cases/matmul/square.yaml");
  const auto result = tcb::run_reference_matmul_square(benchmark_case, 16, 1, 3);

  assert(result.case_name == "matmul_square");
  assert(result.backend == "cpp:reference");
  assert(result.parameters.at("N") == 16);
  assert(result.timing.repeat == 3);
  assert(result.timing.times_sec.size() == 3);
  assert(result.performance.flops == 8192);
  assert(result.validation.status == tcb::ValidationStatus::Passed);
  assert(result.validation.max_abs_error.has_value());
  assert(*result.validation.max_abs_error <= benchmark_case.validation.tolerance.atol);

  const auto path = std::filesystem::temp_directory_path() / "tcb_reference_matmul_result.jsonl";
  tcb::write_result_jsonl(path, {result});

  {
    std::ifstream input(path);
    std::string line;
    std::getline(input, line);
    assert(line.find("\"case_name\":\"matmul_square\"") != std::string::npos);
    assert(line.find("\"backend\":\"cpp:reference\"") != std::string::npos);
  }

  const std::string command =
      "python3 tools/validate_schemas.py --kind result " + path.string() + " >/dev/null";
  const int status = std::system(command.c_str());
  std::filesystem::remove(path);
  assert(status == 0);
}

void test_rejects_non_matmul_case() {
  const auto benchmark_case = tcb::load_benchmark_case("cases/rank3/one_bond.yaml");
  bool rejected = false;
  try {
    (void)tcb::run_reference_matmul_square(benchmark_case, 4, 0, 1);
  } catch (const tcb::ReferenceMatmulError &) {
    rejected = true;
  }
  assert(rejected);
}

} // namespace

int main() {
  test_reference_matmul_square_writes_valid_result();
  test_rejects_non_matmul_case();
  return 0;
}
