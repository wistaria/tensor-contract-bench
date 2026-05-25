#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "tcb/case_loader.hpp"
#include "tcb/result_writer.hpp"
#include "tcb/tblis_matmul.hpp"

namespace {

void test_tblis_matmul_variants_pass_validation() {
  const auto benchmark_case = tcb::load_benchmark_case("cases/matmul/square.yaml");
  const std::vector<std::string> variants = {
      "ik,kj->ij",
      "ik,jk->ij",
      "ki,kj->ij",
      "ki,jk->ij",
  };

  std::vector<tcb::BenchmarkResult> results;
  for (const auto &einsum : variants) {
    auto result = tcb::run_tblis_matmul_square(benchmark_case, einsum, 16, 0, 2);
    assert(result.backend == "cpp:tblis");
    assert(result.einsum == einsum);
    assert(result.validation.status == tcb::ValidationStatus::Passed);
    assert(result.validation.max_abs_error.has_value());
    assert(*result.validation.max_abs_error <= benchmark_case.validation.tolerance.atol);
    results.push_back(result);
  }

  const auto path = std::filesystem::temp_directory_path() / "tcb_tblis_matmul_results.jsonl";
  tcb::write_result_jsonl(path, results);
  const std::string validate_command =
      "python3 tools/validate_schemas.py --kind result '" + path.string() + "' >/dev/null";
  const int status = std::system(validate_command.c_str());
  std::filesystem::remove(path);
  assert(status == 0);
}

} // namespace

int main() {
  test_tblis_matmul_variants_pass_validation();
  return 0;
}
