#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::string quote(const std::filesystem::path &path) {
  return "'" + path.string() + "'";
}

void test_tcb_run_writes_valid_jsonl_for_variant(const char *binary_path, const std::string &einsum) {
  const auto output_path =
      std::filesystem::temp_directory_path() / ("tcb_run_matmul_" + einsum.substr(0, 2) + "_" +
                                               einsum.substr(3, 2) + "_result.jsonl");
  const std::string command = quote(binary_path) +
                              " --case cases/matmul/square.yaml"
                              " --backend cpp:reference"
                              " --einsum " +
                              quote(einsum) +
                              " --N 16"
                              " --warmup 1"
                              " --repeat 3"
                              " --output " +
                              quote(output_path);
  assert(std::system(command.c_str()) == 0);

  {
    std::ifstream input(output_path);
    std::string line;
    std::getline(input, line);
    assert(line.find("\"case_name\":\"matmul_square\"") != std::string::npos);
    assert(line.find("\"backend\":\"cpp:reference\"") != std::string::npos);
    assert(line.find("\"einsum\":\"" + einsum + "\"") != std::string::npos);
    assert(line.find("\"N\":16") != std::string::npos);
  }

  const std::string validate_command =
      "python3 tools/validate_schemas.py --kind result " + quote(output_path) + " >/dev/null";
  const int validation_status = std::system(validate_command.c_str());
  std::filesystem::remove(output_path);
  assert(validation_status == 0);
}

void test_tcb_run_writes_valid_jsonl_for_all_variants(const char *binary_path) {
  const std::vector<std::string> variants = {
      "ik,kj->ij",
      "ik,jk->ij",
      "ki,kj->ij",
      "ki,jk->ij",
  };
  for (const auto &einsum : variants) {
    test_tcb_run_writes_valid_jsonl_for_variant(binary_path, einsum);
  }
}

void test_tcb_run_rejects_unsupported_backend(const char *binary_path) {
  const auto output_path = std::filesystem::temp_directory_path() / "tcb_run_invalid_backend.jsonl";
  const std::string command = quote(binary_path) +
                              " --case cases/matmul/square.yaml"
                              " --backend cpp:blas"
                              " --einsum 'ik,kj->ij'"
                              " --N 16"
                              " --warmup 0"
                              " --repeat 1"
                              " --output " +
                              quote(output_path) + " >/dev/null 2>/dev/null";
  assert(std::system(command.c_str()) != 0);
  std::filesystem::remove(output_path);
}

} // namespace

int main(int argc, char **argv) {
  assert(argc == 2);
  test_tcb_run_writes_valid_jsonl_for_all_variants(argv[1]);
  test_tcb_run_rejects_unsupported_backend(argv[1]);
  return 0;
}
