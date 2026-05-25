#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::string quote(const std::filesystem::path &path) {
  return "'" + path.string() + "'";
}

void test_tcb_run_writes_valid_jsonl(const char *binary_path) {
  const auto output_path = std::filesystem::temp_directory_path() / "tcb_run_matmul_result.jsonl";
  const std::string command = quote(binary_path) +
                              " --case cases/matmul/square.yaml"
                              " --backend cpp:reference"
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
    assert(line.find("\"N\":16") != std::string::npos);
  }

  const std::string validate_command =
      "python3 tools/validate_schemas.py --kind result " + quote(output_path) + " >/dev/null";
  const int validation_status = std::system(validate_command.c_str());
  std::filesystem::remove(output_path);
  assert(validation_status == 0);
}

void test_tcb_run_rejects_unsupported_backend(const char *binary_path) {
  const auto output_path = std::filesystem::temp_directory_path() / "tcb_run_invalid_backend.jsonl";
  const std::string command = quote(binary_path) +
                              " --case cases/matmul/square.yaml"
                              " --backend cpp:blas"
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
  test_tcb_run_writes_valid_jsonl(argv[1]);
  test_tcb_run_rejects_unsupported_backend(argv[1]);
  return 0;
}
