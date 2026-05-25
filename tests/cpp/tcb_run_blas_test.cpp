#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string quote(const std::filesystem::path &path) {
  return "'" + path.string() + "'";
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::size_t count_lines(const std::filesystem::path &path) {
  std::ifstream input(path);
  std::size_t count = 0;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty()) {
      ++count;
    }
  }
  return count;
}

void test_tcb_run_blas_all_variants(const char *binary_path) {
  const auto output_path = std::filesystem::temp_directory_path() / "tcb_run_blas_results.jsonl";
  const std::string command = quote(binary_path) +
                              " --case cases/matmul/square.yaml"
                              " --backend cpp:blas"
                              " --all-einsums"
                              " --N 16"
                              " --warmup 0"
                              " --repeat 2"
                              " --output " +
                              quote(output_path);
  assert(std::system(command.c_str()) == 0);

  const auto contents = read_file(output_path);
  assert(count_lines(output_path) == 4);
  assert(contents.find("\"backend\":\"cpp:blas\"") != std::string::npos);
  assert(contents.find("\"einsum\":\"ik,kj->ij\"") != std::string::npos);
  assert(contents.find("\"einsum\":\"ik,jk->ij\"") != std::string::npos);
  assert(contents.find("\"einsum\":\"ki,kj->ij\"") != std::string::npos);
  assert(contents.find("\"einsum\":\"ki,jk->ij\"") != std::string::npos);

  const std::string validate_command =
      "python3 tools/validate_schemas.py --kind result " + quote(output_path) + " >/dev/null";
  const int validation_status = std::system(validate_command.c_str());
  std::filesystem::remove(output_path);
  assert(validation_status == 0);
}

} // namespace

int main(int argc, char **argv) {
  assert(argc == 2);
  test_tcb_run_blas_all_variants(argv[1]);
  return 0;
}
