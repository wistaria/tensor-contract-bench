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

std::string read_file(const std::filesystem::path &path) {
  std::ifstream input(path);
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

void validate_result_file(const std::filesystem::path &path) {
  const std::string validate_command =
      "python3 tools/validate_schemas.py --kind result " + quote(path) + " >/dev/null";
  assert(std::system(validate_command.c_str()) == 0);
}

void write_small_matmul_case(const std::filesystem::path &case_path) {
  std::ofstream out(case_path);
  out << "name: matmul_square\n"
      << "description: Small square matrix multiplication case for tcb-run tests.\n"
      << "dtype: float64\n"
      << "einsum: ik,kj->ij\n"
      << "inputs:\n"
      << "  A:\n"
      << "    labels: [i, k]\n"
      << "    dims:\n"
      << "      i: N\n"
      << "      k: N\n"
      << "    layout:\n"
      << "      kind: row_major\n"
      << "      physical_order: [i, k]\n"
      << "  B:\n"
      << "    labels: [k, j]\n"
      << "    dims:\n"
      << "      k: N\n"
      << "      j: N\n"
      << "    layout:\n"
      << "      kind: row_major\n"
      << "      physical_order: [k, j]\n"
      << "output:\n"
      << "  labels: [i, j]\n"
      << "  layout:\n"
      << "    kind: row_major\n"
      << "    physical_order: [i, j]\n"
      << "parameters:\n"
      << "  N: [16, 32]\n"
      << "variants:\n"
      << "  output_permutations: selected\n"
      << "  input_permutations: selected\n"
      << "  contraction_permutations:\n"
      << "    - ik,kj->ij\n"
      << "    - ik,jk->ij\n"
      << "    - ki,kj->ij\n"
      << "    - ki,jk->ij\n"
      << "validation:\n"
      << "  reference_backend: numpy\n"
      << "  tolerance:\n"
      << "    rtol: 1.0e-10\n"
      << "    atol: 1.0e-12\n";
}

void test_tcb_run_writes_valid_jsonl_for_explicit_variant(const char *binary_path, const std::string &einsum) {
  const auto output_path = std::filesystem::temp_directory_path() / "tcb_run_explicit_result.jsonl";
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

  const auto contents = read_file(output_path);
  assert(count_lines(output_path) == 1);
  assert(contents.find("\"case_name\":\"matmul_square\"") != std::string::npos);
  assert(contents.find("\"backend\":\"cpp:reference\"") != std::string::npos);
  assert(contents.find("\"einsum\":\"" + einsum + "\"") != std::string::npos);
  assert(contents.find("\"N\":16") != std::string::npos);
  validate_result_file(output_path);
  std::filesystem::remove(output_path);
}

void test_tcb_run_writes_valid_jsonl_for_explicit_variants(const char *binary_path) {
  const std::vector<std::string> variants = {
      "ik,kj->ij",
      "ik,jk->ij",
      "ki,kj->ij",
      "ki,jk->ij",
  };
  for (const auto &einsum : variants) {
    test_tcb_run_writes_valid_jsonl_for_explicit_variant(binary_path, einsum);
  }
}

void test_tcb_run_all_einsums_for_one_n(const char *binary_path) {
  const auto output_path = std::filesystem::temp_directory_path() / "tcb_run_all_einsums_result.jsonl";
  const std::string command = quote(binary_path) +
                              " --case cases/matmul/square.yaml"
                              " --backend cpp:reference"
                              " --all-einsums"
                              " --N 16"
                              " --warmup 0"
                              " --repeat 1"
                              " --output " +
                              quote(output_path);
  assert(std::system(command.c_str()) == 0);
  const auto contents = read_file(output_path);
  assert(count_lines(output_path) == 4);
  assert(contents.find("\"einsum\":\"ik,kj->ij\"") != std::string::npos);
  assert(contents.find("\"einsum\":\"ik,jk->ij\"") != std::string::npos);
  assert(contents.find("\"einsum\":\"ki,kj->ij\"") != std::string::npos);
  assert(contents.find("\"einsum\":\"ki,jk->ij\"") != std::string::npos);
  validate_result_file(output_path);
  std::filesystem::remove(output_path);
}

void test_tcb_run_all_n_for_one_einsum(const char *binary_path) {
  const auto case_path = std::filesystem::temp_directory_path() / "tcb_run_small_all_n.yaml";
  const auto output_path = std::filesystem::temp_directory_path() / "tcb_run_all_n_result.jsonl";
  write_small_matmul_case(case_path);
  const std::string command = quote(binary_path) +
                              " --case " + quote(case_path) +
                              " --backend cpp:reference"
                              " --einsum 'ik,kj->ij'"
                              " --all-N"
                              " --warmup 0"
                              " --repeat 1"
                              " --output " +
                              quote(output_path);
  assert(std::system(command.c_str()) == 0);
  const auto contents = read_file(output_path);
  assert(count_lines(output_path) == 2);
  assert(contents.find("\"N\":16") != std::string::npos);
  assert(contents.find("\"N\":32") != std::string::npos);
  validate_result_file(output_path);
  std::filesystem::remove(case_path);
  std::filesystem::remove(output_path);
}

void test_tcb_run_full_sweep_on_small_case(const char *binary_path) {
  const auto case_path = std::filesystem::temp_directory_path() / "tcb_run_small_matmul_square.yaml";
  const auto output_path = std::filesystem::temp_directory_path() / "tcb_run_small_full_sweep.jsonl";
  write_small_matmul_case(case_path);
  const std::string command = quote(binary_path) +
                              " --case " + quote(case_path) +
                              " --backend cpp:reference"
                              " --all-einsums"
                              " --all-N"
                              " --warmup 0"
                              " --repeat 1"
                              " --output " +
                              quote(output_path);
  assert(std::system(command.c_str()) == 0);
  assert(count_lines(output_path) == 8);
  validate_result_file(output_path);
  std::filesystem::remove(case_path);
  std::filesystem::remove(output_path);
}

#ifndef TCB_HAVE_BLAS
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
#endif

} // namespace

int main(int argc, char **argv) {
  assert(argc == 2);
  test_tcb_run_writes_valid_jsonl_for_explicit_variants(argv[1]);
  test_tcb_run_all_einsums_for_one_n(argv[1]);
  test_tcb_run_all_n_for_one_einsum(argv[1]);
  test_tcb_run_full_sweep_on_small_case(argv[1]);
#ifndef TCB_HAVE_BLAS
  test_tcb_run_rejects_unsupported_backend(argv[1]);
#endif
  return 0;
}
