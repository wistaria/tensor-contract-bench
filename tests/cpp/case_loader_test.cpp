#include <cassert>
#include <filesystem>
#include <fstream>

#include "tcb/case_loader.hpp"

namespace {

void test_load_matmul_case(const std::filesystem::path &repo_root) {
  const auto benchmark_case = tcb::load_benchmark_case(repo_root / "cases" / "matmul" / "square.yaml");

  assert(benchmark_case.name == "matmul_square");
  assert(benchmark_case.dtype == tcb::DType::Float64);
  assert(benchmark_case.einsum == "ik,kj->ij");
  assert(benchmark_case.parameters.at("N").size() == 7);
  assert(benchmark_case.inputs.size() == 2);
  assert(benchmark_case.inputs.at("A").labels == tcb::Labels({"i", "k"}));
  assert(benchmark_case.inputs.at("A").dims.at("i").is_parameter());
  assert(benchmark_case.inputs.at("A").dims.at("i").parameter_name() == "N");
  assert(benchmark_case.output.labels == tcb::Labels({"i", "j"}));
  assert(benchmark_case.variants.contraction_permutations.mode == tcb::PermutationMode::Selected);
  assert(benchmark_case.variants.contraction_permutations.selected_einsums.at(1) == "ik,jk->ij");
  assert(benchmark_case.validation.reference_backend == "numpy");
}

void test_load_rank_cases(const std::filesystem::path &repo_root) {
  const auto rank3 = tcb::load_benchmark_case(repo_root / "cases" / "rank3" / "one_bond.yaml");
  assert(rank3.name == "rank3_one_bond");
  assert(rank3.inputs.at("A").labels == tcb::Labels({"i", "j", "a"}));
  assert(rank3.parameters.at("X").back() == 64);
  assert(rank3.variants.output_permutations.mode == tcb::PermutationMode::All);

  const auto rank4 = tcb::load_benchmark_case(repo_root / "cases" / "rank4" / "two_bond.yaml");
  assert(rank4.name == "rank4_two_bond");
  assert(rank4.inputs.at("B").labels == tcb::Labels({"a", "b", "k", "l"}));
  assert(rank4.parameters.at("X").back() == 128);
  assert(rank4.validation.tolerance.rtol == 1.0e-10);
}

void test_reject_unknown_parameter(const std::filesystem::path &repo_root) {
  (void)repo_root;
  const auto path = std::filesystem::temp_directory_path() / "tcb_invalid_case_unknown_parameter.yaml";
  {
    std::ofstream out(path);
    out << "name: invalid_case\n"
        << "description: Invalid case\n"
        << "dtype: float64\n"
        << "einsum: ik,kj->ij\n"
        << "inputs:\n"
        << "  A:\n"
        << "    labels: [i, k]\n"
        << "    dims:\n"
        << "      i: Missing\n"
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
        << "  N: [16]\n"
        << "variants:\n"
        << "  output_permutations: selected\n"
        << "  input_permutations: selected\n"
        << "  contraction_permutations:\n"
        << "    - ik,kj->ij\n"
        << "validation:\n"
        << "  reference_backend: numpy\n"
        << "  tolerance:\n"
        << "    rtol: 1.0e-10\n"
        << "    atol: 1.0e-12\n";
  }

  bool rejected = false;
  try {
    (void)tcb::load_benchmark_case(path);
  } catch (const tcb::CaseLoadError &) {
    rejected = true;
  }
  std::filesystem::remove(path);
  assert(rejected);
}

} // namespace

int main() {
  const auto repo_root = std::filesystem::current_path();
  test_load_matmul_case(repo_root);
  test_load_rank_cases(repo_root);
  test_reject_unknown_parameter(repo_root);
  return 0;
}
