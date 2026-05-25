#include <cassert>
#include <string>

#include "tcb/benchmark_case.hpp"
#include "tcb/tensor_view.hpp"

namespace {

void test_dtype_helpers() {
  assert(tcb::parse_dtype("float32") == tcb::DType::Float32);
  assert(tcb::parse_dtype("float64") == tcb::DType::Float64);
  assert(tcb::parse_dtype("complex64") == tcb::DType::Complex64);
  assert(tcb::parse_dtype("complex128") == tcb::DType::Complex128);
  assert(!tcb::parse_dtype("int64").has_value());
  assert(tcb::dtype_name(tcb::DType::Complex128) == "complex128");
  assert(tcb::dtype_size_bytes(tcb::DType::Complex128) == 16);
}

void test_layout_helpers() {
  assert(tcb::parse_layout_kind("row_major") == tcb::LayoutKind::RowMajor);
  assert(tcb::parse_layout_kind("column_major") == tcb::LayoutKind::ColumnMajor);
  assert(tcb::parse_layout_kind("strided") == tcb::LayoutKind::Strided);
  assert(!tcb::parse_layout_kind("blocked").has_value());
  assert(tcb::layout_kind_name(tcb::LayoutKind::ColumnMajor) == "column_major");

  tcb::LayoutSpec layout;
  layout.kind = tcb::LayoutKind::Strided;
  layout.physical_order = {"a", "b", "i", "j"};
  layout.strides = {262144, 4096, 64, 1};
  assert(layout.has_explicit_strides());
}

void test_tensor_view() {
  double data[2 * 3 * 4] = {};
  tcb::TensorView view;
  view.data = data;
  view.dtype = tcb::DType::Float64;
  view.shape = {2, 3, 4};
  view.strides = {12, 4, 1};
  view.labels = {"i", "j", "k"};

  assert(view.rank() == 3);
  assert(view.is_valid());
  assert(view.element_count() == 24);
  assert(view.element_size_bytes() == 8);

  view.data = nullptr;
  assert(!view.is_valid());

  view.data = data;
  view.strides = {12, 4};
  assert(!view.is_valid());
}

void test_benchmark_case_structs() {
  tcb::BenchmarkCase benchmark_case;
  benchmark_case.name = "matmul_square";
  benchmark_case.description = "Square matrix multiplication";
  benchmark_case.dtype = tcb::DType::Float64;
  benchmark_case.einsum = "ik,kj->ij";
  benchmark_case.parameters["N"] = {16, 32, 64};

  tcb::InputTensorSpec input_a;
  input_a.labels = {"i", "k"};
  input_a.dims["i"] = tcb::Dimension::parameter("N");
  input_a.dims["k"] = tcb::Dimension::parameter("N");
  input_a.layout.kind = tcb::LayoutKind::RowMajor;
  input_a.layout.physical_order = {"i", "k"};

  tcb::InputTensorSpec input_b;
  input_b.labels = {"k", "j"};
  input_b.dims["k"] = tcb::Dimension::parameter("N");
  input_b.dims["j"] = tcb::Dimension::parameter("N");
  input_b.layout.kind = tcb::LayoutKind::RowMajor;
  input_b.layout.physical_order = {"k", "j"};

  benchmark_case.inputs["A"] = input_a;
  benchmark_case.inputs["B"] = input_b;
  benchmark_case.output.labels = {"i", "j"};
  benchmark_case.output.layout.kind = tcb::LayoutKind::RowMajor;
  benchmark_case.output.layout.physical_order = {"i", "j"};
  benchmark_case.variants.output_permutations.mode = tcb::PermutationMode::Selected;
  benchmark_case.variants.output_permutations.selected = {{"i", "j"}, {"j", "i"}};
  benchmark_case.variants.contraction_permutations.mode = tcb::PermutationMode::Selected;
  benchmark_case.variants.contraction_permutations.selected_einsums = {
      "ik,kj->ij",
      "ik,jk->ij",
  };
  benchmark_case.validation.reference_backend = "cpp:reference";
  benchmark_case.validation.tolerance = {1.0e-10, 1.0e-12};

  assert(benchmark_case.inputs.size() == 2);
  assert(benchmark_case.parameters.at("N").at(2) == 64);
  assert(benchmark_case.inputs.at("A").dims.at("i").is_parameter());
  assert(benchmark_case.inputs.at("A").dims.at("i").parameter_name() == "N");
  assert(benchmark_case.variants.output_permutations.selected.at(1).at(0) == "j");
  assert(benchmark_case.validation.tolerance.atol == 1.0e-12);
}

} // namespace

int main() {
  test_dtype_helpers();
  test_layout_helpers();
  test_tensor_view();
  test_benchmark_case_structs();
  return 0;
}
