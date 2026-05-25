#include "tcb/tblis_matmul.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include "tcb/dense_tensor.hpp"

#define TBLIS_DISABLE_CPLUSPLUS 1
#include <tblis/tblis.h>

#ifdef __APPLE__
#define TCB_TBLIS_SYMBOL(name) "_" name
#else
#define TCB_TBLIS_SYMBOL(name) name
#endif

extern "C" void tcb_tblis_init_tensor_d(tblis_tensor *t, int ndim, len_type *len, double *data, stride_type *stride)
    __asm(TCB_TBLIS_SYMBOL("tblis_init_tensor_d"));
extern "C" void tcb_tblis_init_tensor_scaled_d(tblis_tensor *t, double scalar, int ndim, len_type *len, double *data,
                                               stride_type *stride) __asm(TCB_TBLIS_SYMBOL("tblis_init_tensor_scaled_d"));
extern "C" void tcb_tblis_tensor_mult(const tblis_comm *comm, const tblis_config *cntx, const tblis_tensor *a,
                                      const label_type *idx_a, const tblis_tensor *b, const label_type *idx_b,
                                      tblis_tensor *c, const label_type *idx_c)
    __asm(TCB_TBLIS_SYMBOL("tblis_tensor_mult"));

namespace tcb {
namespace {

double deterministic_a(Index i, Index k) {
  return static_cast<double>(i + 1) * 0.125 + static_cast<double>(k + 1) * 0.03125;
}

double deterministic_b(Index k, Index j) {
  return static_cast<double>(k + 1) * 0.0625 - static_cast<double>(j + 1) * 0.015625;
}

double expected_c(Index i, Index j, Index n) {
  double sum = 0.0;
  for (Index k = 0; k < n; ++k) {
    sum += deterministic_a(i, k) * deterministic_b(k, j);
  }
  return sum;
}

std::vector<std::string> input_terms(const std::string &einsum) {
  const auto arrow = einsum.find("->");
  const auto comma = einsum.find(',');
  if (arrow == std::string::npos || comma == std::string::npos || comma > arrow) {
    throw TblisMatmulError("unsupported matmul einsum: " + einsum);
  }
  return {einsum.substr(0, comma), einsum.substr(comma + 1, arrow - comma - 1)};
}

bool case_contains_variant(const BenchmarkCase &benchmark_case, const std::string &einsum) {
  if (benchmark_case.variants.contraction_permutations.mode != PermutationMode::Selected) {
    return benchmark_case.einsum == einsum;
  }
  const auto &variants = benchmark_case.variants.contraction_permutations.selected_einsums;
  return std::find(variants.begin(), variants.end(), einsum) != variants.end();
}

Index axis_for_label(const Labels &labels, const std::string &label) {
  const auto found = std::find(labels.begin(), labels.end(), label);
  if (found == labels.end()) {
    throw TblisMatmulError("missing label in matmul input: " + label);
  }
  return static_cast<Index>(std::distance(labels.begin(), found));
}

double get_matrix_value(const DenseTensor<double> &matrix, const std::string &row_label, Index row_index,
                        const std::string &column_label, Index column_index) {
  const Index row_axis = axis_for_label(matrix.labels, row_label);
  const Index column_axis = axis_for_label(matrix.labels, column_label);
  Index indices[2] = {0, 0};
  indices[row_axis] = row_index;
  indices[column_axis] = column_index;
  return matrix(indices[0], indices[1]);
}

void set_matrix_value(DenseTensor<double> &matrix, const std::string &row_label, Index row_index,
                      const std::string &column_label, Index column_index, double value) {
  const Index row_axis = axis_for_label(matrix.labels, row_label);
  const Index column_axis = axis_for_label(matrix.labels, column_label);
  Index indices[2] = {0, 0};
  indices[row_axis] = row_index;
  indices[column_axis] = column_index;
  matrix(indices[0], indices[1]) = value;
}

void fill_inputs(DenseTensor<double> &a, DenseTensor<double> &b) {
  const Index n = a.shape.at(0);
  for (Index i = 0; i < n; ++i) {
    for (Index k = 0; k < n; ++k) {
      set_matrix_value(a, "i", i, "k", k, deterministic_a(i, k));
    }
  }
  for (Index k = 0; k < n; ++k) {
    for (Index j = 0; j < n; ++j) {
      set_matrix_value(b, "k", k, "j", j, deterministic_b(k, j));
    }
  }
}

void require_supported_case(const BenchmarkCase &benchmark_case, const std::string &einsum, Index n, int warmup,
                            int repeat) {
  if (benchmark_case.name != "matmul_square") {
    throw TblisMatmulError("TBLIS matmul supports only matmul_square");
  }
  if (benchmark_case.dtype != DType::Float64) {
    throw TblisMatmulError("TBLIS matmul supports only float64");
  }
  if (!case_contains_variant(benchmark_case, einsum)) {
    throw TblisMatmulError("einsum is not listed in the benchmark case variants: " + einsum);
  }
  if (einsum != "ik,kj->ij" && einsum != "ik,jk->ij" && einsum != "ki,kj->ij" && einsum != "ki,jk->ij") {
    throw TblisMatmulError("TBLIS matmul supports only matmul_square contraction variants");
  }
  if (n <= 0) {
    throw TblisMatmulError("matrix size N must be positive");
  }
  const auto parameter = benchmark_case.parameters.find("N");
  if (parameter == benchmark_case.parameters.end() ||
      std::find(parameter->second.begin(), parameter->second.end(), n) == parameter->second.end()) {
    throw TblisMatmulError("matrix size N is not listed in the benchmark case parameters");
  }
  if (warmup < 0) {
    throw TblisMatmulError("warmup must be non-negative");
  }
  if (repeat <= 0) {
    throw TblisMatmulError("repeat must be positive");
  }
}

std::vector<len_type> tblis_lengths(const DenseTensor<double> &tensor) {
  std::vector<len_type> lengths;
  lengths.reserve(tensor.shape.size());
  for (const auto length : tensor.shape) {
    lengths.push_back(static_cast<len_type>(length));
  }
  return lengths;
}

std::vector<stride_type> tblis_strides(const DenseTensor<double> &tensor) {
  std::vector<stride_type> strides;
  strides.reserve(tensor.strides.size());
  for (const auto stride : tensor.strides) {
    strides.push_back(static_cast<stride_type>(stride));
  }
  return strides;
}

std::string label_string(const DenseTensor<double> &tensor) {
  std::string labels;
  labels.reserve(tensor.labels.size());
  for (const auto &label : tensor.labels) {
    if (label.size() != 1) {
      throw TblisMatmulError("TBLIS labels must be single characters");
    }
    labels.push_back(label.front());
  }
  return labels;
}

void tblis_matmul(const DenseTensor<double> &a, const DenseTensor<double> &b, DenseTensor<double> &c) {
  auto a_lengths = tblis_lengths(a);
  auto b_lengths = tblis_lengths(b);
  auto c_lengths = tblis_lengths(c);
  auto a_strides = tblis_strides(a);
  auto b_strides = tblis_strides(b);
  auto c_strides = tblis_strides(c);
  const auto a_labels = label_string(a);
  const auto b_labels = label_string(b);
  const auto c_labels = label_string(c);

  typename std::aligned_storage<sizeof(tblis_tensor), alignof(tblis_tensor)>::type tensor_a_storage;
  typename std::aligned_storage<sizeof(tblis_tensor), alignof(tblis_tensor)>::type tensor_b_storage;
  typename std::aligned_storage<sizeof(tblis_tensor), alignof(tblis_tensor)>::type tensor_c_storage;
  auto *tensor_a = reinterpret_cast<tblis_tensor *>(&tensor_a_storage);
  auto *tensor_b = reinterpret_cast<tblis_tensor *>(&tensor_b_storage);
  auto *tensor_c = reinterpret_cast<tblis_tensor *>(&tensor_c_storage);

  tcb_tblis_init_tensor_d(tensor_a, static_cast<int>(a_lengths.size()), a_lengths.data(),
                          const_cast<double *>(a.values.data()), a_strides.data());
  tcb_tblis_init_tensor_d(tensor_b, static_cast<int>(b_lengths.size()), b_lengths.data(),
                          const_cast<double *>(b.values.data()), b_strides.data());
  tcb_tblis_init_tensor_scaled_d(tensor_c, 0.0, static_cast<int>(c_lengths.size()), c_lengths.data(), c.values.data(),
                                 c_strides.data());

  tcb_tblis_tensor_mult(nullptr, nullptr, tensor_a, a_labels.c_str(), tensor_b, b_labels.c_str(), tensor_c,
                        c_labels.c_str());
}

double mean(const std::vector<double> &values) {
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto mid = values.size() / 2;
  if (values.size() % 2 == 1) {
    return values[mid];
  }
  return (values[mid - 1] + values[mid]) * 0.5;
}

double stddev(const std::vector<double> &values, double average) {
  double sum_sq = 0.0;
  for (double value : values) {
    const double delta = value - average;
    sum_sq += delta * delta;
  }
  return std::sqrt(sum_sq / static_cast<double>(values.size()));
}

ConcreteLayout concrete_layout(Labels labels, std::vector<Index> strides) {
  return ConcreteLayout{labels, labels, strides};
}

std::int64_t matmul_flops(Index n) {
  return 2 * n * n * n;
}

std::int64_t matmul_bytes(Index n) {
  return 3 * n * n * static_cast<Index>(dtype_size_bytes(DType::Float64));
}

} // namespace

BenchmarkResult run_tblis_matmul_square(const BenchmarkCase &benchmark_case, const std::string &einsum, Index n,
                                        int warmup, int repeat) {
  require_supported_case(benchmark_case, einsum, n, warmup, repeat);

  const auto terms = input_terms(einsum);
  auto a = make_row_major_matrix({terms.at(0).substr(0, 1), terms.at(0).substr(1, 1)}, n, n);
  auto b = make_row_major_matrix({terms.at(1).substr(0, 1), terms.at(1).substr(1, 1)}, n, n);
  auto c = make_row_major_matrix({"i", "j"}, n, n);
  fill_inputs(a, b);

  for (int iteration = 0; iteration < warmup; ++iteration) {
    tblis_matmul(a, b, c);
  }

  std::vector<double> times_sec;
  times_sec.reserve(static_cast<std::size_t>(repeat));
  for (int iteration = 0; iteration < repeat; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    tblis_matmul(a, b, c);
    const auto stop = std::chrono::steady_clock::now();
    times_sec.push_back(std::chrono::duration<double>(stop - start).count());
  }

  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < n; ++j) {
      const double expected = expected_c(i, j, n);
      const double abs_error = std::abs(c(i, j) - expected);
      const double rel_error = abs_error / std::max(std::abs(expected), 1.0);
      max_abs_error = std::max(max_abs_error, abs_error);
      max_rel_error = std::max(max_rel_error, rel_error);
    }
  }

  const double average = mean(times_sec);
  const double min_time = *std::min_element(times_sec.begin(), times_sec.end());

  BenchmarkResult result;
  result.case_name = benchmark_case.name;
  result.einsum = einsum;
  result.backend = "cpp:tblis";
  result.language = Language::Cpp;
  result.device = Device::Cpu;
  result.dtype = DType::Float64;
  result.parameters = {{"N", n}};
  result.input_layouts = {
      {"A", concrete_layout(a.labels, {n, 1})},
      {"B", concrete_layout(b.labels, {n, 1})},
  };
  result.output_layout = concrete_layout({"i", "j"}, {n, 1});
  result.timing = {warmup, repeat, times_sec, min_time, median(times_sec), average, stddev(times_sec, average)};
  result.performance.flops = matmul_flops(n);
  result.performance.gflops = (static_cast<double>(result.performance.flops) / min_time) / 1.0e9;
  result.performance.bytes_estimated = matmul_bytes(n);
  result.performance.bandwidth_gbs_estimated =
      (static_cast<double>(*result.performance.bytes_estimated) / min_time) / 1.0e9;
  result.validation.status = (max_abs_error <= benchmark_case.validation.tolerance.atol &&
                              max_rel_error <= benchmark_case.validation.tolerance.rtol)
                                 ? ValidationStatus::Passed
                                 : ValidationStatus::Failed;
  result.validation.max_abs_error = max_abs_error;
  result.validation.max_rel_error = max_rel_error;
  result.system.cpu = "unknown";
  result.system.gpu = std::nullopt;
  result.system.num_threads = 1;
  result.system.compiler = "unknown";
  result.system.compiler_version = std::nullopt;
  result.system.library_version = {{"tblis", "2.0"}};
  return result;
}

} // namespace tcb
