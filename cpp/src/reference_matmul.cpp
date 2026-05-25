#include "tcb/reference_matmul.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

#include "tcb/dense_tensor.hpp"

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

void fill_inputs(DenseTensor<double> &a, DenseTensor<double> &b) {
  const Index n = a.shape.at(0);
  for (Index i = 0; i < n; ++i) {
    for (Index k = 0; k < n; ++k) {
      a(i, k) = deterministic_a(i, k);
    }
  }
  for (Index k = 0; k < n; ++k) {
    for (Index j = 0; j < n; ++j) {
      b(k, j) = deterministic_b(k, j);
    }
  }
}

void naive_matmul(const DenseTensor<double> &a, const DenseTensor<double> &b, DenseTensor<double> &c) {
  const Index n = a.shape.at(0);
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < n; ++j) {
      double sum = 0.0;
      for (Index k = 0; k < n; ++k) {
        sum += a(i, k) * b(k, j);
      }
      c(i, j) = sum;
    }
  }
}

void require_supported_case(const BenchmarkCase &benchmark_case, Index n, int warmup, int repeat) {
  if (benchmark_case.name != "matmul_square") {
    throw ReferenceMatmulError("reference matmul supports only matmul_square");
  }
  if (benchmark_case.dtype != DType::Float64) {
    throw ReferenceMatmulError("reference matmul supports only float64");
  }
  if (benchmark_case.einsum != "ik,kj->ij") {
    throw ReferenceMatmulError("reference matmul supports only ik,kj->ij");
  }
  if (n <= 0) {
    throw ReferenceMatmulError("matrix size N must be positive");
  }
  const auto parameter = benchmark_case.parameters.find("N");
  if (parameter == benchmark_case.parameters.end() ||
      std::find(parameter->second.begin(), parameter->second.end(), n) == parameter->second.end()) {
    throw ReferenceMatmulError("matrix size N is not listed in the benchmark case parameters");
  }
  if (warmup < 0) {
    throw ReferenceMatmulError("warmup must be non-negative");
  }
  if (repeat <= 0) {
    throw ReferenceMatmulError("repeat must be positive");
  }
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

BenchmarkResult run_reference_matmul_square(const BenchmarkCase &benchmark_case, Index n, int warmup, int repeat) {
  require_supported_case(benchmark_case, n, warmup, repeat);

  auto a = make_row_major_matrix({"i", "k"}, n, n);
  auto b = make_row_major_matrix({"k", "j"}, n, n);
  auto c = make_row_major_matrix({"i", "j"}, n, n);
  fill_inputs(a, b);

  for (int iteration = 0; iteration < warmup; ++iteration) {
    naive_matmul(a, b, c);
  }

  std::vector<double> times_sec;
  times_sec.reserve(static_cast<std::size_t>(repeat));
  for (int iteration = 0; iteration < repeat; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    naive_matmul(a, b, c);
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
  result.einsum = benchmark_case.einsum;
  result.backend = "cpp:reference";
  result.language = Language::Cpp;
  result.device = Device::Cpu;
  result.dtype = DType::Float64;
  result.parameters = {{"N", n}};
  result.input_layouts = {
      {"A", concrete_layout({"i", "k"}, {n, 1})},
      {"B", concrete_layout({"k", "j"}, {n, 1})},
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
  result.system.library_version = {{"reference", "unknown"}};
  return result;
}

} // namespace tcb
