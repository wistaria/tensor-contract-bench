#pragma once

#include <stdexcept>
#include <string>

#include "tcb/benchmark_case.hpp"
#include "tcb/result.hpp"

namespace tcb {

class ReferenceMatmulError : public std::runtime_error {
public:
  explicit ReferenceMatmulError(const std::string &message) : std::runtime_error(message) {}
};

BenchmarkResult run_reference_matmul_square(const BenchmarkCase &benchmark_case, const std::string &einsum, Index n,
                                            int warmup, int repeat);

} // namespace tcb
