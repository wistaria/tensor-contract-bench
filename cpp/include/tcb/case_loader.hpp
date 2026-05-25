#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

#include "tcb/benchmark_case.hpp"

namespace tcb {

class CaseLoadError : public std::runtime_error {
public:
  explicit CaseLoadError(const std::string &message) : std::runtime_error(message) {}
};

BenchmarkCase load_benchmark_case(const std::filesystem::path &path);

void validate_benchmark_case(const BenchmarkCase &benchmark_case);

} // namespace tcb
