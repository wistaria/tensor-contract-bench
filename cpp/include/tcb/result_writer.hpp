#pragma once

#include <filesystem>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

#include "tcb/result.hpp"

namespace tcb {

class ResultWriteError : public std::runtime_error {
public:
  explicit ResultWriteError(const std::string &message) : std::runtime_error(message) {}
};

std::string result_to_json_line(const BenchmarkResult &result);

void write_result_jsonl(std::ostream &output, const BenchmarkResult &result);

void write_result_jsonl(const std::filesystem::path &path, const std::vector<BenchmarkResult> &results);

} // namespace tcb
