#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "tcb/dtype.hpp"
#include "tcb/types.hpp"

namespace tcb {

struct ConcreteLayout {
  Labels labels;
  Labels physical_order;
  std::vector<Index> strides;
};

enum class Language {
  Cpp,
  Python,
  Julia,
  Rust,
};

enum class Device {
  Cpu,
  Gpu,
};

enum class ValidationStatus {
  Passed,
  Failed,
  Skipped,
};

struct TimingResult {
  int warmup = 0;
  int repeat = 1;
  std::vector<double> times_sec;
  double min_sec = 0.0;
  double median_sec = 0.0;
  double mean_sec = 0.0;
  double stddev_sec = 0.0;
};

struct PerformanceResult {
  std::int64_t flops = 0;
  double gflops = 0.0;
  std::optional<std::int64_t> bytes_estimated;
  std::optional<double> bandwidth_gbs_estimated;
};

struct ValidationResult {
  ValidationStatus status = ValidationStatus::Skipped;
  std::optional<double> max_abs_error;
  std::optional<double> max_rel_error;
};

struct SystemInfo {
  std::string cpu = "unknown";
  std::optional<std::string> gpu;
  int num_threads = 1;
  std::optional<std::string> compiler;
  std::optional<std::string> compiler_version;
  std::map<std::string, std::optional<std::string>> library_version;
};

struct BenchmarkResult {
  std::string case_name;
  std::string einsum;
  std::string backend;
  Language language = Language::Cpp;
  Device device = Device::Cpu;
  DType dtype = DType::Float64;
  std::map<std::string, Index> parameters;
  std::map<std::string, ConcreteLayout> input_layouts;
  ConcreteLayout output_layout;
  TimingResult timing;
  PerformanceResult performance;
  ValidationResult validation;
  SystemInfo system;
};

std::string language_name(Language language);
std::string device_name(Device device);
std::string validation_status_name(ValidationStatus status);

} // namespace tcb
