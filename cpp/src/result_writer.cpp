#include "tcb/result_writer.hpp"

#include <fstream>

#include <nlohmann/json.hpp>

namespace tcb {
namespace {

nlohmann::json labels_to_json(const Labels &labels) {
  return labels;
}

nlohmann::json layout_to_json(const ConcreteLayout &layout) {
  return {
      {"labels", labels_to_json(layout.labels)},
      {"physical_order", labels_to_json(layout.physical_order)},
      {"strides", layout.strides},
  };
}

nlohmann::json timing_to_json(const TimingResult &timing) {
  return {
      {"warmup", timing.warmup},
      {"repeat", timing.repeat},
      {"times_sec", timing.times_sec},
      {"min_sec", timing.min_sec},
      {"median_sec", timing.median_sec},
      {"mean_sec", timing.mean_sec},
      {"stddev_sec", timing.stddev_sec},
  };
}

nlohmann::json performance_to_json(const PerformanceResult &performance) {
  nlohmann::json json = {
      {"flops", performance.flops},
      {"gflops", performance.gflops},
  };
  if (performance.bytes_estimated.has_value()) {
    json["bytes_estimated"] = *performance.bytes_estimated;
  }
  if (performance.bandwidth_gbs_estimated.has_value()) {
    json["bandwidth_gbs_estimated"] = *performance.bandwidth_gbs_estimated;
  }
  return json;
}

nlohmann::json validation_to_json(const ValidationResult &validation) {
  nlohmann::json json = {
      {"status", validation_status_name(validation.status)},
  };
  if (validation.max_abs_error.has_value()) {
    json["max_abs_error"] = *validation.max_abs_error;
  }
  if (validation.max_rel_error.has_value()) {
    json["max_rel_error"] = *validation.max_rel_error;
  }
  return json;
}

nlohmann::json system_to_json(const SystemInfo &system) {
  nlohmann::json json = {
      {"cpu", system.cpu},
      {"gpu", system.gpu.has_value() ? nlohmann::json(*system.gpu) : nlohmann::json(nullptr)},
      {"num_threads", system.num_threads},
  };
  if (system.compiler.has_value()) {
    json["compiler"] = *system.compiler;
  }
  if (system.compiler_version.has_value()) {
    json["compiler_version"] = *system.compiler_version;
  }
  if (!system.library_version.empty()) {
    nlohmann::json libraries = nlohmann::json::object();
    for (const auto &[name, version] : system.library_version) {
      libraries[name] = version.has_value() ? nlohmann::json(*version) : nlohmann::json(nullptr);
    }
    json["library_version"] = libraries;
  }
  return json;
}

nlohmann::json result_to_json(const BenchmarkResult &result) {
  nlohmann::json input_layouts = nlohmann::json::object();
  for (const auto &[name, layout] : result.input_layouts) {
    input_layouts[name] = layout_to_json(layout);
  }

  return {
      {"case_name", result.case_name},
      {"einsum", result.einsum},
      {"backend", result.backend},
      {"language", language_name(result.language)},
      {"device", device_name(result.device)},
      {"dtype", dtype_name(result.dtype)},
      {"parameters", result.parameters},
      {"input_layouts", input_layouts},
      {"output_layout", layout_to_json(result.output_layout)},
      {"timing", timing_to_json(result.timing)},
      {"performance", performance_to_json(result.performance)},
      {"validation", validation_to_json(result.validation)},
      {"system", system_to_json(result.system)},
  };
}

} // namespace

std::string language_name(Language language) {
  switch (language) {
  case Language::Cpp:
    return "cpp";
  case Language::Python:
    return "python";
  case Language::Julia:
    return "julia";
  case Language::Rust:
    return "rust";
  }
  return "cpp";
}

std::string device_name(Device device) {
  switch (device) {
  case Device::Cpu:
    return "cpu";
  case Device::Gpu:
    return "gpu";
  }
  return "cpu";
}

std::string validation_status_name(ValidationStatus status) {
  switch (status) {
  case ValidationStatus::Passed:
    return "passed";
  case ValidationStatus::Failed:
    return "failed";
  case ValidationStatus::Skipped:
    return "skipped";
  }
  return "skipped";
}

std::string result_to_json_line(const BenchmarkResult &result) {
  return result_to_json(result).dump();
}

void write_result_jsonl(std::ostream &output, const BenchmarkResult &result) {
  output << result_to_json_line(result) << '\n';
  if (!output) {
    throw ResultWriteError("failed to write result JSONL record");
  }
}

void write_result_jsonl(const std::filesystem::path &path, const std::vector<BenchmarkResult> &results) {
  std::ofstream output(path);
  if (!output) {
    throw ResultWriteError("failed to open result JSONL file: " + path.string());
  }
  for (const auto &result : results) {
    write_result_jsonl(output, result);
  }
}

} // namespace tcb
