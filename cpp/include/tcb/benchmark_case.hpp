#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "tcb/dtype.hpp"
#include "tcb/layout.hpp"
#include "tcb/types.hpp"

namespace tcb {

struct Dimension {
  std::variant<Index, std::string> value = Index{1};

  static Dimension extent(Index concrete_extent) { return Dimension{concrete_extent}; }
  static Dimension parameter(std::string parameter_name) { return Dimension{std::move(parameter_name)}; }

  bool is_extent() const { return std::holds_alternative<Index>(value); }
  bool is_parameter() const { return std::holds_alternative<std::string>(value); }

  Index extent_value() const { return std::get<Index>(value); }
  const std::string &parameter_name() const { return std::get<std::string>(value); }
};

struct InputTensorSpec {
  Labels labels;
  std::map<Label, Dimension> dims;
  LayoutSpec layout;
};

struct OutputTensorSpec {
  Labels labels;
  LayoutSpec layout;
};

enum class PermutationMode {
  All,
  Selected,
  None,
};

struct LabelPermutationPolicy {
  PermutationMode mode = PermutationMode::None;
  std::vector<Labels> selected;
};

struct ContractionPermutationPolicy {
  PermutationMode mode = PermutationMode::None;
  std::vector<std::string> selected_einsums;
};

struct CaseVariants {
  LabelPermutationPolicy output_permutations;
  LabelPermutationPolicy input_permutations;
  ContractionPermutationPolicy contraction_permutations;
};

struct Tolerance {
  double rtol = 0.0;
  double atol = 0.0;
};

struct ValidationSpec {
  std::string reference_backend;
  Tolerance tolerance;
};

struct BenchmarkCase {
  std::string name;
  std::string description;
  DType dtype = DType::Float64;
  std::string einsum;
  std::map<std::string, InputTensorSpec> inputs;
  OutputTensorSpec output;
  std::map<std::string, std::vector<Index>> parameters;
  CaseVariants variants;
  ValidationSpec validation;
};

} // namespace tcb
