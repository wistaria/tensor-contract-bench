#include "tcb/case_loader.hpp"

#include <algorithm>
#include <set>
#include <string_view>

#include <yaml-cpp/yaml.h>

namespace tcb {
namespace {

template <typename T>
T require_scalar(const YAML::Node &node, const std::string &path) {
  if (!node || !node.IsScalar()) {
    throw CaseLoadError(path + ": expected scalar");
  }
  try {
    return node.as<T>();
  } catch (const YAML::Exception &error) {
    throw CaseLoadError(path + ": " + error.what());
  }
}

YAML::Node require_node(const YAML::Node &node, const std::string &path) {
  if (!node) {
    throw CaseLoadError(path + ": missing required field");
  }
  return node;
}

YAML::Node require_map(const YAML::Node &node, const std::string &path) {
  if (!node || !node.IsMap()) {
    throw CaseLoadError(path + ": expected mapping");
  }
  return node;
}

YAML::Node require_sequence(const YAML::Node &node, const std::string &path) {
  if (!node || !node.IsSequence()) {
    throw CaseLoadError(path + ": expected sequence");
  }
  return node;
}

Labels parse_labels(const YAML::Node &node, const std::string &path) {
  Labels labels;
  for (const auto &item : require_sequence(node, path)) {
    labels.push_back(require_scalar<std::string>(item, path + "[]"));
  }
  return labels;
}

LayoutKind parse_required_layout_kind(const YAML::Node &node, const std::string &path) {
  const auto name = require_scalar<std::string>(node, path);
  auto kind = parse_layout_kind(name);
  if (!kind.has_value()) {
    throw CaseLoadError(path + ": unsupported layout kind: " + name);
  }
  return *kind;
}

DType parse_required_dtype(const YAML::Node &node, const std::string &path) {
  const auto name = require_scalar<std::string>(node, path);
  auto dtype = parse_dtype(name);
  if (!dtype.has_value()) {
    throw CaseLoadError(path + ": unsupported dtype: " + name);
  }
  return *dtype;
}

LayoutSpec parse_layout(const YAML::Node &node, const std::string &path) {
  require_map(node, path);
  LayoutSpec layout;
  layout.kind = parse_required_layout_kind(require_node(node["kind"], path + ".kind"), path + ".kind");
  layout.physical_order = parse_labels(require_node(node["physical_order"], path + ".physical_order"),
                                       path + ".physical_order");
  if (node["strides"]) {
    for (const auto &item : require_sequence(node["strides"], path + ".strides")) {
      layout.strides.push_back(require_scalar<Index>(item, path + ".strides[]"));
    }
  }
  return layout;
}

InputTensorSpec parse_input_tensor(const YAML::Node &node, const std::string &path) {
  require_map(node, path);
  InputTensorSpec tensor;
  tensor.labels = parse_labels(require_node(node["labels"], path + ".labels"), path + ".labels");
  for (const auto &item : require_map(require_node(node["dims"], path + ".dims"), path + ".dims")) {
    const auto label = item.first.as<std::string>();
    if (item.second.IsScalar()) {
      const auto raw_value = item.second.as<std::string>();
      try {
        const auto extent = item.second.as<Index>();
        tensor.dims[label] = Dimension::extent(extent);
      } catch (const YAML::BadConversion &) {
        tensor.dims[label] = Dimension::parameter(raw_value);
      }
    } else {
      throw CaseLoadError(path + ".dims." + label + ": expected scalar");
    }
  }
  tensor.layout = parse_layout(require_node(node["layout"], path + ".layout"), path + ".layout");
  return tensor;
}

OutputTensorSpec parse_output_tensor(const YAML::Node &node, const std::string &path) {
  require_map(node, path);
  OutputTensorSpec tensor;
  tensor.labels = parse_labels(require_node(node["labels"], path + ".labels"), path + ".labels");
  tensor.layout = parse_layout(require_node(node["layout"], path + ".layout"), path + ".layout");
  return tensor;
}

PermutationMode parse_permutation_mode(std::string_view value, const std::string &path) {
  if (value == "all") {
    return PermutationMode::All;
  }
  if (value == "selected") {
    return PermutationMode::Selected;
  }
  if (value == "none") {
    return PermutationMode::None;
  }
  throw CaseLoadError(path + ": unsupported permutation mode");
}

LabelPermutationPolicy parse_label_permutation_policy(const YAML::Node &node, const std::string &path) {
  LabelPermutationPolicy policy;
  if (node.IsScalar()) {
    policy.mode = parse_permutation_mode(node.as<std::string>(), path);
    return policy;
  }
  policy.mode = PermutationMode::Selected;
  for (const auto &item : require_sequence(node, path)) {
    policy.selected.push_back(parse_labels(item, path + "[]"));
  }
  return policy;
}

ContractionPermutationPolicy parse_contraction_permutation_policy(const YAML::Node &node,
                                                                  const std::string &path) {
  ContractionPermutationPolicy policy;
  if (node.IsScalar()) {
    policy.mode = parse_permutation_mode(node.as<std::string>(), path);
    return policy;
  }
  policy.mode = PermutationMode::Selected;
  for (const auto &item : require_sequence(node, path)) {
    policy.selected_einsums.push_back(require_scalar<std::string>(item, path + "[]"));
  }
  return policy;
}

CaseVariants parse_variants(const YAML::Node &node, const std::string &path) {
  require_map(node, path);
  CaseVariants variants;
  variants.output_permutations = parse_label_permutation_policy(
      require_node(node["output_permutations"], path + ".output_permutations"), path + ".output_permutations");
  variants.input_permutations = parse_label_permutation_policy(
      require_node(node["input_permutations"], path + ".input_permutations"), path + ".input_permutations");
  variants.contraction_permutations = parse_contraction_permutation_policy(
      require_node(node["contraction_permutations"], path + ".contraction_permutations"),
      path + ".contraction_permutations");
  return variants;
}

ValidationSpec parse_validation(const YAML::Node &node, const std::string &path) {
  require_map(node, path);
  ValidationSpec validation;
  validation.reference_backend =
      require_scalar<std::string>(require_node(node["reference_backend"], path + ".reference_backend"),
                                  path + ".reference_backend");
  const auto tolerance = require_map(require_node(node["tolerance"], path + ".tolerance"), path + ".tolerance");
  validation.tolerance.rtol =
      require_scalar<double>(require_node(tolerance["rtol"], path + ".tolerance.rtol"), path + ".tolerance.rtol");
  validation.tolerance.atol =
      require_scalar<double>(require_node(tolerance["atol"], path + ".tolerance.atol"), path + ".tolerance.atol");
  return validation;
}

Labels labels_from_einsum_term(std::string_view term);

std::vector<Labels> parse_einsum_inputs(const std::string &einsum) {
  const auto arrow = einsum.find("->");
  if (arrow == std::string::npos) {
    throw CaseLoadError("einsum: missing ->");
  }

  std::vector<Labels> inputs;
  std::string current;
  for (std::size_t index = 0; index < arrow; ++index) {
    const char ch = einsum[index];
    if (ch == ',') {
      if (current.empty()) {
        throw CaseLoadError("einsum: empty input term");
      }
      inputs.push_back(labels_from_einsum_term(current));
      current.clear();
    } else {
      current.push_back(ch);
    }
  }
  if (current.empty()) {
    throw CaseLoadError("einsum: empty input term");
  }
  inputs.push_back(labels_from_einsum_term(current));
  return inputs;
}

Labels labels_from_einsum_term(std::string_view term) {
  Labels labels;
  labels.reserve(term.size());
  for (char label : term) {
    if (label == ',' || label == '-' || label == '>') {
      throw CaseLoadError("einsum: unexpected separator in term");
    }
    labels.emplace_back(1, label);
  }
  return labels;
}

Labels parse_einsum_output(const std::string &einsum) {
  const auto arrow = einsum.find("->");
  if (arrow == std::string::npos || arrow + 2 >= einsum.size()) {
    throw CaseLoadError("einsum: missing output term");
  }
  return labels_from_einsum_term(std::string_view(einsum).substr(arrow + 2));
}

bool same_items(Labels left, Labels right) {
  std::sort(left.begin(), left.end());
  std::sort(right.begin(), right.end());
  return left == right;
}

void require_unique_labels(const Labels &labels, const std::string &path) {
  const std::set<Label> unique(labels.begin(), labels.end());
  if (unique.size() != labels.size()) {
    throw CaseLoadError(path + ": labels must be unique");
  }
}

void validate_layout(const LayoutSpec &layout, const Labels &labels, const std::string &path) {
  require_unique_labels(layout.physical_order, path + ".physical_order");
  if (!same_items(layout.physical_order, labels)) {
    throw CaseLoadError(path + ".physical_order: must permute tensor labels");
  }
  if (layout.kind == LayoutKind::Strided) {
    if (layout.strides.size() != labels.size()) {
      throw CaseLoadError(path + ".strides: strided layout requires one stride per label");
    }
  } else if (!layout.strides.empty()) {
    throw CaseLoadError(path + ".strides: only strided layouts may specify strides");
  }
  for (Index stride : layout.strides) {
    if (stride <= 0) {
      throw CaseLoadError(path + ".strides: strides must be positive");
    }
  }
}

void validate_dimension(const Dimension &dimension, const std::set<std::string> &parameters,
                        const std::string &path) {
  if (dimension.is_extent()) {
    if (dimension.extent_value() <= 0) {
      throw CaseLoadError(path + ": concrete extent must be positive");
    }
    return;
  }
  if (parameters.count(dimension.parameter_name()) == 0) {
    throw CaseLoadError(path + ": unknown parameter " + dimension.parameter_name());
  }
}

} // namespace

BenchmarkCase load_benchmark_case(const std::filesystem::path &path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path.string());
  } catch (const YAML::Exception &error) {
    throw CaseLoadError(path.string() + ": " + error.what());
  }

  require_map(root, "$");

  BenchmarkCase benchmark_case;
  benchmark_case.name = require_scalar<std::string>(require_node(root["name"], "$.name"), "$.name");
  benchmark_case.description =
      require_scalar<std::string>(require_node(root["description"], "$.description"), "$.description");
  benchmark_case.dtype = parse_required_dtype(require_node(root["dtype"], "$.dtype"), "$.dtype");
  benchmark_case.einsum = require_scalar<std::string>(require_node(root["einsum"], "$.einsum"), "$.einsum");

  for (const auto &item : require_map(require_node(root["inputs"], "$.inputs"), "$.inputs")) {
    const auto name = item.first.as<std::string>();
    benchmark_case.inputs[name] = parse_input_tensor(item.second, "$.inputs." + name);
  }

  benchmark_case.output = parse_output_tensor(require_node(root["output"], "$.output"), "$.output");

  for (const auto &item : require_map(require_node(root["parameters"], "$.parameters"), "$.parameters")) {
    const auto name = item.first.as<std::string>();
    for (const auto &value : require_sequence(item.second, "$.parameters." + name)) {
      const auto extent = require_scalar<Index>(value, "$.parameters." + name + "[]");
      if (extent <= 0) {
        throw CaseLoadError("$.parameters." + name + "[]: extent must be positive");
      }
      benchmark_case.parameters[name].push_back(extent);
    }
  }

  benchmark_case.variants = parse_variants(require_node(root["variants"], "$.variants"), "$.variants");
  benchmark_case.validation = parse_validation(require_node(root["validation"], "$.validation"), "$.validation");

  validate_benchmark_case(benchmark_case);
  return benchmark_case;
}

void validate_benchmark_case(const BenchmarkCase &benchmark_case) {
  if (benchmark_case.name.empty()) {
    throw CaseLoadError("name: must not be empty");
  }
  if (benchmark_case.description.empty()) {
    throw CaseLoadError("description: must not be empty");
  }
  if (benchmark_case.inputs.empty()) {
    throw CaseLoadError("inputs: must not be empty");
  }

  std::set<std::string> parameter_names;
  for (const auto &[name, values] : benchmark_case.parameters) {
    if (values.empty()) {
      throw CaseLoadError("parameters." + name + ": must not be empty");
    }
    parameter_names.insert(name);
  }

  const auto input_terms = parse_einsum_inputs(benchmark_case.einsum);
  const auto output_term = parse_einsum_output(benchmark_case.einsum);
  if (input_terms.size() != benchmark_case.inputs.size()) {
    throw CaseLoadError("einsum: input term count must match inputs");
  }

  std::size_t input_index = 0;
  for (const auto &[name, tensor] : benchmark_case.inputs) {
    require_unique_labels(tensor.labels, "inputs." + name + ".labels");
    if (tensor.labels != input_terms.at(input_index)) {
      throw CaseLoadError("inputs." + name + ".labels: must match einsum term order");
    }
    if (tensor.dims.size() != tensor.labels.size()) {
      throw CaseLoadError("inputs." + name + ".dims: keys must match labels");
    }
    for (const auto &label : tensor.labels) {
      const auto dim = tensor.dims.find(label);
      if (dim == tensor.dims.end()) {
        throw CaseLoadError("inputs." + name + ".dims: missing label " + label);
      }
      validate_dimension(dim->second, parameter_names, "inputs." + name + ".dims." + label);
    }
    validate_layout(tensor.layout, tensor.labels, "inputs." + name + ".layout");
    ++input_index;
  }

  require_unique_labels(benchmark_case.output.labels, "output.labels");
  if (benchmark_case.output.labels != output_term) {
    throw CaseLoadError("output.labels: must match einsum output term order");
  }
  validate_layout(benchmark_case.output.layout, benchmark_case.output.labels, "output.layout");

  if (benchmark_case.variants.contraction_permutations.mode == PermutationMode::Selected) {
    for (const auto &einsum : benchmark_case.variants.contraction_permutations.selected_einsums) {
      const auto variant_inputs = parse_einsum_inputs(einsum);
      const auto variant_output = parse_einsum_output(einsum);
      if (variant_inputs.size() != benchmark_case.inputs.size()) {
        throw CaseLoadError("variants.contraction_permutations: input term count must match inputs");
      }
      if (!same_items(variant_output, benchmark_case.output.labels)) {
        throw CaseLoadError("variants.contraction_permutations: output term must permute output labels");
      }
      std::size_t variant_input_index = 0;
      for (const auto &[name, tensor] : benchmark_case.inputs) {
        if (!same_items(variant_inputs.at(variant_input_index), tensor.labels)) {
          throw CaseLoadError("variants.contraction_permutations: term for " + name +
                              " must permute input labels");
        }
        ++variant_input_index;
      }
    }
  }
}

} // namespace tcb
