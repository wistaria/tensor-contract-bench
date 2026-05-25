#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include "tcb/types.hpp"

namespace tcb {

enum class LayoutKind {
  RowMajor,
  ColumnMajor,
  Strided,
};

struct LayoutSpec {
  LayoutKind kind = LayoutKind::RowMajor;
  Labels physical_order;
  std::vector<Index> strides;

  bool has_explicit_strides() const { return !strides.empty(); }
};

inline std::string_view layout_kind_name(LayoutKind kind) {
  switch (kind) {
  case LayoutKind::RowMajor:
    return "row_major";
  case LayoutKind::ColumnMajor:
    return "column_major";
  case LayoutKind::Strided:
    return "strided";
  }
  return "unknown";
}

inline std::optional<LayoutKind> parse_layout_kind(std::string_view name) {
  if (name == "row_major") {
    return LayoutKind::RowMajor;
  }
  if (name == "column_major") {
    return LayoutKind::ColumnMajor;
  }
  if (name == "strided") {
    return LayoutKind::Strided;
  }
  return std::nullopt;
}

} // namespace tcb
