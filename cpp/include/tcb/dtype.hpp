#pragma once

#include <cstddef>
#include <optional>
#include <string_view>

namespace tcb {

enum class DType {
  Float32,
  Float64,
  Complex64,
  Complex128,
};

inline std::string_view dtype_name(DType dtype) {
  switch (dtype) {
  case DType::Float32:
    return "float32";
  case DType::Float64:
    return "float64";
  case DType::Complex64:
    return "complex64";
  case DType::Complex128:
    return "complex128";
  }
  return "unknown";
}

inline std::optional<DType> parse_dtype(std::string_view name) {
  if (name == "float32") {
    return DType::Float32;
  }
  if (name == "float64") {
    return DType::Float64;
  }
  if (name == "complex64") {
    return DType::Complex64;
  }
  if (name == "complex128") {
    return DType::Complex128;
  }
  return std::nullopt;
}

inline std::size_t dtype_size_bytes(DType dtype) {
  switch (dtype) {
  case DType::Float32:
    return 4;
  case DType::Float64:
    return 8;
  case DType::Complex64:
    return 8;
  case DType::Complex128:
    return 16;
  }
  return 0;
}

} // namespace tcb
