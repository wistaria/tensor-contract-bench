#pragma once

#include <cstddef>
#include <numeric>
#include <vector>

#include "tcb/dtype.hpp"
#include "tcb/types.hpp"

namespace tcb {

struct TensorView {
  void *data = nullptr;
  DType dtype = DType::Float64;
  std::vector<Index> shape;
  std::vector<Index> strides;
  Labels labels;

  std::size_t rank() const { return shape.size(); }

  bool is_valid() const {
    if (data == nullptr || shape.empty() || shape.size() != strides.size() || shape.size() != labels.size()) {
      return false;
    }
    for (Index extent : shape) {
      if (extent <= 0) {
        return false;
      }
    }
    for (Index stride : strides) {
      if (stride <= 0) {
        return false;
      }
    }
    return true;
  }

  Index element_count() const {
    if (shape.empty()) {
      return 0;
    }
    return std::accumulate(shape.begin(), shape.end(), Index{1},
                           [](Index product, Index extent) { return product * extent; });
  }

  std::size_t element_size_bytes() const { return dtype_size_bytes(dtype); }
};

} // namespace tcb
