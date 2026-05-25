#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "tcb/tensor_view.hpp"

namespace tcb {

template <typename T>
struct DenseTensor {
  DType dtype = DType::Float64;
  std::vector<Index> shape;
  std::vector<Index> strides;
  Labels labels;
  std::vector<T> values;

  TensorView view() {
    return TensorView{values.data(), dtype, shape, strides, labels};
  }

  const T &operator()(Index row, Index column) const {
    return values[static_cast<std::size_t>(row * strides.at(0) + column * strides.at(1))];
  }

  T &operator()(Index row, Index column) {
    return values[static_cast<std::size_t>(row * strides.at(0) + column * strides.at(1))];
  }
};

inline DenseTensor<double> make_row_major_matrix(Labels labels, Index rows, Index columns) {
  DenseTensor<double> tensor;
  tensor.dtype = DType::Float64;
  tensor.shape = {rows, columns};
  tensor.strides = {columns, 1};
  tensor.labels = std::move(labels);
  tensor.values.resize(static_cast<std::size_t>(rows * columns));
  return tensor;
}

} // namespace tcb
