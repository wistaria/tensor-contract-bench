# tensor-contract-bench
A tensor contraction benchmark suite.

This project is a benchmark suite for systematically comparing the performance
of tensor contraction libraries in C++, Python, Julia, and Rust, including
differences between CPU/GPU execution and memory layouts.

The main target contractions are:

1. Products of square matrices
2. Contractions between rank-3 tensors
3. Contractions between rank-4 tensors
4. Cases with uniform bond dimensions
5. Cases with highly non-uniform bond dimensions
6. Cases with different input and output index orders
7. Cases with row-major, column-major, and strided layouts
8. Differences between CPU and GPU backends

The goal is not only to measure FLOP/s, but also to measure how index order,
output layout, strides, and backend choices affect performance for the same
mathematical contraction.
