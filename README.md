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

## C++ benchmarks

Configure and build the C++ runner:

```sh
cmake -S . -B build
cmake --build build
```

Run the reference backend for all selected square-matmul contraction variants:

```sh
build/cpp/tcb-run \
  --case cases/matmul/square.yaml \
  --backend cpp:reference \
  --all-einsums \
  --N 16 \
  --warmup 1 \
  --repeat 5 \
  --output results/reference_matmul_square.jsonl
```

Every output row is JSONL and should validate against
`schemas/result.schema.json`.

## TBLIS backend

TBLIS support is optional and is enabled only when configuring with
`TCB_ENABLE_TBLIS=ON`. Install or build TBLIS first, then point CMake at that
installation with `CMAKE_PREFIX_PATH`.

One local source-build workflow is:

```sh
git clone https://github.com/MatthewsResearchGroup/tblis.git build/_deps/tblis-src
git -C build/_deps/tblis-src submodule update --init --recursive

cd build/_deps/tblis-src
./configure \
  --prefix="$OLDPWD/build/tblis-install" \
  --with-blis-config-family=generic \
  --disable-hwloc
cmake --build . --parallel 4
cmake --install .
cd -
```

Then configure this project with TBLIS enabled:

```sh
cmake -S . -B build \
  -DTCB_ENABLE_TBLIS=ON \
  -DCMAKE_PREFIX_PATH="$PWD/build/tblis-install"
cmake --build build
```

Run the TBLIS backend:

```sh
mkdir -p results

build/cpp/tcb-run \
  --case cases/matmul/square.yaml \
  --backend cpp:tblis \
  --all-einsums \
  --N 16 \
  --warmup 1 \
  --repeat 5 \
  --output results/tblis_matmul_square.jsonl
```

Validate the result file:

```sh
python3 tools/validate_schemas.py \
  --kind result \
  results/tblis_matmul_square.jsonl
```

Current C++ TBLIS coverage is `matmul_square` for the selected einsum variants
listed in `cases/matmul/square.yaml`. Rank-3 and rank-4 contractions are later
Milestone 1 work.
