Tensor Contraction Benchmark Suite 仕様書案

1. 目的

本OSSは、C++、Python、Julia、Rustにおけるテンソル縮約ライブラリの性能を、CPU/GPUおよびメモリレイアウトの違いを含めて体系的に比較するためのベンチマークスイートである。

対象とする縮約は、主に以下である。

1. 正方行列同士の積
2. 3本足テンソル同士の縮約
3. 4本足テンソル同士の縮約
4. ボンド次元が一様な場合
5. ボンド次元が大きく異なる場合
6. 入力・出力の添字順序を変えた場合
7. row-major / column-major / strided layout を変えた場合
8. CPU / GPU backend の違い

単なる FLOPS 測定ではなく、同じ数学的縮約に対して、添字順序・出力レイアウト・stride・backend がどの程度性能に影響するかを測定することを目的とする。

⸻

2. プロジェクト名

候補：

tensor-contract-bench

短縮名：

tcb

コマンド名：

tcb-run
tcb-compare
tcb-generate

⸻

3. 対象言語

必須

* C++
* Python
* Julia

可能であれば対応

* Rust

⸻

4. 対象backend

4.1 C++ backend

必須：

* naive reference implementation
* BLAS / CBLAS
* TBLIS

推奨：

* Eigen
* xtensor
* OpenMP parallel reference

GPU候補：

* cuBLAS
* cuTENSOR
* oneDNN / oneMKL
* rocBLAS / hipBLAS

⸻

4.2 Python backend

必須：

* NumPy
* opt_einsum
* PyTorch
* JAX
* CuPy

推奨：

* pytblis
* TensorFlow
* cotengra

⸻

4.3 Julia backend

必須：

* LinearAlgebra
* TensorOperations.jl

推奨：

* Tullio.jl
* Einsum.jl
* OMEinsum.jl
* ITensors.jl
* TBLIS.jl または BliContractor.jl

⸻

4.4 Rust backend

候補：

* ndarray
* matrixmultiply
* tblis crate
* candle
* burn
* tch-rs

Rust対応は初期段階では experimental とする。

⸻

5. ベンチマーク対象

5.1 行列積

基本形：

C_{ij} = sum_k A_{ik} B_{kj}

ただし、以下の縮約をすべて対象にする。

名称	数式	意味
row-row	C_{ij} = A_{ik} B_{jk}	Aの行とBの行を縮約
row-col	C_{ij} = A_{ik} B_{kj}	通常の行列積
col-row	C_{ij} = A_{ki} B_{jk}	Aの列とBの行を縮約
col-col	C_{ij} = A_{ki} B_{kj}	Aの列とBの列を縮約

実際には、以下のような einsum で表現する。

ik,kj->ij
ik,jk->ij
ki,jk->ij
ki,kj->ij

さらに出力添字順序も変える。

ik,kj->ij
ik,kj->ji

5.2 行列サイズ

初期セット：

N = 16, 32, 64, 128, 256, 512, 1024, 2048, 4096

GPU向け追加：

N = 8192, 16384

小サイズ性能を見るための追加：

N = 2, 4, 8

⸻

6. 3本足テンソル縮約

6.1 基本形

3階テンソル同士の1本縮約：

C_{i j k l} = sum_a A_{i j a} B_{a k l}

einsum表記：

ija,akl->ijkl

対象とする縮約位置：

ija,akl->ijkl
iaj,kla->ijkl
aij,kal->ijkl
ija,kla->ijkl

一般には、Aの3本のうち1本、Bの3本のうち1本を縮約するため、

3 × 3 = 9通り

を対象にする。

6.2 出力添字順序

非縮約添字が4本あるので、

4! = 24通り

を原則として対象にする。

例：

ija,akl->ijkl
ija,akl->ijlk
ija,akl->ikjl
...

6.3 複数本縮約

3階テンソル同士の2本縮約：

C_{i l} = sum_{a b} A_{i a b} B_{a b l}

einsum表記：

iab,abl->il

縮約パターンは、

choose(3,2) × choose(3,2) × 2! = 18通り

を基本セットに含める。

⸻

7. 4本足テンソル縮約

7.1 1本縮約

A_{i j k a} B_{a l m n} -> C_{i j k l m n}

einsum：

ijka,almn->ijklmn

縮約位置：

4 × 4 = 16通り

出力添字数は6本なので、全順列は

6! = 720通り

ただし全実行は重すぎるため、初期実装では以下の代表セットに制限する。

* 自然順序
* 完全逆順
* A由来添字を前に置く
* B由来添字を前に置く
* interleaved order
* stride最悪化が期待される順序
* ランダム代表順序

7.2 2本縮約

典型例：

A_{i j a b} B_{a b k l} -> C_{i j k l}

einsum：

ijab,abkl->ijkl

縮約位置数：

choose(4,2) × choose(4,2) × 2! = 72通り

出力添字数は4本なので、

4! = 24通り

原則として全出力順序を対象にする。

7.3 3本縮約

典型例：

A_{i a b c} B_{a b c j} -> C_{i j}

einsum：

iabc,abcj->ij

縮約位置数：

choose(4,3) × choose(4,3) × 3! = 96通り

出力添字は2本なので、

2通り

を全て対象にする。

⸻

8. 次元設定

8.1 一様次元

すべての足の次元が等しいケース。

dimensions:
  type: uniform
  values:
    D: [2, 4, 8, 16, 32, 64, 128]

例：

A[D,D,D,D], B[D,D,D,D]

8.2 異方的次元

足ごとに大きく異なるケース。

例：

dimensions:
  type: anisotropic
  values:
    small: [2, 4]
    medium: [16, 32]
    large: [128, 256, 512]

代表パターン：

[small, small, large, large]
[small, large, small, large]
[large, small, large, small]
[medium, large, small, medium]

8.3 物理次元・ボンド次元型

テンソルネットワークで典型的な設定。

physical dimension: d = 2, 3, 4
bond dimension: χ = 8, 16, 32, 64, 128, 256

例：

A[d, χ, χ]
B[χ, χ, d]

または

A[d, χ1, χ2, χ3]
B[χ3, χ4, χ5, d]

⸻

9. データ型

必須：

float32
float64
complex64
complex128

初期実装では以下を優先する。

float64
float32

GPU backend では以下も optional。

float16
bfloat16
tf32

⸻

10. メモリレイアウト

10.1 基本layout

row_major
column_major

10.2 permuted layout

同じ論理テンソルに対して、物理メモリ上の軸順序を変える。

例：

logical_shape: [i, j, a, b]
physical_order: [i, j, a, b]

または

logical_shape: [i, j, a, b]
physical_order: [a, b, i, j]

10.3 strided view

コピーなし転置を模擬するため、stride付きviewをサポートする。

layout:
  kind: strided
  shape: [64, 64, 64, 64]
  strides: [262144, 4096, 64, 1]

10.4 non-contiguous input

以下を区別する。

contiguous input
transposed view
permuted view
sliced view
padded view

⸻

11. ベンチケース定義形式

ベンチケースは YAML で定義する。

例：

name: rank4_rank4_two_bond_uniform
description: "Rank-4 x Rank-4 tensor contraction with two contracted legs"
dtype: float64
inputs:
  A:
    labels: [i, j, a, b]
    dims:
      i: D
      j: D
      a: X
      b: X
    layout:
      kind: row_major
      physical_order: [i, j, a, b]
  B:
    labels: [a, b, k, l]
    dims:
      a: X
      b: X
      k: D
      l: D
    layout:
      kind: row_major
      physical_order: [a, b, k, l]
output:
  labels: [i, j, k, l]
  layout:
    kind: row_major
    physical_order: [i, j, k, l]
parameters:
  D: [8, 16, 32, 64]
  X: [8, 16, 32, 64, 128]
variants:
  output_permutations: all
  input_permutations: selected
  contraction_permutations: all
validation:
  reference_backend: numpy
  tolerance:
    rtol: 1.0e-10
    atol: 1.0e-12

⸻

12. 実行CLI

12.1 ケース生成

tcb-generate \
  --suite rank4-rank4 \
  --output cases/generated/rank4_rank4.yaml

12.2 ベンチ実行

tcb-run \
  --cases cases/generated/rank4_rank4.yaml \
  --backend cpp:tblis \
  --repeat 10 \
  --warmup 3 \
  --output results/tblis_rank4.jsonl

12.3 複数backend実行

tcb-run \
  --cases cases/generated/matmul.yaml \
  --backend cpp:blas \
  --backend cpp:tblis \
  --backend python:numpy \
  --backend python:torch \
  --backend julia:tensoroperations \
  --output results/all.jsonl

12.4 結果比較

tcb-compare \
  --input results/all.jsonl \
  --group-by case_name,dtype,backend \
  --metric median_time

⸻

13. 出力形式

結果は JSON Lines とする。

{
  "case_name": "rank4_rank4_two_bond_uniform",
  "einsum": "ijab,abkl->ijkl",
  "backend": "cpp:tblis",
  "language": "cpp",
  "device": "cpu",
  "dtype": "float64",
  "parameters": {
    "D": 32,
    "X": 64
  },
  "input_layouts": {
    "A": {
      "labels": ["i", "j", "a", "b"],
      "physical_order": ["i", "j", "a", "b"],
      "strides": [262144, 8192, 64, 1]
    },
    "B": {
      "labels": ["a", "b", "k", "l"],
      "physical_order": ["a", "b", "k", "l"],
      "strides": [131072, 2048, 64, 1]
    }
  },
  "output_layout": {
    "labels": ["i", "j", "k", "l"],
    "physical_order": ["i", "j", "k", "l"],
    "strides": [32768, 1024, 32, 1]
  },
  "timing": {
    "warmup": 3,
    "repeat": 10,
    "times_sec": [0.0131, 0.0129, 0.0130],
    "min_sec": 0.0129,
    "median_sec": 0.0130,
    "mean_sec": 0.0130
  },
  "performance": {
    "flops": 8589934592,
    "gflops": 660.8,
    "bytes_estimated": 100663296,
    "bandwidth_gbs_estimated": 7.74
  },
  "validation": {
    "status": "passed",
    "max_abs_error": 1.2e-12,
    "max_rel_error": 3.4e-13
  },
  "system": {
    "cpu": "unknown",
    "gpu": null,
    "num_threads": 16,
    "compiler": "gcc",
    "compiler_version": "13.2",
    "library_version": {
      "tblis": "unknown"
    }
  }
}

⸻

14. 性能指標

必須：

min time
median time
mean time
standard deviation
GFLOP/s

推奨：

estimated memory bandwidth
arithmetic intensity
temporary allocation size
peak memory usage

縮約のFLOP数は原則として、

2 × 出力要素数 × 縮約空間サイズ

複素数の場合は別途扱う。

⸻

15. 検証

15.1 correctness check

各backendの出力を reference backend と比較する。

初期reference：

Python NumPy einsum

C++単体の場合：

naive reference implementation

15.2 許容誤差

float32:
  rtol: 1.0e-5
  atol: 1.0e-6
float64:
  rtol: 1.0e-10
  atol: 1.0e-12
complex64:
  rtol: 1.0e-5
  atol: 1.0e-6
complex128:
  rtol: 1.0e-10
  atol: 1.0e-12

15.3 validation mode

tcb-run --validate always
tcb-run --validate sample
tcb-run --validate none

⸻

16. TBLIS backend仕様

16.1 C++ TBLIS backend

TBLIS backend は以下をサポートする。

arbitrary rank tensor contraction
arbitrary stride
arbitrary label order
float / double
complex float / complex double
multi-threading

16.2 TBLIS呼び出し方針

内部表現：

TensorView {
    void* data;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::vector<char> labels;
    DType dtype;
}

TBLISには、添字文字列とstride情報を渡す。

概念的には：

tblis_tensor A;
tblis_tensor B;
tblis_tensor C;
tblis_init_tensor_scaled(..., &A, ...);
tblis_init_tensor_scaled(..., &B, ...);
tblis_init_tensor(..., &C, ...);
tblis_tensor_mult(
    nullptr,
    nullptr,
    &A, "ijab",
    &B, "abkl",
    &C, "ijkl"
);

16.3 TBLIS固有測定項目

num_threads
block size
TBLIS version
BLIS version if available
environment variables

⸻

17. ディレクトリ構成

tensor-contract-bench/
  README.md
  LICENSE
  CMakeLists.txt
  pyproject.toml
  Project.toml
  Cargo.toml
  docs/
    spec.md
    benchmark_cases.md
    backend_api.md
    result_schema.md
  cases/
    matmul/
      square.yaml
      layout_variants.yaml
    rank3/
      one_bond.yaml
      two_bond.yaml
    rank4/
      one_bond.yaml
      two_bond.yaml
      three_bond.yaml
    generated/
  schemas/
    case.schema.json
    result.schema.json
  cpp/
    include/tcb/
      tensor_view.hpp
      benchmark_case.hpp
      backend.hpp
      timer.hpp
      validation.hpp
    src/
      runner.cpp
      case_loader.cpp
      reference_backend.cpp
    backends/
      blas/
      tblis/
      eigen/
  python/
    tcb/
      runner.py
      case_loader.py
      backends/
        numpy_backend.py
        torch_backend.py
        cupy_backend.py
        jax_backend.py
        pytblis_backend.py
  julia/
    TCBench/
      src/
        TCBench.jl
        backends/
          TensorOperationsBackend.jl
          TBLISBackend.jl
  rust/
    tcb-rs/
      src/
        main.rs
        backends/
  tools/
    generate_cases.py
    compare_results.py
    plot_results.py
  results/
    .gitkeep

⸻

18. 初期マイルストーン

Milestone 0: 仕様固定

* YAML case schema
* JSONL result schema
* benchmark naming rule
* dtype rule
* layout rule

Milestone 1: C++最小実装

対象：

matmul
rank3 × rank3 one-bond contraction
rank4 × rank4 two-bond contraction

backend：

naive
BLAS
TBLIS

Milestone 2: Python runner

backend：

numpy
torch
cupy
jax
opt_einsum

Milestone 3: Julia runner

backend：

LinearAlgebra
TensorOperations.jl
Tullio.jl
TBLIS.jl

Milestone 4: GPU backend

backend：

cuBLAS
cuTENSOR
CuPy
PyTorch CUDA
JAX CUDA

Milestone 5: Rust backend

backend：

ndarray
matrixmultiply
tblis crate

⸻

19. 最初に実装すべきベンチケース

19.1 matmul-basic

ik,kj->ij
ik,jk->ij
ki,kj->ij
ki,jk->ij

サイズ：

N = 16, 32, 64, 128, 256, 512, 1024

19.2 rank3-one-bond

ija,akl->ijkl
iaj,kla->ijkl
aij,kla->ijkl

サイズ：

D = 4, 8, 16, 32
X = 4, 8, 16, 32, 64

19.3 rank4-two-bond

ijab,abkl->ijkl
iabj,kabl->ijkl
abij,klab->ijkl

サイズ：

D = 4, 8, 16, 32
X = 4, 8, 16, 32, 64, 128

⸻

20. READMEで掲げる特徴

- Cross-language tensor contraction benchmark suite
- C++, Python, Julia, and experimental Rust support
- CPU and GPU backend support
- TBLIS, BLAS, NumPy, PyTorch, JAX, CuPy, TensorOperations.jl support
- Systematic benchmark of tensor index layout
- Row-major, column-major, transposed, permuted, and strided views
- Matrix multiplication and rank-3/rank-4 tensor contractions
- JSONL result format for reproducible analysis

⸻

21. ライセンス

MIT License

⸻

22. リポジトリの最初の成果物

初期コミットでは以下を入れる。

README.md
docs/spec.md
schemas/case.schema.json
schemas/result.schema.json
cases/matmul/square.yaml
cases/rank3/one_bond.yaml
cases/rank4/two_bond.yaml
cpp/include/tcb/tensor_view.hpp
cpp/include/tcb/backend.hpp
cpp/src/runner.cpp
cpp/backends/reference/
cpp/backends/tblis/
tools/generate_cases.py
tools/compare_results.py

⸻

23. 最小実装の方針

最初は、C++ + TBLIS + naive reference に集中する

最小到達点：

tcb-run \
  --case cases/rank4/two_bond.yaml \
  --backend cpp:tblis \
  --repeat 5 \
  --validate always \
  --output result.jsonl
