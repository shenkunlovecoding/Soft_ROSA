# Soft ROSA

Soft ROSA is a pure Python / PyTorch reference implementation of
**differentiable suffix-match attention**.

The goal of this folder is not speed. The goal is to answer a narrower
question first:

1. Can we express the core ROSA longest-suffix dynamic program as a fully
   differentiable operator?
2. Does that operator move toward Hard ROSA as the temperatures grow?
3. Can we run a small end-to-end experiment in plain PyTorch before writing
   Triton or CUDA kernels?

This directory provides a "yes, first reference" implementation for that path.

## Files

* `soft_rosa.py`
  * `soft_rosa_forward()`: differentiable Soft ROSA reference.
  * `hard_rosa_reference()`: simple causal Hard ROSA baseline for comparison.
  * `affine_scan_serial()`: serial reference for the diagonal affine scan.
  * `diagonal_affine_scan()`: CUDA-extension diagonal scan with Triton/CPU fallback.
  * Exact soft selection logic with a lower-intermediate-memory no-aux path.
* `soft_rosa_cuda.py`
  * Lazy-loaded CUDA extension wrapper for the diagonal affine scan autograd path.
  * Uses a packed-diagonal 1D scan implemented with CUB `DeviceScan::InclusiveScanByKey`.
* `soft_rosa_triton.py`
  * Triton forward/backward kernels for the diagonal affine scan.
  * GPU fallback module used when the CUDA extension path is unavailable.
* `demo.py`
  * Runs three small experiments:
    * approximation-to-hard check,
    * gradient sanity check,
    * tiny trainability demo.
* `qkv1bit.py`
  * Unified QKV-1bit entrypoints:
    * `qkv1bit_forward(..., backend=...)`
    * `qkv1bit_rosa(..., backend=...)`
  * Reference Hard ROSA forward for the QKV-1bit setting.
  * Two finite-difference backward paths:
    * `finite_diff_bwd_channelwise()`: flip one channel at a time.
    * `finite_diff_bwd_all_channels()`: flip all channels at once at a fixed time.
* `qkv1bit_triton.py`
  * Triton hard-forward backend using a diagonal associative scan over the
    equality matrix.
  * Triton backward kernels for the specialized all-channels-at-once proxy
    gradient.
* `qkv1bit_cuda.py`
  * Lazy-loaded CUDA forward/backward mirror with the same Python API.
  * Uses a `K <= 64` packed-bit fast path in CUDA, with fallback for larger K.
* `csrc/qkv1bit.cpp`, `csrc/qkv1bit.cu`
  * CUDA extension for the QKV-1bit backend, including the specialized
    backward kernels.
  * The CUDA fast path bit-packs each 1-bit stream history into `uint64_t`,
    uses bitwise compare + `__clzll` in forward, and uses a packed-domain proxy
    backward in CUDA.
* `qkv1bit_demo.py`
  * Proves the channel-independence trick numerically.
  * Checks forward parity for `reference / triton / cuda`.
  * Checks gradient parity against the channelwise finite-difference baseline.
  * Benchmarks `ref-channelwise / ref-all-channels / triton / cuda`.
* `ops.py`
  * `soft_rosa_ops`: operator-style wrapper with `rosa_soft`-like `[B, H, T, D]` interface.
  * `soft_rosa_serial_ops`: force serial diagonal scan.
  * `soft_rosa_parallel_ops`: force the GPU scan path (CUDA extension with Triton fallback).
  * `qkv1bit_rosa_ops`: hard QKV-1bit operator for `[B, T, H*C]`.
* `benchmark_ops.py`
  * Benchmarks serial vs parallel diagonal scan.
  * Benchmarks this project's operator wrappers against `rosa_soft` operators.
  * Benchmarks `qkv1bit_rosa_ops` for `reference / triton / cuda`.
* `MIGRATION.md`
  * Migration notes from `rosa_soft` imports and APIs to this project.

## Core idea

For query `q_i` and key `k_j`, define a soft match score

`mu(i, j) = sigmoid(alpha * sim(q_i, k_j))`

Then define the soft suffix length dynamic program

`ell(i, j) = mu(i, j) * (1 + ell(i - 1, j - 1))`

with zero boundary conditions outside the sequence.

Finally, aggregate values with

`y_i = sum_j softmax_j(gamma * ell(i, j)) * v_{j + 1}`

using only causal `j < i` positions.

This is the direct continuous relaxation of the hard suffix-length DP.

## Current implementation choices

* Similarity:
  * `dot`
  * `cosine`
  * `cosine_margin`

The demo uses `cosine_margin = 2 * cosine - 1` because it makes toy exact-match
experiments sharper: identical one-hot vectors map to `+1`, orthogonal one-hot
vectors map to `-1`.

* Scan:
  * CPU keeps the serial diagonal recurrence as the reference path.
  * CUDA now uses a native CUDA-extension diagonal scan with backward support.
  * The CUDA path packs diagonals into a 1D stream and runs a keyed segmented scan through CUB.
  * Triton remains available as a GPU fallback path for the scan only.

* Complexity:
  * Current reference path is still `O(T^2)` work and `O(T^2)` memory.
  * It is intended as a correctness and training-behavior prototype.

## How to run

From this directory:

```bash
python demo.py
```

The demo prints:

1. How well Soft ROSA matches Hard ROSA as `alpha` and `gamma` increase.
2. Whether gradients propagate through query, key, and value tensors.
3. Whether a tiny learned embedding setup can reduce loss toward Hard ROSA
   targets with annealed temperatures.

For the QKV-1bit specialization:

```bash
python qkv1bit_demo.py --device cuda --repeat 1
```

This second demo prints:

1. a direct channel-independence check,
2. exact forward parity for `reference`, `triton`, and `cuda`,
3. exact gradient equivalence against the channelwise finite-difference baseline,
4. a benchmark showing both the `~C` algorithmic gain and the backend speedup.

The public API for the QKV-1bit operator is:

```python
from soft_rosa import qkv1bit_rosa

y = qkv1bit_rosa(q, k, v, K=6, backend="triton")
```

where `q/k/v` have shape `[B, T, H*C]` and the last dimension is treated as a
flat set of independent 1-bit streams. Float inputs are quantized by sign.
On CUDA, the fast path bit-packs the history when `K <= 64`.
That fast path has three main pieces:

1. forward uses packed history windows and `xor + __clzll` to compute suffix
   match lengths without a byte-wise inner loop;
2. `dv` backward uses an `atomicAdd` scatter from `best_j + 1`, reducing that
   part from a per-target scan to an `O(T)` write pattern;
3. `dq/dk` backward keeps the current proxy-gradient semantics but evaluates
   the flipped candidates in packed form inside CUDA instead of byte-wise
   replays.

For the operator-style wrapper and cross-project benchmark:

```bash
python benchmark_ops.py --device cuda --repeat 1
```

This script prints:

1. serial vs parallel diagonal scan timing,
2. operator benchmark for:
   * `soft_rosa_serial_ops`
   * `soft_rosa_parallel_ops`
   * `soft_rosa_ops`
   * `rosa_soft_ops`
   * `rosa_sufa_ops`
   * `rosa_scan_ops`
3. QKV-1bit operator benchmark for:
   * `qkv1bit_rosa_ops(..., backend="reference")`
   * `qkv1bit_rosa_ops(..., backend="triton")`
   * `qkv1bit_rosa_ops(..., backend="cuda")`

## Current demo result

On the current toy sequence

`[0, 1, 0, 1, 2, 0, 1, 3, 0, 1, 2, 0, 1, 4]`

the reference implementation currently shows:

* Hard-match endpoint accuracy: `1.000` at `alpha=48`, `gamma=48`
* Matched-token accuracy: `1.000`
* Gradient sanity check: non-zero gradients on query, key, and value
* Tiny trainability demo:
  * initial loss: `0.155938`
  * final loss: `0.000000`
  * final matched-token accuracy: `1.000`

This is not a proof, but it is a good first signal that:

* the forward recurrence matches the intended behavior on a repeated-pattern toy
  case,
* the operator remains trainable with plain autograd,
* annealed temperatures can move the soft path toward the hard baseline.

For `qkv1bit_demo.py`, the current CUDA run shows:

* simultaneous-flip channel-slice error: `0.000000`
* forward parity: exact for `reference / triton / cuda`
* max gradient difference to channelwise reference:
  * `dq`: `0.000000`
  * `dk`: `0.000000`
  * `dv`: `0.000000`
* benchmark:
  * `B=1, T=16, N=8, K=6`: Triton `249.14x`, CUDA `495.38x`
  * `B=1, T=12, N=16, K=6`: Triton `235.05x`, CUDA `562.43x`
  * `B=1, T=12, N=32, K=6`: Triton `407.71x`, CUDA `767.32x`

That is exactly the effect we want: when channels are independent, flipping all
channels at a fixed time step produces the same per-channel gradient information
as flipping them one-by-one, but with about `C` times fewer forward calls.

For `benchmark_ops.py`, the current CUDA run currently shows for the exact-soft path:

* serial vs parallel diagonal scan:
  * `B*H=4, T=64`: `223.15x`
  * `B*H=8, T=128`: `212.30x`
  * `B*H=8, T=256`: `2783.78x`
* operator benchmark:
  * `B=1, H=2, T=64, D=8`
    * `soft_rosa_serial_ops`: `197.37 ms`
    * `soft_rosa_parallel_ops`: `1.96 ms`
    * `soft_rosa_ops`: `1.60 ms`
  * `B=1, H=4, T=128, D=8`
    * `soft_rosa_serial_ops`: `774.36 ms`
    * `soft_rosa_parallel_ops`: `1.86 ms`
    * `soft_rosa_ops`: `1.80 ms`
* QKV-1bit operator benchmark:
  * `B=1, T=16, N=8, K=6`
    * `reference`: `708.49 ms`
    * `triton`: `1.19 ms`
    * `cuda`: `0.83 ms`
  * `B=1, T=12, N=16, K=6`
    * `reference`: `285.58 ms`
    * `triton`: `1.23 ms`
    * `cuda`: `0.82 ms`
  * `B=1, T=12, N=32, K=6`
    * `reference`: `329.21 ms`
    * `triton`: `1.39 ms`
    * `cuda`: `0.81 ms`

## Windows JIT Note

If a Windows session hits `LNK1104` or a garbled `cp1` decode error while
building the CUDA extension, the common root cause is a locked `.pyd` from an
older Python process. The manual recovery steps are:

1. Close any Python or Codex process that may still have the extension loaded.
2. Remove the cached extension directories under:
   `C:\Users\Administrator\AppData\Local\torch_extensions\torch_extensions\Cache\py310_cu128`
3. Re-run the benchmark or demo in a fresh terminal.
4. If needed, enter the newest cache directory and run `ninja -v` to see the
   raw linker error directly.

The same workflow applies to both the exact-soft scan extension and the
`qkv1bit` extension, since both are loaded through PyTorch's JIT extension
mechanism on Windows.

## What "correct" means in this folder

For this first prototype, "correct" means:

* the recurrence matches the intended soft DP definition,
* autograd flows cleanly,
* on a repeated toy sequence, higher temperatures move the soft operator toward
  the hard causal suffix-match baseline,
* a small optimization loop can reduce the gap further.

This folder does **not** yet claim:

* a production-quality parallel scan,
* exact convergence guarantees in all unmatched edge cases,
* a final training recipe for large-scale LLM runs.

## Next steps

* replace the serial scan with a real parallel scan kernel,
* add an optional null / no-match branch,
* add a 1-bit per-channel QKV mode directly matching the RWKV-8 setting,
* compare the training behavior against `rosa_soft` STE and `wind_rosa`.
