#include <ATen/ATen.h>

#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

using at::Tensor;

namespace {

constexpr int kBlockSize = 256;

struct AffineState {
    float g;
    float a;
};

struct AffineCombine {
    __device__ __forceinline__ AffineState operator()(const AffineState& lhs, const AffineState& rhs) const {
        return {rhs.g * lhs.g, rhs.g * lhs.a + rhs.a};
    }
};

template <typename T>
__device__ __forceinline__ T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

__global__ void diagonal_scan_forward_kernel(
    const float* x,
    float* y,
    int batch,
    int seq_len
) {
    int b = blockIdx.x;
    int d = blockIdx.y;
    if (b >= batch || d >= seq_len) {
        return;
    }

    using BlockScan = cub::BlockScan<AffineState, kBlockSize>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ AffineState carry_state;

    const float* x_b = x + static_cast<size_t>(b) * seq_len * seq_len;
    float* y_b = y + static_cast<size_t>(b) * seq_len * seq_len;
    int diag_len = seq_len - d;
    int num_tiles = ceil_div(diag_len, kBlockSize);

    if (threadIdx.x == 0) {
        carry_state = {1.0f, 0.0f};
    }
    __syncthreads();

    for (int tile = 0; tile < num_tiles; ++tile) {
        int idx = tile * kBlockSize + threadIdx.x;
        bool valid = idx < diag_len;

        int r = idx + d;
        int c = idx;
        float g = valid ? x_b[static_cast<size_t>(r) * seq_len + c] : 1.0f;
        AffineState local = {g, valid ? g : 0.0f};

        AffineState prefix;
        BlockScan(temp_storage).InclusiveScan(local, prefix, AffineCombine{});
        __syncthreads();

        AffineState carry = carry_state;
        AffineState combined = {prefix.g * carry.g, prefix.g * carry.a + prefix.a};
        if (valid) {
            y_b[static_cast<size_t>(r) * seq_len + c] = combined.a;
        }
        __syncthreads();

        if (threadIdx.x == kBlockSize - 1 || idx == diag_len - 1) {
            carry_state = combined;
        }
        __syncthreads();
    }
}

__global__ void diagonal_scan_backward_kernel(
    const float* x,
    const float* y,
    const float* grad_output,
    float* grad_input,
    int batch,
    int seq_len
) {
    int b = blockIdx.x;
    int d = blockIdx.y;
    if (b >= batch || d >= seq_len) {
        return;
    }

    using BlockScan = cub::BlockScan<AffineState, kBlockSize>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ AffineState carry_state;

    const float* x_b = x + static_cast<size_t>(b) * seq_len * seq_len;
    const float* y_b = y + static_cast<size_t>(b) * seq_len * seq_len;
    const float* g_b = grad_output + static_cast<size_t>(b) * seq_len * seq_len;
    float* o_b = grad_input + static_cast<size_t>(b) * seq_len * seq_len;
    int diag_len = seq_len - d;
    int num_tiles = ceil_div(diag_len, kBlockSize);

    if (threadIdx.x == 0) {
        carry_state = {1.0f, 0.0f};
    }
    __syncthreads();

    for (int tile = num_tiles - 1; tile >= 0; --tile) {
        int rev_idx = tile * kBlockSize + threadIdx.x;
        bool valid = rev_idx < diag_len;
        int idx = diag_len - 1 - rev_idx;

        int r = idx + d;
        int c = idx;
        float a = (valid && r < seq_len - 1) ? x_b[static_cast<size_t>(r + 1) * seq_len + (c + 1)] : 1.0f;
        float b_term = valid ? g_b[static_cast<size_t>(r) * seq_len + c] : 0.0f;
        AffineState local = {a, b_term};

        AffineState prefix;
        BlockScan(temp_storage).InclusiveScan(local, prefix, AffineCombine{});
        __syncthreads();

        AffineState carry = carry_state;
        AffineState combined = {prefix.g * carry.g, prefix.g * carry.a + prefix.a};
        if (valid) {
            float y_prev = idx > 0 ? y_b[static_cast<size_t>(r - 1) * seq_len + (c - 1)] : 0.0f;
            o_b[static_cast<size_t>(r) * seq_len + c] = (y_prev + 1.0f) * combined.a;
        }
        __syncthreads();

        if (threadIdx.x == kBlockSize - 1 || rev_idx == diag_len - 1) {
            carry_state = combined;
        }
        __syncthreads();
    }
}

}  // namespace

Tensor diagonal_scan_forward_cuda(const Tensor& x) {
    auto y = at::zeros_like(x);
    int batch = static_cast<int>(x.size(0));
    int seq_len = static_cast<int>(x.size(1));
    dim3 grid(batch, seq_len);
    diagonal_scan_forward_kernel<<<grid, kBlockSize>>>(x.data_ptr<float>(), y.data_ptr<float>(), batch, seq_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

Tensor diagonal_scan_backward_cuda(const Tensor& x, const Tensor& y, const Tensor& grad_output) {
    auto grad_input = at::zeros_like(x);
    int batch = static_cast<int>(x.size(0));
    int seq_len = static_cast<int>(x.size(1));
    dim3 grid(batch, seq_len);
    diagonal_scan_backward_kernel<<<grid, kBlockSize>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        batch,
        seq_len
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_input;
}
