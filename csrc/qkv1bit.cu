#include <ATen/ATen.h>

#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

using at::Tensor;
using uchar = unsigned char;

__global__ void qkv1bit_forward_kernel(
    const uchar* q,
    const uchar* k,
    const uchar* v,
    uchar* y,
    int64_t* best_j,
    int64_t* best_len,
    int num_streams,
    int seq_len,
    int K
) {
    int s = blockIdx.x;
    if (s >= num_streams) {
        return;
    }

    const uchar* q_s = q + static_cast<size_t>(s) * seq_len;
    const uchar* k_s = k + static_cast<size_t>(s) * seq_len;
    const uchar* v_s = v + static_cast<size_t>(s) * seq_len;
    uchar* y_s = y + static_cast<size_t>(s) * seq_len;
    int64_t* bj_s = best_j + static_cast<size_t>(s) * seq_len;
    int64_t* bl_s = best_len + static_cast<size_t>(s) * seq_len;

    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        int cur_best_len = 0;
        int cur_best_j = -1;
        int max_check = min(K, i + 1);

        for (int j = 0; j < i; ++j) {
            int max_match = min(max_check, j + 1);
            int len = 0;

            while (len < max_match && q_s[i - len] == k_s[j - len]) {
                ++len;
            }

            if (len > cur_best_len || (len == cur_best_len && len > 0 && j > cur_best_j)) {
                cur_best_len = len;
                cur_best_j = j;
            }
        }

        bj_s[i] = cur_best_j;
        bl_s[i] = cur_best_len;
        y_s[i] = cur_best_j >= 0 ? v_s[cur_best_j + 1] : static_cast<uchar>(0);
    }
}

__device__ __forceinline__ uchar maybe_flip(uchar x, bool should_flip) {
    return should_flip ? static_cast<uchar>(1 - x) : x;
}

__device__ int compute_best_j_with_optional_flip(
    const uchar* q,
    const uchar* k,
    int seq_len,
    int K,
    int flip_t,
    bool flip_query,
    bool flip_key,
    int i
) {
    int cur_best_len = 0;
    int cur_best_j = -1;
    int max_check = min(K, i + 1);

    for (int j = 0; j < i; ++j) {
        int max_match = min(max_check, j + 1);
        int len = 0;

        while (len < max_match) {
            int qi = i - len;
            int kj = j - len;
            uchar qv = maybe_flip(q[qi], flip_query && qi == flip_t);
            uchar kv = maybe_flip(k[kj], flip_key && kj == flip_t);
            if (qv != kv) {
                break;
            }
            ++len;
        }

        if (len > cur_best_len || (len == cur_best_len && len > 0 && j > cur_best_j)) {
            cur_best_len = len;
            cur_best_j = j;
        }
    }

    return cur_best_j;
}

__global__ void qkv1bit_backward_qk_kernel(
    const uchar* q,
    const uchar* k,
    const uchar* v,
    const uchar* base_y,
    const float* grad_output,
    float* grad_x,
    int num_streams,
    int seq_len,
    int K,
    bool flip_query
) {
    int s = blockIdx.x;
    if (s >= num_streams) {
        return;
    }

    const uchar* q_s = q + static_cast<size_t>(s) * seq_len;
    const uchar* k_s = k + static_cast<size_t>(s) * seq_len;
    const uchar* v_s = v + static_cast<size_t>(s) * seq_len;
    const uchar* y_s = base_y + static_cast<size_t>(s) * seq_len;
    const float* go_s = grad_output + static_cast<size_t>(s) * seq_len;
    float* gx_s = grad_x + static_cast<size_t>(s) * seq_len;

    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        float accum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            int off_j = compute_best_j_with_optional_flip(
                q_s,
                k_s,
                seq_len,
                K,
                t,
                flip_query,
                !flip_query,
                i
            );
            uchar off_y = off_j >= 0 ? v_s[off_j + 1] : static_cast<uchar>(0);
            accum += (static_cast<float>(off_y) - static_cast<float>(y_s[i])) * go_s[i];
        }

        uchar x_t = flip_query ? q_s[t] : k_s[t];
        float flip_dir = 1.0f - 2.0f * static_cast<float>(x_t);
        gx_s[t] = accum * flip_dir;
    }
}

__global__ void qkv1bit_backward_v_kernel(
    const uchar* v,
    const int64_t* best_j,
    const float* grad_output,
    float* grad_v,
    int num_streams,
    int seq_len
) {
    int s = blockIdx.x;
    if (s >= num_streams) {
        return;
    }

    const uchar* v_s = v + static_cast<size_t>(s) * seq_len;
    const int64_t* bj_s = best_j + static_cast<size_t>(s) * seq_len;
    const float* go_s = grad_output + static_cast<size_t>(s) * seq_len;
    float* gv_s = grad_v + static_cast<size_t>(s) * seq_len;

    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        float accum = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            if (bj_s[i] >= 0 && bj_s[i] + 1 == t) {
                accum += go_s[i];
            }
        }
        gv_s[t] = accum;
    }
}

std::tuple<Tensor, Tensor, Tensor> qkv1bit_forward_cuda(const Tensor& q, const Tensor& k, const Tensor& v, int64_t K) {
    auto output = at::zeros_like(v);
    auto best_j = at::full({q.size(0), q.size(1)}, -1, q.options().dtype(at::kLong));
    auto best_len = at::zeros({q.size(0), q.size(1)}, q.options().dtype(at::kLong));

    int num_streams = static_cast<int>(q.size(0));
    int seq_len = static_cast<int>(q.size(1));
    int threads = 1;
    while (threads < seq_len && threads < 256) {
        threads <<= 1;
    }

    qkv1bit_forward_kernel<<<num_streams, threads>>>(
        q.data_ptr<uchar>(),
        k.data_ptr<uchar>(),
        v.data_ptr<uchar>(),
        output.data_ptr<uchar>(),
        best_j.data_ptr<int64_t>(),
        best_len.data_ptr<int64_t>(),
        num_streams,
        seq_len,
        static_cast<int>(K)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {output, best_j, best_len};
}

std::tuple<Tensor, Tensor, Tensor> qkv1bit_backward_cuda(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& output,
    const Tensor& best_j,
    const Tensor& grad_output,
    int64_t K
) {
    auto dq = at::zeros_like(grad_output);
    auto dk = at::zeros_like(grad_output);
    auto dv = at::zeros_like(grad_output);

    int num_streams = static_cast<int>(q.size(0));
    int seq_len = static_cast<int>(q.size(1));
    int threads = 1;
    while (threads < seq_len && threads < 256) {
        threads <<= 1;
    }

    qkv1bit_backward_qk_kernel<<<num_streams, threads>>>(
        q.data_ptr<uchar>(),
        k.data_ptr<uchar>(),
        v.data_ptr<uchar>(),
        output.data_ptr<uchar>(),
        grad_output.data_ptr<float>(),
        dq.data_ptr<float>(),
        num_streams,
        seq_len,
        static_cast<int>(K),
        true
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    qkv1bit_backward_qk_kernel<<<num_streams, threads>>>(
        q.data_ptr<uchar>(),
        k.data_ptr<uchar>(),
        v.data_ptr<uchar>(),
        output.data_ptr<uchar>(),
        grad_output.data_ptr<float>(),
        dk.data_ptr<float>(),
        num_streams,
        seq_len,
        static_cast<int>(K),
        false
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    qkv1bit_backward_v_kernel<<<num_streams, threads>>>(
        v.data_ptr<uchar>(),
        best_j.data_ptr<int64_t>(),
        grad_output.data_ptr<float>(),
        dv.data_ptr<float>(),
        num_streams,
        seq_len
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {dq, dk, dv};
}
