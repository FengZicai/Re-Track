#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <THC/THC.h>

#include "sampling_gpu.h"

extern THCState *state;


int gather_points_wrapper_fast(int b, int n, int m, 
    at::Tensor inp_tensor, at::Tensor idx_tensor, at::Tensor out_tensor){
    const float *inp = inp_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *out = out_tensor.data<float>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    gather_points_kernel_launcher_fast(b, n, m, inp, idx, out, stream);
    return 1;
}


int gather_points_grad_wrapper_fast(int b, int n, int m,
    at::Tensor out_g_tensor, at::Tensor idx_tensor, at::Tensor inp_g_tensor) {

    const float *out_g = out_g_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *inp_g = inp_g_tensor.data<float>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    gather_points_grad_kernel_launcher_fast(b, n, m, out_g, idx, inp_g, stream);
    return 1;
}


int furthest_point_sampling_wrapper(int b, int n, int m, 
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

    const float *points = points_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx, stream);
    return 1;
}
