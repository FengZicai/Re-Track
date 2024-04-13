#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "selection_sort_gpu.h"

extern THCState *state;

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

int selection_sort_wrapper_fast(int b, int n, int m, int k, 
    at::Tensor dist_tensor, at::Tensor outi_tensor, at::Tensor out_tensor) {
    CHECK_INPUT(dist_tensor);
    const float *dist = dist_tensor.data<float>();
    int *outi = outi_tensor.data<int>();
    float *out = out_tensor.data<float>();
    
    cudaStream_t stream = THCState_getCurrentStream(state);
    selection_sort_kernel_launcher_fast(b, n, m, k, dist, outi, out, stream);
    return 1;
}
