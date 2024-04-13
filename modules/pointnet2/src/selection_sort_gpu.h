#ifndef _SELECTION_SORT_GPU_H
#define _SELECTION_SORT_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int selection_sort_wrapper_fast(int b, int n, int m, int k, 
    at::Tensor dist_tensor, at::Tensor outi_tensor, at::Tensor out_tensor);

void selection_sort_kernel_launcher_fast(int b, int n, int m, int k,
    const float *dist, int *outi, float *out, cudaStream_t stream);

#endif
