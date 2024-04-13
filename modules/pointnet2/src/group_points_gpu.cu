#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "group_points_gpu.h"

// input: grad_out (b,m,nsample,c), idx (b,m,nsample), 
// output: grad_points (b,n,c)
__global__ void group_points_grad_kernel_fast(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points) {
    int batch_index = blockIdx.x;
    idx += m*nsample*batch_index;
    grad_out += m*nsample*c*batch_index;
    grad_points += n*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                 atomicAdd(&grad_points[ii*c+l], grad_out[j*nsample*c+k*c+l]);
            }
        }
    }
}

void group_points_grad_kernel_launcher_fast(int b, int n, int c, int m, int nsample, 
    const float *grad_out, const int *idx, float *grad_points, cudaStream_t stream) {
    // grad_out: (B, C, npoints, nsample)
    // idx: (B, npoints, nsample)
    // output:
    //      grad_points: (B, C, N)
    cudaError_t err;
    // dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    // dim3 threads(THREADS_PER_BLOCK);

    group_points_grad_kernel_fast<<<b,256, 0, stream>>>(b,n,c,m,nsample,grad_out,idx,grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


// input: points (b,n,c), idx (b,m,nsample)
// output: out (b,m,nsample,c)
__global__ void group_points_kernel_fast(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    int batch_index = blockIdx.x;
    points += n*c*batch_index;
    idx += m*nsample*batch_index;
    out += m*nsample*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                out[j*nsample*c+k*c+l] = points[ii*c+l];
            }
        }
    }
}


void group_points_kernel_launcher_fast(int b, int n, int c, int m, int nsample, 
    const float *points, const int *idx, float *out, cudaStream_t stream) {
    // points: (B, C, N)
    // idx: (B, npoints, nsample)
    // output:
    //      out: (B, C, npoints, nsample)
    cudaError_t err;
    // dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    // dim3 threads(THREADS_PER_BLOCK);

    group_points_kernel_fast<<<b,256, 0, stream>>>(b,n,c,m,nsample,points,idx,out);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
