#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "sampling_gpu.h"

 
__global__ void gather_points_kernel_fast(int b,int n,int m,const float * __restrict__ inp,const int * __restrict__ idx,float * __restrict__ out){
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
      int a=idx[i*m+j];
      out[(i*m+j)*3+0]=inp[(i*n+a)*3+0];
      out[(i*m+j)*3+1]=inp[(i*n+a)*3+1];
      out[(i*m+j)*3+2]=inp[(i*n+a)*3+2];
    }
  }
}

void gather_points_kernel_launcher_fast(int b, int n, int m, 
    const float *inp, const int *idx, float *out, cudaStream_t stream) {
    // points: (B, C, N)
    // idx: (B, npoints)
    // output:
    //      out: (B, C, npoints)

    cudaError_t err;
    // dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    // dim3 threads(THREADS_PER_BLOCK);

    gather_points_kernel_fast<<<dim3(2,8,1),512, 0, stream>>>(b, n, m, inp, idx, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void gather_points_grad_kernel_fast(int b,int n,int m,const float * __restrict__ out_g,const int * __restrict__ idx,float * __restrict__ inp_g){
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
      int a=idx[i*m+j];
      atomicAdd(&inp_g[(i*n+a)*3+0],out_g[(i*m+j)*3+0]);
      atomicAdd(&inp_g[(i*n+a)*3+1],out_g[(i*m+j)*3+1]);
      atomicAdd(&inp_g[(i*n+a)*3+2],out_g[(i*m+j)*3+2]);
    }
  }
}

void gather_points_grad_kernel_launcher_fast(int b, int n, int m, 
    const float *out_g, const int *idx, float *inp_g, cudaStream_t stream) {
    // grad_out: (B, C, npoints)
    // idx: (B, npoints)
    // output:
    //      grad_points: (B, C, N)

    cudaError_t err;
    // dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    // dim3 threads(THREADS_PER_BLOCK);

    gather_points_grad_kernel_fast<<<dim3(2,8,1),512, 0, stream>>>(b,n,m,out_g,idx,inp_g);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i, int idx1, int idx2){
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(int b, int n, int m, 
    const float *__restrict__ dataset, float *__restrict__ temp, int *__restrict__ idxs) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    if (m <= 0) return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int batch_index = blockIdx.x;
    dataset += batch_index * n * 3;
    temp += batch_index * n;
    idxs += batch_index * m;

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0)
    idxs[0] = old;

    __syncthreads();
    for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    float x1 = dataset[old * 3 + 0];
    float y1 = dataset[old * 3 + 1];
    float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
        float x2, y2, z2;
        x2 = dataset[k * 3 + 0];
        y2 = dataset[k * 3 + 1];
        z2 = dataset[k * 3 + 2];
        // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
        // if (mag <= 1e-3)
        // continue;

        float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
        float d2 = min(d, temp[k]);
        temp[k] = d2;
        besti = d2 > best ? k : besti;
        best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 1024) {
        if (tid < 512) {
            __update(dists, dists_i, tid, tid + 512);
        }
        __syncthreads();
    }

    if (block_size >= 512) {
        if (tid < 256) {
            __update(dists, dists_i, tid, tid + 256);
        }
        __syncthreads();
    }
    if (block_size >= 256) {
        if (tid < 128) {
            __update(dists, dists_i, tid, tid + 128);
        }
        __syncthreads();
    }
    if (block_size >= 128) {
        if (tid < 64) {
            __update(dists, dists_i, tid, tid + 64);
        }
        __syncthreads();
    }
    if (block_size >= 64) {
        if (tid < 32) {
            __update(dists, dists_i, tid, tid + 32);
        }
        __syncthreads();
    }
    if (block_size >= 32) {
        if (tid < 16) {
            __update(dists, dists_i, tid, tid + 16);
        }
        __syncthreads();
    }
    if (block_size >= 16) {
        if (tid < 8) {
            __update(dists, dists_i, tid, tid + 8);
        }
        __syncthreads();
    }
    if (block_size >= 8) {
        if (tid < 4) {
            __update(dists, dists_i, tid, tid + 4);
        }
        __syncthreads();
    }
    if (block_size >= 4) {
        if (tid < 2) {
            __update(dists, dists_i, tid, tid + 2);
        }
        __syncthreads();
    }
    if (block_size >= 2) {
        if (tid < 1) {
            __update(dists, dists_i, tid, tid + 1);
        }
        __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0)
        idxs[j] = old;
    }
}

void furthest_point_sampling_kernel_launcher(int b, int n, int m, 
    const float *dataset, float *temp, int *idxs, cudaStream_t stream) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);

    switch (n_threads) {
        case 1024:
        furthest_point_sampling_kernel<1024><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 512:
        furthest_point_sampling_kernel<512><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 256:
        furthest_point_sampling_kernel<256><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 128:
        furthest_point_sampling_kernel<128><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 64:
        furthest_point_sampling_kernel<64><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 32:
        furthest_point_sampling_kernel<32><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 16:
        furthest_point_sampling_kernel<16><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 8:
        furthest_point_sampling_kernel<8><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 4:
        furthest_point_sampling_kernel<4><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 2:
        furthest_point_sampling_kernel<2><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 1:
        furthest_point_sampling_kernel<1><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        default:
        furthest_point_sampling_kernel<512><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
