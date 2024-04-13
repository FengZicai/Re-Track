#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "selection_sort_gpu.h"
#include "cuda_utils.h"


// input: k (1), distance matrix dist (b,m,n)
// output: idx (b,m,n), dist_out (b,m,n)
// only the top k results within n are useful
__global__ void selection_sort_kernel_fast(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    int batch_index = blockIdx.x;
    dist+=m*n*batch_index;
    outi+=m*n*batch_index;
    out+=m*n*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    // copy from dist to dist_out
    for (int j=index;j<m;j+=stride) {
        for (int s=0;s<n;++s) {
            out[j*n+s] = dist[j*n+s];
            outi[j*n+s] = s;
        }
    }

    float *p_dist;
    for (int j=index;j<m;j+=stride) {
        p_dist = out+j*n;
        // selection sort for the first k elements
        for (int s=0;s<k;++s) {
            int min=s; 
            // find the min
            for (int t=s+1;t<n;++t) {
                if (p_dist[t]<p_dist[min]) {
                    min = t;
                }
            }
            // swap min-th and i-th element
            if (min!=s) {
                float tmp = p_dist[min];
                p_dist[min] = p_dist[s];
                p_dist[s] = tmp;
                int tmpi = outi[j*n+min];
                outi[j*n+min] = outi[j*n+s];
                outi[j*n+s] = tmpi;
            }
        }
    }
}
    
void selection_sort_kernel_launcher_fast(int b, int n, int m, int k, \
    const float *dist, int *outi, float *out, cudaStream_t stream) {

    //  Input:
    //  k: int32, number of k SMALLEST elements selected
    //  dist: (b,m,n) float32 array, distance matrix, m query points, n dataset points
    //  Output:
    //  outi: (b,m,n) int32 array, first k in n are indices to the top k
    //  out: (b,m,n) float32 array, first k in n are the top k


    cudaError_t err;

    //dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    //dim3 threads(THREADS_PER_BLOCK);

    selection_sort_kernel_fast<<<b, 256, 0, stream>>>(b, n, m, k, dist, outi, out);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
