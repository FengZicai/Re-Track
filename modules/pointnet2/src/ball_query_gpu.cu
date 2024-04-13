#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_gpu.h"
#include "cuda_utils.h"

// input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx (b,m,nsample), pts_cnt (b,m)
__global__ void ball_query_kernel_fast(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    int batch_index = blockIdx.x;
    xyz1 += n*3*batch_index;
    xyz2 += m*3*batch_index;
    idx += m*nsample*batch_index;
    pts_cnt += m*batch_index; // counting how many unique points selected in local region

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        int cnt = 0;
        for (int k=0;k<n;++k) {
            if (cnt == nsample)
                break; // only pick the FIRST nsample points in the ball
            float x2=xyz2[j*3+0];
            float y2=xyz2[j*3+1];
            float z2=xyz2[j*3+2];
            float x1=xyz1[k*3+0];
            float y1=xyz1[k*3+1];
            float z1=xyz1[k*3+2];
            float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
            if (d<radius) {
                if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                    for (int l=0;l<nsample;++l)
                        idx[j*nsample+l] = k;
                }
                idx[j*nsample+cnt] = k;
                cnt+=1;
            }
        }
        pts_cnt[j] = cnt;
    }
}


void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample,
    const float *xyz, const float *new_xyz, int *idx, int *pts_cnt, cudaStream_t stream) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)
    //      pts_cnt: (B,M)

    cudaError_t err;

    //dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    //dim3 threads(THREADS_PER_BLOCK);

    ball_query_kernel_fast<<<b, 256, 0, stream>>>(b, n, m, radius, nsample, xyz, new_xyz, idx, pts_cnt);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}