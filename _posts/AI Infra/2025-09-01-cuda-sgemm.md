---
layout: post
title: CUDA SGEMM
date: 2025-09-01 07:43 +0000
categories: [CUDA]
tags: [CUDA, basic]
pin: true
math: true
mermaid: true
---

通用矩阵乘（GEMM）计算公式为 $C=\alpha AB + \beta C$，为简便计算，下面的 $\alpha$ 和 $\beta$ 分别设置为 1 和 0。

完整代码请参考个人仓库：[OpenKernel]()

## CPU SGEMM
通过 3 层 for 循环实现 CPU 上的 SGEMM， 计算次数 $m \times n \times k$。
```c++
#define OFFSET(row, col, ld) (((row) * ld) + (col))
void cpuSgemm(
    float* a, float* b, float* c,
    const int M,
    const int N,
    const int K,
    const float alpha = 1.0f,
    const float beta = 0.0f
){
    for (int m = 0; m < M; ++m){
        for (int n = 0; n < N; ++n){
            float col = 0.0f;
            for (int k = 0; k < K; ++k){
                col += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = alpha * col + beta * c[OFFSET(m, n, N)];
        }
    }
}
```

## Naive SGEMM
简单的数据并行，访存不行
```c++
// dim3 block(32, 32);
// dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
__global__ void naiveSgemmkernel(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c,
    const int M, const int N, const int K, const float alpha, const float beta
){
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (m < M && n < N){
        float sum = 0.0f;
        for (int k = 0; k < K; ++k){
            sum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = alpha * sum + beta * c[OFFSET(m, n, N)];
    }
}
```

## SGEMM w/ smem
利用 Shared Mem 对 SGEMM 进行优化

```c++
__global__ void sgemmKernelV1(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c,
    const int M, const int N, const int K, const float alpha, const float beta
){
    const int TILE_SIZE = 16;

    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float sa[TILE_SIZE][TILE_SIZE];
    __shared__ float sb[TILE_SIZE][TILE_SIZE];

    float tileSum = 0.0f;
    int nIter = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int i = 0; i < nIter; ++i){
        sa[threadIdx.x][threadIdx.y] = (m < M && (i * TILE_SIZE + threadIdx.y) < K) ? a[OFFSET(m, i * TILE_SIZE + threadIdx.y, K)] : 0.0f;
        sb[threadIdx.x][threadIdx.y] = (n < N && (i * TILE_SIZE + threadIdx.x) < K) ? b[OFFSET(i * TILE_SIZE + threadIdx.x, n, N)] : 0.0f;
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j){
            tileSum += sa[threadIdx.x][j] * sb[j][threadIdx.y];
        }

        __syncthreads();
    }

    if (m < M && n < N) {
        c[m * N + n] = alpha * tileSum + beta * c[m * N + n];
    }
}
```

每个线程计算 (TM, TN) 个元素
``` c++
// #define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
/*  
const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;
dim3 block = dim3(BN / TN, BM / TM);
dim3 grid = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
*/
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmKernelV2(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c,
    const int M, const int N, const int K, const float alpha, const float beta
){
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int load_a_smem_m = tid / (BK / 4);           // tid / 2
    int load_a_smem_k = (tid % (BK / 4)) * 4;     // (tid % 2 == 0) ? 0 : 4
    int load_b_smem_k = tid >> 5;           // tid / 32
    int load_b_smem_n = (tid & 31) << 2;    // (tid % 32) * 4

    int load_a_gmem_m = blockIdx.y * BM + load_a_smem_m;
    int load_b_gmem_n = blockIdx.x * BN + load_b_smem_n;

    float tileSum[TM][TN] = {0.0};
    __shared__ float sa[BM][BK];
    __shared__ float sb[BK][BN];

    int nIter = (K + BK - 1) / BK;
    for (int bk = 0; bk < nIter; ++bk){
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        FLOAT4(sa[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[OFFSET(load_a_gmem_m, load_a_gmem_k, K)]);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        FLOAT4(sb[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[OFFSET(load_b_gmem_k, load_b_gmem_n, N)]);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k){
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm){
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn){
                    tileSum[tm][tn] += sa[threadIdx.y * TM + tm][k] * sb[k][threadIdx.x * TN + tn];
                }
            }
        }

        __syncthreads();
    }

    // write back to global mem
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = blockIdx.y * BM + threadIdx.y * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = blockIdx.x * BN + threadIdx.x * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(tileSum[i][j]);
        }
    }
}
```

## 解决 Bank Conflict 问题
上节通过利用 Shared Memory 大幅提高了访存效率，进而提高了性能，本节将进一步优化 Shared Memory 的使用。

Shared Memory一共划分为32个Bank，每个Bank的宽度为4 Bytes，如果需要**访问同一个Bank的多个数据，就会发生Bank Conflict**。例如一个Warp的32个线程，如果访问的地址分别为0、4、8、...、124，就不会发生Bank Conflict，只占用Shared Memory一拍的时间；如果访问的地址为0、8、16、...、248，这样一来地址0和地址128对应的数据位于同一Bank、地址4和地址132对应的数据位于同一Bank，以此类推，那么就需要占用Shared Memory两拍的时间才能读出。

``` c++
// #define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
/*  
const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;
dim3 block = dim3(BN / TN, BM / TM);
dim3 grid = dim3((N + BN - 1) / BN, (M + BM - 1) / BM);
*/
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmKernelV3(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c,
    const int M, const int N, const int K, const float alpha, const float beta
){
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BK][BM];
    __shared__ float s_b[BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        s_a[load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2         ]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2         ]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}
```

## Reference
[1] https://zhuanlan.zhihu.com/p/657632577
[2] 
