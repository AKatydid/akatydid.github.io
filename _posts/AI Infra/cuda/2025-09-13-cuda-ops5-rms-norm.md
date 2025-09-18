---
layout: post
title: CUDA-Operators-5-RMSNorm
date: 2025-09-13 08:49 +0000
categories: [CUDA]
tags: [CUDA]
pin: false
math: true
mermaid: true
---

## 1.RMSNorm

RMSNorm 是一种归一化操作，使用 均方根（Root Mean Square, RMS）实现归一化，计算公式为：$RMSNorm(x)=\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2} +\epsilon }$，代码表示如下。

```python
x = x / torch.sqrt(torch.mean(x ** 2, dim = 1, keepdim=True) + 1e-5)
```

### 1.1.Implement

RMSNorm 实现分为 2 Pass，Pass 1 计算均方根，Pass 2 对每个元素进行 Normalization。均方根的计算涉及归约的过程，采用 Block 内归，具体代码如下所示。

```c
#define WARP_SIZE 32
#define FLOAT4(ptr) (reinterpret_cast<float*>(&(ptr))[0])

template<const int WarpSize>
__device__ __inline__ float warpReduceSum(float val){
#pragma unroll
    for (int mask = WarpSize >> 1; mask >= 1; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}


template<const int BN, const int BK>
__global__ void rmsnorm_f32_kernel(float* x, float* out, const int N, const int K){
  const int WARP_NUM = BK / WARP_SIZE;
  __shared__ float smem[BN][WARP_NUM];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;

  int lane = tx % WARP_SIZE;
  int warp_id = tx / WARP_SIZE;

  float* cur_line_addr = x + (bx * BN + ty) * K;
  float val = cur_line_addr[tx] * cur_line_addr[tx];
  for (int i = tx + BK; i < K; i += BK){
    val += cur_line_addr[i] * cur_line_addr[i];
  }

  val = warpReduceSum<WARP_SIZE>(val);

  if (lane == 0)
    smem[ty][warp_id] = val;
  __syncthreads();

  if (tx == 0){
    float norm = smem[ty][0] / K;
    for (int i = 1; i < WARP_NUM; ++i){
      norm += smem[ty][i] / K;
    }
    smem[ty][0] = norm;
  }
  __syncthreads();

  float norm_val = rsqrtf(smem[ty][0] + 1e-5);
  for (int i = tx; i < K; i += BK){
    out[(bx * BN + ty) * K + i] = cur_line_addr[i] * norm_val;
  }
}
```

上面的实现中，每个 Block 计算 BN 行元素，每行元素由 BK 个线程计算，在 K 较小时性能较高，结果如下图所示。但是当 K 极大时，每个线程需要访问 K / BK 次 global mem，可以通过 Block 算 (BN, BK') 小块，然后 Block 之间归约进行优化。

![Desktop View](/assets/img/blog/CUDA/rmsnorm_res.png){: width="500" height="400" }

## Reference
[1] DefTruth, Many Others. LeetCUDA: A Modern CUDA Learn Notes with PyTorch for Beginners. 2025. https://github.com/xlite-dev/LeetCUDA.git.
