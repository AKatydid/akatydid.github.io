---
layout: post
title: CUDA-Operators-0-elementwise
date: 2025-09-1 06:52 +0000
categories: [CUDA]
tags: [CUDA]
pin: false
math: true
mermaid: true
---

本系列文章重点阐述了各类算子的逐步优化过程，涵盖 CUDA 常用算子，并对不同算子的性能瓶颈进行分析。各类算子完整代码请参考个人仓库 [OpenKernels](https://github.com/AKatydid/OpenKernels.git)。

Element-wise 算子运算时没有数据之间的依赖关系，且运算比较简单，瓶颈在访存上。优化手段一般为：（1）向量化；（2）数学近似。下面以 ReLU 和 GELU 算子为例，展示逐步优化的过程。

## 1.ReLU
### 1.1 Naive
基础的内核实现如下，每个线程计算 1 个数即可。

```c
// Relu x: N, y: N y=max(0,x)
// grid(N/256), block(K=256)
__global__ void relu_f32_kernel(float *x, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    y[idx] = fmaxf(0.0f, x[idx]);
}
```

### 1.2 向量化优化
使用 FLOAT4 向量化计算，每个线程计算 4 个数。

```c
// Relu x: N, y: N y=max(0,x) Vec4
// grid(N/256/4), block(256/4)
__global__ void relu_f32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = fmaxf(0.0f, reg_x.x);
    reg_y.y = fmaxf(0.0f, reg_x.y);
    reg_y.z = fmaxf(0.0f, reg_x.z);
    reg_y.w = fmaxf(0.0f, reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}
```

## 2.GELU

GELU 的公式为：$GELU(x) = \frac{x}{2}(1+erf(\frac{x}{\sqrt{2}}))$。GELU 可以通过数学近似计算的方法优化，优化后的表达式为： $GELU(x) = 0.5x(1+tanh(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)))$。

注意，当输入数值过大的时候，可能会发生溢出，一般需要对输入数值做裁剪。
```c
#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f

x = fminf(fmaxf(x, MIN_EXP_F32), MAX_EXP_F32);  // clipping
```

### 2.1 Naive 近似计算

```c
#define SQRT_2_PI (M_SQRT2 *M_2_SQRTPI * 0.5f)

__inline__ __device__ float gelu_tanh_approximate(float x){
  return 0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715 * x * x *x)));
}

// block(256)
// grid((N + block.x - 1) / block.x)
__global__ void gelu_f32_kernel(float *x, float *y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float v = fminf(fmaxf(x[idx], MIN_EXP_F32), MAX_EXP_F32);
    y[idx] = gelu_tanh_approximate(v);
  }
}
```

### 2.2 向量化优化

```c
const int BN = 256;
// block(BN / 4)
// grid((N + BN - 1) / BN)
__global__ void gelu_f32x4_kernel(float *x, float *y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_y;

  reg_x.x = fminf(fmaxf(reg_x.x, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.y = fminf(fmaxf(reg_x.y, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.z = fminf(fmaxf(reg_x.z, MIN_EXP_F32), MAX_EXP_F32);
  reg_x.w = fminf(fmaxf(reg_x.w, MIN_EXP_F32), MAX_EXP_F32);

  reg_y.x = gelu_tanh_approximate(reg_x.x);
  reg_y.y = gelu_tanh_approximate(reg_x.y);
  reg_y.z = gelu_tanh_approximate(reg_x.z);
  reg_y.w = gelu_tanh_approximate(reg_x.w);

  if ((idx + 0) < N) {
    FLOAT4(y[idx]) = reg_y;
  }
}
```

## reference
[1] DefTruth, Many Others. LeetCUDA: A Modern CUDA Learn Notes with PyTorch for Beginners. 2025. https://github.com/xlite-dev/LeetCUDA.git.

