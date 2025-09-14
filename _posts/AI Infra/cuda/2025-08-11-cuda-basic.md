---
layout: post
title: CUDA-Basic
date: 2025-08-11 11:38 +0000
categories: [CUDA]
tags: [CUDA, basic]
pin: false
math: true
mermaid: true
---

## 1. CUDA 编程结构
GPU 编程通常是异构环境（多个CPU，GPU），CPU 和 GPU 通过 PCIe 总线相互通信，也是通过 PCIe 总线分隔开的。所以，我们需要区分**CPU及其内存** 和 **GPU及其内存**。

​注意，目前不考虑统一寻址，调试程序在编写时，内存调度采用在 CPU（Host）和 GPU（Device）来回 copy 的方法。

​一种完整的 CUDA 应用可能的执行顺序如 <i>Fig 1</i> 所示，<b>核函数被调用后，控制权马上归还主机线程</b>。
![Desktop View](/assets/img/blog/CUDA/host-device.png){: width="500" height="350" }
<center><i>Fig 1.</i> CUDA应用可能的执顺序</center>

CUDA 编程结构主要涉及四个方面：内存、线程、核函数、错误处理。下面将分别介绍。

### 1.1 内存
CUDA提供的API可以分配管理**设备(Device)**上的内存，API 如下所示。

| 标准C函数 | CUDA C 函数 |   说明   |
| :-------: | :---------: | :------: |
|  malloc   | cudaMalloc  | 内存分配 |
|  memcpy   | cudaMemcpy  | 内存复制 |
|  memset   | cudaMemset  | 内存设置 |
|   free    |  cudaFree   | 释放内存 |

`cudaMemcpy` 最关键，用于拷贝 Host 内存到 Device 上，能完成下述几种拷贝类型（cudaMemcpyKind）。拷贝方向如参数字面所示，如果函数执行成功，则会返回 `cudaSuccess`，否则返回 `cudaErrorMemoryAllocation`。

```c++
/** cudaMemcpyKind 参数：
* cudaMemcpyHostToHost
* cudaMemcpyHostToDevice
* cudaMemcpyDeviceToHost
* cudaMemcpyDeviceToDevice 
*/
cudaError_t cudaMemcpy(void * dst,const void * src,size_t count, cudaMemcpyKind kind);
```

GPU 内存层次较为复杂，内存层次大致分为：全局内存、共享内存、纹理内存、常量内存、本地内存和寄存器，如 *Fig 2* 所示，后续将进行具体介绍。

![Desktop View](/assets/img/blog/CUDA/mem.png){: width="780" height="450" }
<center><i>Fig 2.</i> CUDA 内存层次 overview</center>

### 1.2 线程
CUDA 是 SIMT（Single Instruction, Multi Threads）架构，**多个线程执行同一份代码**，每个线程拥有独立的寄存器、程序计数器和状态，可根据自身数据条件（如分支判断）选择执行或跳过指令。

核函数则是多个线程执行的“同一份代码”。从线程角度而言，一个核函数只能有 1 个 grid，1 个 grid 可以有多个 block，1 个 block 可以有很多线程。即 grid 在编程中的表达形式：`<block_num_dim3, thread_num_dim3>`，具体层次结构如 *Fig 3* 所示。

![Desktop View](/assets/img/blog/CUDA/thread.png){: width="400" height="450" }
<center><i>Fig 3.</i> CUDA 线程层次结构 overview</center>

<b>每个线程都执行同一份的串行代码</b>，怎么让这段相同的代码对应不同的数据呢？使用标记，让块内线程彼此区分开。CUDA 中依靠 `blockIdx` (线程块 block 在线程网格 grid 内的位置索引)和 `threadIdx` (线程 thread 在线程块 block 内的位置索引)。两个结构体基于 uint3 的定义，包含三个无符号整数结构。

``` c++
struct __device_builtin__ dim3
{
    unsigned int x, y, z;
}
```

需要注意，**不同 block 内的线程是物理隔离的，不能相互影响。** 相同 block 内的线程可以完成：（1）同步；（2）共享一块内存。

> **SIMD vs SIMT**
> 
> SIMD 单指令多数据，属于**向量机制**。例如，2 个 float 数组相加，如果 SIMD 指令长度为 16B，那么可以将 4 个 float 填充到向量寄存器，通过一条向量加法指令即可做 4 次 add 指令的工作。但是，向量指令不允许每个分支有不同的操作，每次必须填充满（例如，剩余 3 个数，但向量指令长度需要 4 个才能填满，则需要填充 1 个数）。
> 
> 相比之下，SIMT 中，线程具有单独的资源（如：PC，Reg File），某些线程可以选择不执行。也就是说，同一时刻所有线程被分配给相同的指令，<b>SIMD 必须执行，而 SIMT 可以根据需要不执行</b>。这样 SIMT 保证线程级并行，而 SIMD 则是指令级并行。
> 
> 总结如下，SIMT 包括以下 SIMD 不具有的关键特性，保证了各线程之间的独立（线程 ID 可以通过 `blockIdx` 和 `threadIdx` 获取）。
> 
> 1. 每个 Thread 有单独的指令地址计数器
> 2. 每个 Thread 有单独的寄存器状态
> 3. 每个 Thread 可以有一个独立的执行路径

### 1.3 核函数

#### 1.3.1 核函数调用
核函数的调用通过以下 ANSI C 扩展出的 CUDA C 指令。
``` c
kernel_name<<<grid,block>>>(argument list);
```

下面是一个示例，当配置为 <<<4, 8>>>，即 block = 4， thread = 8 时，核函数的线程分配如图 *Fig 4* 所示。
![Desktop View](/assets/img/blog/CUDA/thread-2.png){: width="800" height="450" }
<center><i>Fig 4.</i> CUDA 线程布局示例</center>

<b>核函数是同时复制到多个线程执行。</b>为了让多个线程对应到不同的数据，需要给线程一个唯一的标识，由于设备内存是线性的，我们可以根据 `threadIdx` 和 `blockIdx` 来组合获得对应的线程的唯一标识，如 *Fig 5* 所示。

![Desktop View](/assets/img/blog/CUDA/thread-3.png){: width="550" height="450" }
<center><i>Fig 5.</i> CUDA 线程布局示例2</center>

当主机 (Host) 启动核函数之后，控制权马上回到主机，而不是主机等待设备完成核函数的运行。如果需要主机 (Host) 等待设备端执行，可以使用下面的同步指令。
``` c
cudaError_t cudaDeviceSynchronize(void);
```

#### 1.3.2 核函数编写
核函数声明模板：`__global__ void kernel_name(argument list);`。CUDA C 中还有一些其他的限定符，如下表所示。

|    限定符    |     执行      |                                               调用                                                | 备注                      |
| :----------: | :-----------: | :-----------------------------------------------------------------------------------------------: | ------------------------- |
| \_\_global__ | Device 端执行 | 可以从 Host 调用<br />也可以从 [计算能力](https://developer.nvidia.cn/cuda-gpus) ≥3的 Device 调用 | 必须有一个void的返回类型  |
| _\_device__  | Device 端执行 |                                           Device 端调用                                           | -                         |
|  _\_host__   |  Host 端执行  |                                            Host 端调用                                            | 可以省略 \_\_host\_\_标识 |

kernel 函数编写有以下限制：**（1）只能访问 Device 端内存；（2）必须有void返回类型；（3）不支持可变数量的参数；（4）不支持静态变量；（5）显示异步行为。**

核函数目的是加速计算，在保证运算正确性前提下，通过并行大量计算核心实现加速计算的目标。而 Kernel 函数正确性验证，通常使用 CPU 代码比较计算精度。

#### 1.3.3 核函数计时
通常的 C 语言程序计时如下，一般调用系统 lib 中的cpu计时器函数，例如 Linux 中的 `gettimeofday(&tp,NULL)`。

``` c
#include <time.h>

clock_t s, e;
s = clock();
// ... 运行程序
e = clock();
double duration = (double)(finish - start) / CLOCKS_PER_SEC;
```

但是，由于核函数与主机程序是异步的，且具有启动开销，使用 `clock()` 统计的时间一般**偏大**，具体如下图 *Fig 6* 所示。

![Desktop View](/assets/img/blog/CUDA/time.png){: width="550" height="650" }
<center><i>Fig 6.</i> Kernel 运行时间</center>

一般而言，使用 CUDA C 内置的 API 为核函数计时，一般在计时前会 warm up 运行几次，然后多次运行求平均时间。
``` c++
  /* warm up */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  /* kernel func */
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
```

### 1.4 错误处理
错误处理目的是帮助定位 bug 位置。CUDA 常用错误处理函数如下所示。

``` c++
#define CHECK_CUDA(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}
```

## 2. CUDA 执行模型
### 2.1 流式处理器（SM）
GPU 架构围绕流式多处理器（SM）扩展阵列搭建，通过复制多个 SM 来实现硬件并行，每个 SM 支持数数百线程并发执行。

如 *Fig 6* 所示，当一个 kernel function 被启动时，<b>多个 Block 会被同时分配到可用的 SM 上执行</b>。Block 被分配到某一个 SM 上以后，将分为多个线程束，每个线程束一般是 32 个线程（目前的 GPU 都是32个线程，但不保证未来还是 32 个）。当一个 Blcok 被分配给一个 SM 后，它就只能在这个 SM 上执行了，不会重新分配到其他 SM 上了，多个线程块可以被分配到同一个 SM 上。

![Desktop View](/assets/img/blog/CUDA/sm.png){: width="675" height="450" }
<center><i>Fig 6.</i> 线程映射关系</center>

Block 是逻辑产物。在计算机里，内存总是一维线性存在的，所以执行起来也是一维的访问线程块中的线程。编程模式中，二维三维的线程块，只是方便写程序。
图 *Fig 7* 从逻辑和硬件角度描述了 CUDA 编程模型对应的硬件。

![Desktop View](/assets/img/blog/CUDA/sm-2.png){: width="500" height="450" }
<center><i>Fig 7.</i> CUDA 编程模型与硬件对应关系</center>

SM 中的**共享内存和寄存器**是关键的资源，一个 Block 中的 Threads 通过共享内存和寄存器相互通信协调。

因为 SM 有限，虽然从 CUDA 编程模型层面看，所有 Thread 都是并行执行的，但是在微观上看，所有 Block 是分批次的在物理层面的机器上执行，Block 里不同的 Thread 可能进度都不一样（因为可能被分到不同的 Warp 上）。但是，<b>同一个 Warp 内的线程拥有相同的进度。</b>

并行会引起竞争，多线程以未定义的顺序访问同一个数据，就导致了不可预测的行为，CUDA 只提供了 Block 内同步机制，Block 之间无法同步。

### 2.2 线程束（Warp）
基本所有设备线程束维持在 32 （切割线程束按照 x 方向切，即一个线程束中的threadIdx.x 连续变化），也就是说，每个 SM 上有多个 Block，一个 Block 有多个线程（可以是几百个，但不会超过某个最大值，最大值一般是 1024）；但是，从机器的角度，<b>在某时刻 T，SM上只执行一个线程束，也就是 32 个线程在同时同步执行，线程束中的每个线程执行同一条指令，包括有分支的部分</b>。

#### 2.2.1 线程束分化
由于 SIMT 的特性，**Warp 分化会产生严重的性能下降。** Warp 分化即同一个 Warp 中的线程，执行不同的指令，例如下面代码示例所示，当一个 Warp 执行时，如果 16个 Thread 执行 if 中的代码，而另外 16 个执行 else 中的代码，则产生 Warp 分化。**条件分支越多，并行性削弱越严重。**

解决方案：从 Warp 角度去解决，根本思路是避免同一个线程束内的线程分化。补充说明下，当一个线程束中所有的线程都执行if或者，都执行else时，不存在性能下降；<b>只有当线程束内有分歧产生分支的时候，性能才会急剧下降。</b>

例如，假设 Block = 64，下面是一个比较低效的分支
``` c++
__global__ void mathKernel1(float *c)
{
	int tid = blockIdx.x* blockDim.x + threadIdx.x;

	float a = 0.0;
	float b = 0.0;
    // 编译器会优化，可以使用 bool ipred = (tid % 2 == 0); 替代条件判断
	if (tid % 2 == 0)
	{
		a = 100.0f;
	}
	else
	{
		b = 200.0f;
	}
	c[tid] = a + b;
}
```
换一种方法，得到相同但是错乱的结果（顺序可以后期调整），那么下面代码就会很高效。
* 第一个线程束内的线程编号 tid 从 0 到 31，tid / warpSize 都等于0，那么就都执行 if 语句；
* 第二个线程束内的线程编号 tid 从 32 到 63，tid / warpSize 都等于1，执行 else；

```c
__global__ void mathKernel2(float *c)
{
	int tid = blockIdx.x* blockDim.x + threadIdx.x;
	float a = 0.0;
	float b = 0.0;
	if ((tid/warpSize) % 2 == 0)
	{
		a = 100.0f;
	}
	else
	{
		b = 200.0f;
	}
	c[tid] = a + b;
}
```

#### 2.2.2 资源分配
如上文所述，每个 SM 上执行的基本单位是 Warp，指令调度器会将同一条指令广播给 Warp 内的全部线程。当出现分支发散时，Warp 会通过线程掩码控制部分线程执行、部分线程闲置（逻辑上不执行当前指令，但 Warp 整体依然在执行），直至分支路径全部完成。从 SM 调度角度看，Warp 可以分为：（1）已激活。已分配到 SM，具备执行所需资源，但可能因等待数据或资源而暂时阻塞；（2）未激活。Block 尚未被分配到 SM，Warp 尚未加载到片上执行。

Warp 一旦被激活来到片上，就不会再离开 SM 直到执行结束。而每个 SM 上有多少个线程束处于激活状态，取决于以下资源：
- 程序计数器
- 寄存器
- 共享内存

换句话说，一个 SM 上能激活多少个 Block 或 Warp（实际运行的是 Warp）取决于 SM 中可用的寄存器和共享内存，以及 Kernel 所需的寄存器和共享内存大小。这是一个 tradeoff，当 kernel 占用的资源较少，那么更多的线程处于活跃状态，相反则线程越少，如图 *Fig 8* 所示。

![Desktop View](/assets/img/blog/CUDA/resource-reg.png){: width="675" height="450" }
![Desktop View](/assets/img/blog/CUDA/resource-shared_mem.png){: width="675" height="450" }
<center><i>Fig 8.</i> 资源分配 Tradeoff 示例 </center>

当寄存器和共享内存分配给了 Block，这个 Block 处于活跃状态，所包含的 Warp 称为活跃线程束。活跃的线程束可分为以下三类。
- 选定的线程束
- 阻塞的线程束
- 符合条件的线程束

​当 SM 执行某个线程束时，执行的这个线程束叫做选定的线程束，准备要执行的叫符合条件的线程束，如果线程束不符合条件还没准备好则为阻塞的线程束。
​满足下面的要求，线程束才算是符合条件的：（1）32 个 CUDA Core 可以用于执行；（2）执行所需资源全部就位。

​由于计算资源是在 Warp 之间分配的，且 Warp 的整个生命周期都在片上，所以 Warp 上下文切换非常快速。因此，可以通过排布流水来实现 访存-计算 延迟隐藏。

由于 Warp 上下文（寄存器状态、程序计数器等）保存在 SM 的片上寄存器文件中，Warp 之间切换开销极小。所以，SM 调度器通过在活跃 Warp 之间轮转计算资源，实现访存与计算的重叠，从而隐藏全局内存访问的延迟，提高硬件利用率。

## 3.CUDA 内存模型

### 3.1 CUDA 内存层次
CUDA的存储体系结构包括全局内存（Global Memory）、共享内存（Shared Memory）、常量内存（Constant Memory）/ 纹理内存（Texture Memory）、本地内存（Local Memory）和寄存器，如 *Fig 9* 所示。

* **全局内存（Global Memory）**
GPU中最大的内存（即 HBM 内存），可以被所有块上的所有线程访问。然而，访问全局内存通常比其他内存类型慢，因此需要进行优化以避免性能下降，常见优化手段：合并内存访问和使用共享内存。

* **共享内存（Shared Memory）**
同一个 Block 内的线程可以通过共享内存共享数据。相比访问全局内存至少**快10倍**。但共享内存的容量有限（A100 192 KB/SM），无法被其他线程块访问。

* **纹理内存和常量内存（Texture and Constant Memory）**
GPU 提供针对特定数据类型优化的特殊内存，如常量内存（只读数据）和纹理内存（二维图像）。**所有线程可访问**这类内存，其访问**速度接近共享内存**。将部分数据放入常量或纹理内存，可优化数据访问，减轻共享内存压力，从而提升整体内存性能。

* **本地内存（Local Memory）**
每个线程都可以使用自己的本地内存，可以在其中存储临时变量。专用于每个单独的线程。

![Desktop View](/assets/img/blog/CUDA/mem2.png){: width="480" height="450" }
<center>Fig 9. CUDA 内存层级</center>

### 3.2 缓存层级
GPU 有如下 4 种缓存，每个SM都有一个一级缓存，所有 SM 公用一个二级缓存。

* L1 Cache
* L2 Cache
* 只读常量缓存：`__ldg(const T* ptr)` 通过只读数据缓存加载全局内存数据，拥有专门的带宽，只读缓存大小一般是几十 KB，**能够避免污染 L1 cache**。\
```c
__global__ void kernel(const float* __restrict__ A, float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // 使用 __ldg 从只读缓存加载
        // 一般用于：
        // （1）常量只读大数组：卷积中的权重、查找表；
        // （2）广播式访问：只读缓存有专门的广播机制，可以让多个线程共享一次加载的结果。
        float val = __ldg(&A[i]);
        B[i] = val * 2.0f;
    }
}
```
* 只读纹理缓存

从 global memory → share memory 过程为：线程从 global memory 经过 L1/L2 cache 加载到寄存器，再显式写入 shared memory。
具体而言，线程发起 load 指令 `smem[i] = gmem[j]` 时，编译器对应生成 `ld.global` + `st.shared` 的 PTX 指令组合，然后，会访问 L1 Cache，没有命中则发往 L2 Cache，L2 也没有命中的话，会去 HBM 上，数据按照 cache line 为单位搬运，**数据放入寄存器后**通过 `st.shared` 写入 shared mem。

但是，在 Ampere 架构后，引入 cp.async 指令，能直接从全局内存加载数据到SM共享内存中（只过 L2 Cache），省略了中间寄存器文件（RF）的过程。

![Desktop View](/assets/img/blog/CV/1756800421772.jpg){: width="600" height="400" }

L1 Cache 是每个 SM 内独有的（和 smem 在物理上是一块存储），L2 Cache 是全局共享的。

### 3.3 内存管理

Host 端内存一般采用分页式虚拟内存管理。应用程序所看到的是连续的虚拟地址空间，而对应的物理页帧可能是非连续分布的。操作系统可以在运行时通过页调度，将物理页帧换出到磁盘或映射到不同的物理内存位置，而应用程序对此感知不到。

Host 和 Device 交互时，GPU 会通过 DMA 直接访问 Host 物理地址，但是，如果 Host 的内存可能会在传输过程中发生变化（OS 换入/换出页，修改页表，逻辑指针能通过逻辑地址+页表查出物理地址，但是，**DMA 是直接操作物理地址的**，如果在传输中发生页偏移，将找不到数据）。

通常解决方案为：
* 锁页（page lock / pinning）：将相关物理页固定，防止被操作系统换出。
* 临时缓冲区复制（staging buffer copy）：驱动将数据先拷贝到一块 固定页内存

![Desktop View](/assets/img/blog/CUDA/mem_trans.png){: width="480" height="450" }
<center>Fig 10. CUDA 内存移动，左图是正常分配过程：锁页-复制到固定内存-复制到设备，右图是固定内存，直接传输到设备上</center>

如 *Fig 10* 所示，CUDA 默认采用淋湿缓冲区复制的方法。同时，CUDA 支持锁页机制，通过使用下面的 API 分配/释放 固定内存，这些内存是页面锁定的，可以直接传输到设备的。
```c
cudaError_t cudaMallocHost(void ** devPtr,size_t count)

cudaError_t cudaFreeHost(void *ptr)
```

固定内存的释放和分配成本比可分页内存要高很多，但是传输速度更快，所以对于大规模数据，固定内存效率更高。数据的传输可以通过 CUDA Stream 掩盖，具体见第 4 章节。

### 3.4 shlf 指令

shuffle instruction **作用在 warp 内**，允许两个线程见**相互访问对方的寄存器**，不通过共享内存或者全局内存，延迟极低，且不消耗内存。
首先提出一个束内线程的概念，英文名 lane，简单的说，就是一个 warp 内的线程 ID，在 【0,31】内，计算方式如下：

```c
unsigned int LaneID = threadIdx.x % 32;
unsigned int warpID = threadIdx.x / 32;
```

例如，在同一个 block 中的 thread_0 和 thread_32 的 lane_id 都为 0。

#### 3.4.1 shlf 指令不同形式
shlf 指令一共有 4 种形式，mask 代表线程掩码，用于指示 warp 内的活跃线程，用 32 bit 表示，例如 `0xffffffff` 则表示全部线程活跃，只有活跃的线程会执行 shlf 指令。mask 能够保证在 warp 内部分线程 inactive 时（例如，分支）， shuffle 行为是安全的。
* `__shfl_sync`：最通用的版本，任意 lane 寻址，从指定的 srcLane 线程取 var 值，即**将 srcLane 的 var 值数据广播到所有 mask 的线程中**，如下图示例所示。
```c
T __shfl_sync(unsigned mask, T var, int srcLane, int width = warpSize);
```

![Desktop View](/assets/img/blog/CUDA/shfl_.png){: width="520" height="500" }

* `__shfl_up_sync`：每个线程取 lane_id - delta 的值（如果越界就返回自己的值）。
```c
T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = warpSize);
```
* `__shfl_down_sync`：每个线程取 lane_id + delta 的值。
```c
T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width = warpSize);
```
`__shfl_up_sync` 和 `__shfl_down_sync` 常用于 Scan 类型算子。从输入角度看，`__shfl_up_sync` 是将 `lane_id → lane_id + delta`，即对应结果角度，当前 lane 线程获取 `lane - delta` 线程的 `val` 寄存器数据。若 `lane - delta < 0`，则**不执行操作**。具体示例，如下图所示。 
![Desktop View](/assets/img/blog/CUDA/shfl_up_sync.png){: width="500" height="500" }
而 `__shfl_down_sync` 从结果看，当前 lane 线程获取 `lane + delta` 线程的 `val` 寄存器数据，具体示例如下图所示。
![Desktop View](/assets/img/blog/CUDA/shfl_down_sync.png){: width="500" height="500" }

* `__shfl_xor_sync`：每个线程取 (lane_id ^ laneMask) 的线程的值，常用于树形归约或交换对称计算
```c
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = warpSize);
```
例如，`__shfl_xor_sync` 指令在于理解 lane_id ^ laneMask 这个过程，代码与图例如下所示。
```c
#param unroll
for (int i = 1; i < warpSize; i *= 2){
  // mask 表示选中的线程，一般为 warp 所有的 32 个线程，即 0xffffffff 或 -1
  val += __shfl_xor_sync(-1, val, i);
}
```

![Desktop View](/assets/img/blog/CUDA/shfl_xor_sync.png){: width="480" height="450" }

## 4.CUDA Stream

上面主要围绕 CUDA Kernel 进行介绍，关注 Device 端的优化，本章节则聚焦于 Host 端怎么优化 CUDA 应用。

### 4.1 CUDA Stream 介绍
CUDA Stream 是用于组织和调度一系列 GPU 操作的逻辑队列。Host 端提交完异步操作（例如，内存拷贝、核函数启动）后，Device 端会有一个默认 Stream 保存提交的操作（不一定立即执行），之后控制权返回到 Host。如果 Host 端继续遇到下一个异步操作，则继续放入默认 Stream 中。

同一个 Stream 的操作是**顺序执行**的，但是，多个 stream 是**并发执行**的，我们可以自定义 CUDA Stream 实现流水线或双缓冲技术。

隐式声明的流（空流）无法管理，必须显示声明（非空流）。基于流的异步内核启动和数据传输支持以下类型的粗粒度并发：
* 重叠 Host 和 Device 计算
* 重叠 Host 计算和 Host-Device 数据传输
* 重叠 Host 设备数据传输和 Device 计算
* 并发多个 Device

之前的内存拷贝 `cudaMemcpy` 是一个同步操作，异步输出传输 API 如下所示。
```c
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);
```
**执行异步数据传输时，Host 端的内存必须是固定的，非分页的！** 可以通过上文介绍的 `cudaError_t cudaMallocHost(void **ptr, size_t size);` 固定内存。

在非空流中执行内核需要在启动核函数的时候加入一个附加的启动配置，下面是相关 API。
```c
// 创建 Stream
cudaStream_t stream;
cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags,int priority);   // 带优先级

// 查询流优先级
cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority);

// 在 Stream 启动 Kernel 需要附带非空流
kernel_name<<<grid, block, sharedMemSize, stream>>>(argument list);

// 查询 Stream 是否完成
cudaError_t status = cudaError_t cudaStreamQuery(cudaStream_t stream);

// 同步 Stream 直至完成
cudaError_t cudaStreamSynchronize(cudaStream_t stream);

// Stream Destroy API，调用销毁时，stream 可能还在执行，当 stream 执行完成后才会进行销毁
cudaError_t cudaStreamDestroy(cudaStream_t stream);
```

下面是一段典型的多流调度 CUDA 操作的示例。
```c
for (int i = 0; i < nStreams; i++) {
    int offset = i * bytesPerStream;
    cudaMemcpyAsync(&d_a[offset], &a[offset], bytePerStream, streams[i]);
    kernel<<grid, block, 0, streams[i]>>(&d_a[offset]);
    cudaMemcpyAsync(&a[offset], &d_a[offset], bytesPerStream, streams[i]);
}
for (int i = 0; i < nStreams; i++) {
    cudaStreamSynchronize(streams[i]);
}
```
假设 nStreams=3，所有传输和核启动都是流水执行的，如图 *Fig 11* 所示。

![Desktop View](/assets/img/blog/CUDA/stream.png){: width="600" height="500" }
<center>Fig 11. CUDA Stream 示例</center>

非空流可以创建为 阻塞流 和 非阻塞流，**空流是阻塞流**，`cudaStreamCreate` 默认创建阻塞流。阻塞流之间会有隐式同步，当一个阻塞流发生阻塞，其他阻塞流必须等待。
```c
// 假设 stream_1 和 stream_2 都是阻塞流
kernel_1<<<1, 1, 0, stream_1>>>();
kernel_2<<<1, 1>>>();
kernel_3<<<1, 1, 0, stream_2>>>();
```

上面代码中，有 3 个流，阻塞执行，具体过程为：kernel_1 被启动，控制权返回 Host，然后 Host 启动 kernel_2，但是 Device 不会马上执行 kernel_2，它会等到 kernel_1 执行完毕再执行 Kernel_2；同理，启动完 kernel_2 后，Host 继续启动 kernel_3，在 Device 上，kernel_3 会等待直到 kernel_2 执行完。但是，从主机的角度，3 个核都是异步的。

如果想实现并行 3 个流，可以使用下面的 API 创建 非阻塞流，如果上面的 stream_1 和 stream_2 是非阻塞流，则 3 个 Kernel 将并行执行。
```c
/**
*  flags 参数选项
*  cudaStreamDefault -> 默认阻塞流
*  cudaStreamNonBlocking -> 非阻塞流
*/
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
```

流同样可以显示同步，下面是同步流的 API 及介绍。

| API                                          | 阻塞情况                               |
| -------------------------------------------- | -------------------------------------- |
| `cudaDeviceSynchronize()`                    | 阻塞主机线程，直到 GPU 上所有任务完成  |
| `cudaStreamSynchronize(cudaStream_t stream)` | 阻塞主机线程，直到指定流的所有任务完成 |
| `cudaMemcpy`（同步版本）                     | 阻塞主机线程，直到数据传输完成         |


### 4.2 重叠内核执行和数据传输

下面以一个向量加法为例，向量加法各个元素之间不存在依赖关系，通过 `cudaMemcpyAsync` 和多流执行不同位置的向量加法内核，实现重叠内核执行和数据传输。具体代码如下所示。

```c
cudaStream_t stream[N_SEGMENT];
for(int i=0;i<N_SEGMENT;i++)
{
    CHECK(cudaStreamCreate(&stream[i]));
}
cudaEvent_t start,stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);
for(int i=0;i<N_SEGMENT;i++)
{
    int ioffset=i*iElem;
    CHECK(cudaMemcpyAsync(&a_d[ioffset],&a_h[ioffset],nByte/N_SEGMENT,cudaMemcpyHostToDevice,stream[i]));
    CHECK(cudaMemcpyAsync(&b_d[ioffset],&b_h[ioffset],nByte/N_SEGMENT,cudaMemcpyHostToDevice,stream[i]));
    sumArraysGPU<<<grid,block,0,stream[i]>>>(&a_d[ioffset],&b_d[ioffset],&res_d[ioffset],iElem);
    CHECK(cudaMemcpyAsync(&res_from_gpu_h[ioffset],&res_d[ioffset],nByte/N_SEGMENT,cudaMemcpyDeviceToHost,stream[i]));
}
//timer
CHECK(cudaEventRecord(stop, 0));
CHECK(cudaEventSynchronize(stop));
```

## Reference
[1] https://face2ai.com/program-blog/
[2] https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec4.pdf
