<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

## Introduction
- LLM的内存占用很大
- LLM推理在内存分配不规则的情况下表现出显着的动态行为（基于最大请求长度的静态预分配导致严重的碎片；vLLM引入PagedAttention解决动态KVCache挑战，但其应用仅限于KVCache，同时隔离激活内存分配）
    - 将张量分为三类：权重、激活和KVCache。
    ![image](https://hackmd.io/_uploads/HJxUYCTl0le.png)
        - （1）请求长度的变化。对于LLaMA-3-8B的prefill阶段，262K输入比2k输入使用了更大比例的激活。由于请求长度在服务期间是不可预测的，现有框架根据最大可能长度来预先分配激活张量，可能导致GPU内存利用率不足
        - （2）推理阶段的变化会导致激活使用的显著波动。262K输入的LLaMA-3-8B从预填充过渡到解码时，激活比例从40%以上下降到10%以下
        - （3）最新的模型架构GQA、MLA、Jamba专注于KVCache压缩，但是其他部分显存占用仍非常多，瓶颈已经从以KVCache为中心转变为包含所有动态张量
    - 现有方法存在局限
        - Chunked prefill由于KVCache读取放大，在长上下文推理中导致严重的低效率。
        - PD分离的方案通常依赖于在多个GPU上引入冗余副本来实现弹性。
    - 本质原因：运行时内存空间和KVCache空间之间的隔离。由于三种张量是隔离的，因此在它们的内存空间之间产生了不可逾越的鸿沟，最终降低了内存利用率和系统性能。
    ![image](https://hackmd.io/_uploads/SJrYq0l0gx.png)
    
eLLM将所有物理内存统一到一个共享池中，该池可以在KVCache和激活内存之间动态分配，从而减少碎片并最大限度地提高利用率。
![image](https://hackmd.io/_uploads/r1thCCxRlx.png)
KVCache和激活可以相互“借用”内存
eLLM引入了两种类型的虚拟节点，将张量对象的虚拟地址空间与底层物理内存块解耦
eLLM提出了一种轻量级调度策略，通过协调弹性内存管理来优化内存分配、任务调度和资源转换

contributions：
- 识别LLM内存瓶颈：KV cache和activation的内存隔离
- 提出弹性张量抽象：将虚拟地址与物理内存解耦并统一内存管理
- 设计弹性内存机制：GPU内存动态膨胀/收缩机制、基于CPU的弹性缓冲区、SLO驱动的调度策略
## Background
### LLM Inference
Time To First Token (TTFT) SLO
Time Per Output Token (TPOT) SLO
### LLM服务系统中的内存管理
#### Memory Efficiency
- 自回归过程的memory-bound属性
- GPU内存无法容纳传入请求时，可能会产生相当大的排队延迟
#### Memory Management Mechanisms
![image](https://hackmd.io/_uploads/SyM-ukbRlx.png)
- 早期LLM服务系统继承了从深度学习（DL）框架中抽象出来的静态张量模型，为张量分配固定大小的物理连续内存块，并采用基于池的内存重用策略。但是，KVCache是动态扩展的，并且具有不可预测的生命周期和序列长度，传统方案会导致严重的碎片化
- PagedAttention引入了专门针对KVCache的虚拟化抽象，通过页表机制进行管理，消除了连续内存分配的要求。但是它根据模型的最大请求长度预先分配激活和参数的GPU内存，剩余部分为KV块
## Motivation
### Dynamic Memory in LLM
![image](https://hackmd.io/_uploads/Byw3ckbCge.png)
- 请求长度的变化
随着模型上下文长度从数千激增到数百万，内存组成发生了根本性的变化，激活内存占用大幅提高，导致KVCache可用空间减少。
- 模型架构创新
架构创新（如GQA、MLA、Jamba）通过高效的KVCache减轻了长上下文内存压力。MPT-30B（8个GPU）、Yi-34B（4个GPU）、Jamba-52B（2个GPU），这些同样都能处理200k上下文
### 内存利用不足
#### Underutilized Activation Space
![image](https://hackmd.io/_uploads/Bkg4aybRxx.png)
- 未充分利用的激活空间
用2K输入长度的LLaMA3-8B最大上下文长度262K评估vLLM。在预填充阶段，只有35%的分配激活内存处于活动状态，剩下65%的空闲。真实数据中，超过90%的请求批次使用的上下文长度不到模型最大上下文长度的30%。
在解码阶段，激活内存利用率进一步下降到仅1%，因为每个解码步骤只处理少量token。随着CoT的普及，解码周期还变得越来越长。
#### Overloaded KVCache Space
上下文窗口的扩展和前缀缓存技术的采用，显着增加需要保留的KV条目；但是可用的KVCache空间，却随上下文窗口的扩展而矛盾地减少了
内存利用效率低下已成为现代LLM服务系统的一个重大瓶颈
### Our Solution: Elastic Memory
Memory Ballooning是一种经典的操作系统机制，它放松了内存隔离，允许主机系统和虚拟机之间的动态内存重新分配。它利用操作系统虚拟内存管理系统的页表，通过映射关系传播实现内存重新分配。 --> eLLM根据实际需求为KVCache和激活动态分配内存资源。
## Design
### eLLM Overview
![image](https://hackmd.io/_uploads/ryhsrRb0xl.png)
三个核心组件：
- 虚拟张量抽象
- 弹性内存机制
- 轻量级调度策略
### Virtual Tensor Abstraction
eTensor利用GPU虚拟内存管理（VMM）来弥合现有LLM服务系统中激活和KVCache之间的抽象差距
eTensor可以被视为一个数组指针结构，它引用GPU虚拟地址空间内的连续段
#### KV eTensor and Activation eTensor
鉴于KVCache和激活的不同分配粒度和使用模式，eTensor引入了两种张量类型：KV eTensor和activation eTensor
![image](https://hackmd.io/_uploads/HkWl8CbRee.png)
- KVCache表现出：稳定的大小扩展、不频繁的访问模式和推理期间的持久保留
    - --> 在KV eTensor中，虚拟地址以最大并发为每个请求预先分配，虚拟地址空间段的大小等于模型上下文长度
    - --> 物理块则在实际写入期间再按需分配
- activations表现出：具有较短寿命、较高访问频率
    - 大小不均匀的虚拟地址段，对虚拟地址空间进行频繁和细粒度的管理

两种类型的eTensors都将其虚拟地址段（称为tensor slots）与物理内存块的颗粒度严格对齐，从而在访问效率和fragmentation控制之间实现最佳平衡
#### eTensor Pools and Unified Physical Pool
eLLM不会在eTensor实例生命周期结束时立即从物理资源中取消映射，而是将它们标记为已映射的可用tensor slots，从而实现高效的内存重用。
KV eTensor池对传入请求采用最佳匹配算法。给定一组可用的预映射的tensor slots，算法优先选择满足要求的最小可行的slots，如果不存在这样的slots，则触发按需映射（保留了框架原有的Best-Fit with Coalescing (BFC)策略，分配时采用最佳适应、回收时块合并）
### Elastic Memory Mechanism
两个层次的弹性：
- GPU内部内存膨胀和收缩
- GPU-CPU卸载和获取
#### Memory Inflation and Deflation
Inflation操作通过从activation内存池来动态扩展KVCache容量
- Inflation trigger：在KVCache分配请求时，系统首先验证KV内存池是否包含足够的物理内存块，如果不足，则向激活内存池发出借用请求
- Memory reclamation：activation池通过触发轻量级garbage collection (GC)机制来满足请求，GC阶段识别并取消映射分配给非activation eTensor对象的物理内存块
- Ownership transfer：回收的chunks在逻辑上从activation池迁移到KVCache池
- On-demand remapping：GPU虚拟内存管理器将这些块动态映射到目标KV eTensors的虚拟地址空间

GPU内部的弹性内存机制
![image](https://hackmd.io/_uploads/HyuA0SXAgl.png)
- (a)：最初GPU保留历史KVCache，现有的LLM服务系统由于KVCache空间不足，无法立即处理新到达的请求
- (b)：随着内存膨胀，eLLM可以通过借用空闲激活块来及时处理预填充请求
- (c)：通过内存膨胀，eLLM即使在高内存压力下也能实现更大的批次处理作业（因为解码阶段的memory-bound性质）
- (d)：随着内存收缩，eLLM可以返还借用的内存
#### Memory Offloading and Fetching
GPU-CPU之间的弹性内存管理
长上下文LLM服务中，优化系统响应延迟（TTFT）面临更多挑战
- 更多的内存争用导致严重的请求排队延迟
- 预填充阶段超线性扩展的计算复杂性

因此，eLLM能够在预填充阶段主动卸载KVCache到CPU DRAM，减少了排队延迟，以改善TTFT，还可能聚合更大的解码批量大小，从而提高吞吐量

可行性：
- 不立即需要的、新生成的KVCache可以主动卸载，并且只有在相应请求的解码被安排时才能取回
- Transformers的多层结构支持通过层间并行实现overlapping的计算和通信
- KVCache的迁移只需要$𝑂(𝑁)$的线性通信开销，而预填充阶段的self-attention计算会产生$𝑂(𝑁^2)$复杂性

虽然CPU缓冲区改善了请求排队，但是它引入了prefill-prefer趋势，这会影响TPOT指标 --> 调度策略
### Lightweight Scheduling Strategy
- 基本调度策略采用简单的启发式方法，仅依靠元数据更新和二进制决策来最大限度地提高内存利用率，确保延迟敏感的在线服务符合SLO
- 通过结合简洁的SLO感知缓冲区扩展策略，eLLM有效地平衡了TTFT和TPOT之间的权衡
#### Request Scheduling
为了避免潜在死锁风险，eLLM中，当前迭代所需的所有基本KVCache和激活内存资源必须同时分配
- 预填充阶段：此阶段需要大量激活和KVCache内存资源，将KVCache卸载到CPU缓冲区（deflation）
- 解码阶段：将KVCache取回GPU内存来优化资源利用率（inflation）
#### SLO-aware Buffer Scaling Policy
![image](https://hackmd.io/_uploads/HkwTgv7Cgx.png)
TTFT和TPOT之间的权衡 --> 
- eLLM引入了逻辑缓冲区，这是物理缓冲区的抽象，它在其固定的最大容量内动态调整实际可用大小
- 一种SLO感知的逻辑缓冲区缩放算法，如果检测到TPOT违规，则减小逻辑缓冲区大小以限制预填充请求
    - 如果检测到TTFT违规，则增加逻辑缓冲区大小以优化TTFT
    - 如果TTFT或TPOT在特定调度迭代窗口内超过预定义的SLO阈值五次，则触发违规事件，改变缓冲区调整因子
## Implementation
### Decoding speculative pre-mapping
#### Overlapping (Un)mapping Overheads
branch speculation principles，提前主动启动异步内存分配
通过最小的额外内存映射（<50MB）有效地掩盖了映射开销，实现了解码映射成本的完全重叠
#### Asynchronous unmapping
当触发了取消映射10个slots的事件时，系统不需要立即取消插槽的映射，然后重新分配其关联的物理块
通过利用GPU VMM将单个物理块映射到多个虚拟地址的能力，物理块可以一边分配给一个新的tensor slots，同时异步执行原始slots的取消映射

由于[eLLM: Elastic Memory Management Framework for Efficient LLM Serving](https://arxiv.org/abs/2506.15155)没有开源代码，所以这里只是尝试复现这篇工作的内容，并尽量地解耦合以便拓展到其他框架（暂未实现）

---