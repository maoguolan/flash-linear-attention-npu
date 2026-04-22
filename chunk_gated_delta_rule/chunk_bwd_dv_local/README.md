# ChunkBwdDvLocal 算子说明
🔗 [查看源码](https://gitcode.com/Wei_NaChuan/flash-linear-attention-npu/tree/main/chunk_gated_delta_rule/chunk_bwd_dv_local)
`ChunkBwdDvLocal` 是 Gated Delta Rule（门控 Delta 规则）线性注意力机制反向传播过程中的自定义算子。该算子根据查询（Q）、键（K）、门控（g）和输出梯度（dO）计算 Value 的本地梯度 $dV_{\text{local}}$。

---

## 1. 算子功能

在分块序列模型反向传播中，计算以下张量的梯度：

- **dV**：Value 的本地梯度

计算公式为：

$$dV_{\text{local}} = \text{mask}\!\left(\exp(g_{\text{col}} - g_{\text{row}})\right) \odot (K Q^T) \cdot dO$$

分解为三个阶段：

| 阶段 | 执行单元 | 计算 | 说明 |
|------|----------|------|------|
| Phase 1 | Cube (AIC) | $W_s = K \times Q^T$ | 矩阵乘，生成 chunk 内 attention score 矩阵 |
| Phase 1.5 | Vector (AIV) | $W_{s\_gated} = \text{mask}(\exp(g)) \odot W_s$ | gating：exp、下三角 mask、逐元素乘 |
| Phase 2 | Cube (AIC) | $dV = W_{s\_gated} \times dO$ | 矩阵乘，生成最终梯度输出 |

---

## 2. 接口定义

### 2.1 ACLNN 接口

每个算子分为两段式调用流程：

1. **获取 workspace 与执行器**  
   调用 `aclnnChunkBwdDvLocalGetWorkspaceSize` 接口，获取算子执行所需的 workspace 大小，并创建执行器（executor）。

2. **执行算子计算**  
   调用 `aclnnChunkBwdDvLocal` 接口，在指定的 workspace 和执行器下完成计算。

对应以下 C++ 接口：
```cpp
// 获取执行所需的 workspace 大小
aclnnStatus aclnnChunkBwdDvLocalGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *dO,
    const aclTensor *g,
    const aclTensor *gGammaOptional,
    const aclTensor *aOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

// 执行算子
aclnnStatus aclnnChunkBwdDvLocal(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

---

## 3. 参数说明

### 3.1 输入参数（Inputs）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|---|---|
| `q` | 输入 | 必选 | Query 输入张量 | 参与反向计算；接口执行前会先转为连续内存 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, K]` | 支持 |
| `k` | 输入 | 必选 | Key 输入张量 | 参与反向计算；接口执行前会先转为连续内存 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, K]` | 支持 |
| `dO` | 输入 | 必选 | 前向输出的梯度张量 | 参与反向计算；接口执行前会先转为连续内存 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, V]` | 支持 |
| `g` | 输入 | 必选 | Gate 输入张量（门控衰减系数） | 参与 gating 计算；接口执行前会先转为连续内存 | `FLOAT16`、`BFLOAT16`、`FLOAT` | `ND` | `[B, H, T]` | 支持 |
| `gGammaOptional` | 输入 | 接口为必传；当前实现要求传空 | 预留门控参数输入 | 当前 tiling 明确要求 `g_gamma == nullptr`，**必须传空指针** | `FLOAT` | `ND` | 未启用 | - |
| `aOptional` | 输入 | 接口为必传；当前实现要求传空 | 预留参数输入 | 当前 tiling 明确要求 `A == nullptr`，**必须传空指针** | `FLOAT16`、`BFLOAT16` | `ND` | 未启用 | - |
| `cuSeqlensOptional` | 输入 | 可选 | 变长序列的累计长度信息 | 变长模式输入，形如 `[0, T1, T1+T2, ...]`，形状为 `[N+1]` | `INT64` | `ND` | 1 维 | - |
| `chunkIndicesOptional` | 输入 | 可选 | 分块索引信息 | 变长模式输入，形状为 `[num_chunks, 2]` | `INT64` | `ND` | 2 维 | - |

### 3.2 属性参数（Attributes）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 使用说明 | 数据类型 | 取值约束 |
|---|---|---|---|---|---|---|
| `scale` | 输入 | 必选 | 缩放系数 | `op_def` 默认值为 `1.0f`；推荐按 `1 / sqrt(K)` 设置 | `double` | 建议按 `1 / sqrt(K)` 设置 |
| `chunkSize` | 输入 | 必选 | 分块大小 | `op_def` 默认值为 `64`；当前 tiling 仅支持 `64` 或 `128` | `int64_t` | 仅支持 `64` / `128` |

### 3.3 输出参数（Outputs）

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|
| `out`（即 `dV`） | 输出 | Value 的本地梯度输出张量 | `FLOAT16`、`BFLOAT16` | `ND` | `[B, H, T, V]` | 支持 |
| `workspaceSize` | 输出 | Device 侧所需 workspace 大小 | `uint64_t` | - | 标量 | - |
| `executor` | 输出 | 算子执行器，封装了计算流程 | `aclOpExecutor*` | - | - | - |

### 3.4 形状与约束

- `q`、`k` 的形状必须为 `[B, H, T, K]`。
- `dO` 的形状必须为 `[B, H, T, V]`。
- `g` 的形状必须为 `[B, H, T]`。
- 当前实现要求 `K = 128`。
- 当前实现要求 `V = 128`。
- `chunkSize` 当前仅支持 `64` 或 `128`。
- 当启用变长模式时，`cuSeqlensOptional` 和 `chunkIndicesOptional` 必须同时提供；当前实现仅支持 `B = 1`。
- 当前实现要求 `gGammaOptional` 和 `aOptional` 传空指针，否则执行失败。

### 3.5 补充说明

- 接口层会先对 `q`、`k`、`dO`、`g` 做连续化处理，因此这些输入可以是非连续 Tensor。
- 若输出张量本身是非连续视图，接口层会在内部通过 `ViewCopy` 将连续计算结果回写到目标输出。

---

## 4. 调用约束与执行语义

### 4.1 执行流程

该算子采用两阶段执行模型：

1. 调用 `aclnnChunkBwdDvLocalGetWorkspaceSize`：
   - 完成参数合法性检查
   - 将输入 Tensor 转换为连续内存（Contiguous）
   - 构建执行器（executor）
   - 返回所需 `workspaceSize`

2. 调用 `aclnnChunkBwdDvLocal`：
   - 在指定 `workspace` 和 `executor` 下执行计算
   - 内部通过 Cube（AIC）+ Vector（AIV）协同完成三阶段计算

---

### 4.2 内存行为

- 输入张量 `q/k/dO/g`：
  - 若为非连续 Tensor，将在内部自动转换为连续布局再参与计算
- 输出张量 `out`：
  - 若为非连续视图，将通过内部 `ViewCopy` 回写结果

---

### 4.3 可选参数约束

- `cuSeqlensOptional` 和 `chunkIndicesOptional`：
  - 同时出现时启用变长模式（varlen）
  - 变长模式仅支持 `B = 1`

- `gGammaOptional` 和 `aOptional`：
  - 接口层存在，但当前实现未启用
  - **必须传入空指针，否则执行失败**

---

### 4.4 形状约束（强约束）

必须满足以下条件：

- `q, k`: `[B, H, T, K]`
- `dO`: `[B, H, T, V]`
- `g`: `[B, H, T]`

额外限制：

- `K = 128`
- `V = 128`
- `chunkSize ∈ {64, 128}`

---

### 4.5 变长模式（VarLen）

当提供 `cuSeqlensOptional` 时：

- `chunkIndicesOptional` 必须同时提供
- 当前实现仅支持：

```text
B = 1
```

---

### 4.6 数值语义

- `scale`：
  - 必须显式传入
  - 推荐设置为：

```text
1 / sqrt(K)
```

---

## 5. Torch 测试调用示例

```python
import torch
import torch_npu
import math

def test_chunk_bwd_dv_local():
    # 参数设置
    B, H, T, K, V = 1, 8, 32768, 128, 128
    chunk_size = 64
    scale = 1.0 / math.sqrt(K)
    device = "npu:0"
    dtype = torch.bfloat16

    # 构造输入
    q = torch.randn(B, H, T, K, device=device, dtype=dtype)
    k = torch.randn(B, H, T, K, device=device, dtype=dtype)
    d_o = torch.randn(B, H, T, V, device=device, dtype=dtype)
    g = torch.randn(B, H, T, device=device, dtype=torch.float32)

    # 调用算子
    dv = torch_npu.npu_chunk_bwd_dv_local(
        q, k, d_o, g,
        g_gamma=None,
        A=None,
        cu_seqlens=None,
        chunk_indices=None,
        scale=scale,
        chunk_size=chunk_size
    )

    print("dv shape:", dv.shape)
    assert dv.shape == (B, H, T, V)
    print("Execution Successful!")

if __name__ == "__main__":
    test_chunk_bwd_dv_local()
```

---

## 6. 目录结构

```text
chunk_bwd_dv_local/
├── examples/
│   └── test_aclnn_chunk_bwd_dv_local_variable.cpp
├── op_api/
│   ├── aclnn_chunk_bwd_dv_local.cpp
│   ├── aclnn_chunk_bwd_dv_local.h
│   ├── chunk_bwd_dv_local.cpp
│   └── chunk_bwd_dv_local.h
├── op_host/
│   ├── chunk_bwd_dv_local_def.cpp
│   ├── chunk_bwd_dv_local_tiling.cpp
│   ├── chunk_bwd_dv_local_tiling.h
│   └── CMakeLists.txt
└── op_kernel/
    ├── chunk_bwd_dv_local_common.h
    ├── chunk_bwd_dv_local_cube.h
    ├── chunk_bwd_dv_local_vector.h
    └── chunk_bwd_dv_local.cpp
```
