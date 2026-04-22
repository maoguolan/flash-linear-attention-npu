# RecurrentGatedDeltaRule 算子说明
🔗 [查看源码](https://gitcode.com/Wei_NaChuan/flash-linear-attention-npu/tree/main/chunk_gated_delta_rule/recurrent_gated_delta_rule)
`RecurrentGatedDeltaRule` 是用于循环门控 Delta 规则（Recurrent Gated Delta Rule，RGDR）的自定义算子，主要应用于线性注意力机制的推理场景。该算子在每个时间步根据当前输入 $q_t, k_t, v_t$ 和上一隐藏状态 $S_{t-1}$，计算当前输出 $o_t$ 并更新隐藏状态 $S_t$。

---

## 1. 算子功能

在每个时间步 $t$，网络根据输入计算注意力输出并更新隐藏状态：

- **out**：当前时间步注意力输出
- **state**：更新后的隐藏状态（原地更新）

计算公式为：

$$S_t := \alpha_t \cdot \text{Diag}(\alpha_{kt}) \cdot S_{t-1} + \beta_t \cdot (v_t - \alpha_t \cdot \text{Diag}(\alpha_{kt}) \cdot S_{t-1} k_t) k_t^T$$

$$o_t := \frac{S_t q_t}{\sqrt{d_k}}$$

其中：

| 符号 | 含义 | 形状 |
|------|------|------|
| $S_{t-1}, S_t$ | 隐藏状态矩阵 | $\mathbb{R}^{d_v \times d_k}$ |
| $q_t, k_t$ | Query / Key 向量 | $\mathbb{R}^{d_k}$ |
| $v_t$ | Value 向量 | $\mathbb{R}^{d_v}$ |
| $\alpha_t = e^{g_t}$ | 标量门控衰减系数 | $\mathbb{R}$ |
| $\alpha_{kt} = e^{gk_t}$ | 逐维门控衰减系数 | $\mathbb{R}^{d_k}$ |
| $\beta_t$ | 更新门控系数 | $\mathbb{R}$ |
| $o_t$ | 当前时间步输出 | $\mathbb{R}^{d_v}$ |

---

## 2. 接口定义

### 2.1 ACLNN 接口

每个算子分为两段式调用流程：

1. **获取 workspace 与执行器**  
   调用 `aclnnRecurrentGatedDeltaRuleGetWorkspaceSize` 接口，获取算子执行所需的 workspace 大小，并创建执行器（executor）。

2. **执行算子计算**  
   调用 `aclnnRecurrentGatedDeltaRule` 接口，在指定的 workspace 和执行器下完成计算。

对应以下 C++ 接口：
```cpp
// 获取执行所需的 workspace 大小
aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *beta,
    aclTensor *stateRef,
    const aclTensor *actualSeqLengths,
    const aclTensor *ssmStateIndices,
    const aclTensor *g,
    const aclTensor *gk,
    const aclTensor *numAcceptedTokens,
    float scaleValue,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

// 执行算子
aclnnStatus aclnnRecurrentGatedDeltaRule(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

---

## 3. 参数说明

### 3.1 输入参数（Inputs）

令 $B$ 为 batch size，$T = \sum_i L_i$ 为累积序列长度，$N_k$ 为 key 头数，$N_v$ 为 value 头数，$D_k$ 为 key 维度，$D_v$ 为 value 维度，BlockNum 为状态块总数。

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|---|---|
| `query` | 输入 | 必选 | 公式中的 $q$；不支持空 Tensor | 接口执行前会先转为连续内存 | `BFLOAT16` | `ND` | `(T, Nk, Dk)` | 支持 |
| `key` | 输入 | 必选 | 公式中的 $k$；不支持空 Tensor | 接口执行前会先转为连续内存 | `BFLOAT16` | `ND` | `(T, Nk, Dk)` | 支持 |
| `value` | 输入 | 必选 | 公式中的 $v$；不支持空 Tensor | 接口执行前会先转为连续内存 | `BFLOAT16` | `ND` | `(T, Nv, Dv)` | 支持 |
| `beta` | 输入 | 必选 | 公式中的 $\beta$；不支持空 Tensor | 接口执行前会先转为连续内存 | `BFLOAT16` | `ND` | `(T, Nv)` | 支持 |
| `stateRef` | 输入&输出 | 必选 | 状态矩阵 $S$，算子执行后原地更新；不支持空 Tensor | **不支持非连续 Tensor**，算子执行后直接写入其存储 | `BFLOAT16` | `ND` | `(BlockNum, Nv, Dv, Dk)` | 不支持 |
| `actualSeqLengths` | 输入 | 必选 | 各 batch 的有效序列长度；不支持空 Tensor | 各元素之和等于 $T$ | `INT32` | `ND` | `(B,)` | 支持 |
| `ssmStateIndices` | 输入 | 必选 | 输入序列到状态矩阵的映射索引，`state[ssmStateIndices[i]]` 表示第 $i$ 个 token 对应的状态块；不支持空 Tensor | 取值范围 `[0, BlockNum)` | `INT32` | `ND` | `(T,)` | 支持 |
| `g` | 输入 | 可选 | 标量衰减系数 $\alpha_t = e^g$ | 传 `nullptr` 时等价于全 0（$\alpha_t = 1$，即无标量衰减） | `FLOAT32` | `ND` | `(T, Nv)` | 支持 |
| `gk` | 输入 | 可选 | 逐维衰减系数 $\alpha_{kt} = e^{gk}$ | 传 `nullptr` 时等价于全 0（$\alpha_{kt} = \mathbf{1}$，即无逐维衰减） | `FLOAT32` | `ND` | `(T, Nv, Dk)` | 支持 |
| `numAcceptedTokens` | 输入 | 可选 | 每个序列接受的 token 数量 | 传 `nullptr` 时默认全部接受 | `INT32` | `ND` | `(B,)` | 支持 |

### 3.2 属性参数（Attributes）

| 参数名 | 输入/输出 | 必选/可选 | 描述 | 使用说明 | 数据类型 | 取值约束 |
|---|---|---|---|---|---|---|
| `scaleValue` | 输入 | 可选（默认 `1.0`） | Query 的缩放因子 | `op_def` 默认值为 `1.0f`；推荐按 `1 / sqrt(Dk)` 设置 | `float` | 推荐 `1 / sqrt(Dk)` |

### 3.3 输出参数（Outputs）

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度（Shape） | 非连续 Tensor |
|---|---|---|---|---|---|---|
| `out` | 输出 | 公式中的 $o$，当前时间步注意力输出 | `BFLOAT16` | `ND` | `(T, Nv, Dv)` | 支持 |
| `stateRef` | 输入&输出 | 更新后的隐藏状态矩阵（原地更新） | `BFLOAT16` | `ND` | `(BlockNum, Nv, Dv, Dk)` | 不支持 |
| `workspaceSize` | 输出 | Device 侧所需 workspace 大小 | `uint64_t` | - | 标量 | - |
| `executor` | 输出 | 算子执行器，封装了计算流程 | `aclOpExecutor*` | - | - | - |

### 3.4 形状与约束

- `query`、`key` 的形状为 `(T, Nk, Dk)`。
- `value` 的形状为 `(T, Nv, Dv)`。
- `beta` 的形状为 `(T, Nv)`。
- `stateRef` 的形状为 `(BlockNum, Nv, Dv, Dk)`，不支持非连续 Tensor。
- `actualSeqLengths` 为长度 $B$ 的一维 INT32 张量，各元素之和等于 $T$。
- `ssmStateIndices` 为长度 $T$ 的一维 INT32 张量，取值范围 `[0, BlockNum)`。
- 当前仅支持 `BFLOAT16` 精度（query/key/value/beta/stateRef/out）。

### 3.5 补充说明

- 输入张量 `query/key/value/beta/g/gk` 等支持非连续 Tensor 输入，接口层会自动进行连续化处理。
- `stateRef` 为原地输入输出，**不支持非连续 Tensor**，算子执行后直接写入其存储。
- 输出 `out` 若为非连续视图，将通过内部 `ViewCopy` 回写结果。

---

## 4. 调用约束与执行语义

### 4.1 执行流程

该算子采用两阶段执行模型：

1. 调用 `aclnnRecurrentGatedDeltaRuleGetWorkspaceSize`：
   - 完成参数合法性检查
   - 构建执行器（executor）
   - 返回所需 `workspaceSize`

2. 调用 `aclnnRecurrentGatedDeltaRule`：
   - 在指定 `workspace` 和 `executor` 下执行计算
   - 内部通过 AICore kernel 完成逐 token 循环更新

---

### 4.2 内存行为

- 输入张量 `query/key/value/beta/g/gk`：
  - 若为非连续 Tensor，将在内部自动转换为连续布局再参与计算
- `stateRef`：
  - 原地输入输出，**不支持非连续 Tensor**
- 输出张量 `out`：
  - 若为非连续视图，将通过内部 `ViewCopy` 回写结果

---

### 4.3 形状约束（强约束）

必须满足以下条件：

- `query, key`: `(T, Nk, Dk)`
- `value`: `(T, Nv, Dv)`
- `beta`: `(T, Nv)`
- `stateRef`: `(BlockNum, Nv, Dv, Dk)`
- `actualSeqLengths`: `(B,)`，各元素之和等于 `T`
- `ssmStateIndices`: `(T,)`，取值范围 `[0, BlockNum)`

---

### 4.4 数值语义

- `g` 传 `nullptr` 时：$\alpha_t = e^0 = 1$，即无标量衰减
- `gk` 传 `nullptr` 时：$\alpha_{kt} = \mathbf{1}$，即无逐维衰减
- `scaleValue` 推荐设置为：

```text
1 / sqrt(Dk)
```

---

## 5. ACLNN C++ 测试调用示例

完整调用示例见 [test_aclnn_recurrent_gated_delta_rule.cpp](./examples/test_aclnn_recurrent_gated_delta_rule.cpp)。

```cpp
#include "aclnnop/aclnn_recurrent_gated_delta_rule.h"

int main()
{
    // 1. 初始化 ACL、device、stream（参考 aclInit / aclrtSetDevice / aclrtCreateStream）
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    // ... 省略初始化代码 ...

    // 2. 构造输入张量（示例参数）
    int32_t batchSize = 32, mtp = 2, headKNum = 4, headVNum = 8, dimV = 128, dimK = 128;
    // query/key: (T, Nk, Dk) = (batchSize*mtp, headKNum, dimK)
    // value/out: (T, Nv, Dv) = (batchSize*mtp, headVNum, dimV)
    // beta/g:    (T, Nv)     = (batchSize*mtp, headVNum)
    // stateRef:  (BlockNum, Nv, Dv, Dk) = (batchSize*mtp, headVNum, dimV, dimK)
    // actualSeqLengths: (B,) = (batchSize,)
    // ssmStateIndices:  (T,) = (batchSize*mtp,)
    // ... 省略 tensor 创建代码，参考完整示例 ...

    // 3. 调用第一段接口获取 workspace 大小
    uint64_t workspaceSize = 0;
    float scaleValue = 1.0f;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
        query, key, value, beta, stateRef,
        actSeqLen, ssmStaId,
        gama, gamak, numAccTok,
        scaleValue, attnOut,
        &workspaceSize, &executor);

    // 4. 申请 workspace
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 5. 调用第二段接口执行算子
    ret = aclnnRecurrentGatedDeltaRule(workspaceAddr, workspaceSize, executor, stream);

    // 6. 同步等待
    aclrtSynchronizeStream(stream);

    // 7. 释放资源（参考完整示例）
    return 0;
}
```

---

## 6. 目录结构

```text
recurrent_gated_delta_rule/
├── docs/
│   └── aclnnRecurrentGatedDeltaRule.md
├── examples/
│   └── test_aclnn_recurrent_gated_delta_rule.cpp
├── op_host/
│   ├── op_api/
│   │   ├── aclnn_recurrent_gated_delta_rule.cpp
│   │   ├── aclnn_recurrent_gated_delta_rule.h
│   │   ├── recurrent_gated_delta_rule.cpp
│   │   └── recurrent_gated_delta_rule.h
│   ├── recurrent_gated_delta_rule_def.cpp
│   ├── recurrent_gated_delta_rule_infershape.cpp
│   ├── recurrent_gated_delta_rule_tiling.cpp
│   ├── recurrent_gated_delta_rule_tiling.h
│   └── CMakeLists.txt
└── op_kernel/
    ├── recurrent_gated_delta_rule_tiling_data.h
    ├── recurrent_gated_delta_rule.h
    └── recurrent_gated_delta_rule.cpp
```
