# Weights Tab — Feature Spec

## 1. Overview

在现有 KV Cache Calculator 基础上新增 **Weights** 页面，计算模型权重（parameters）的总量和构成。

**核心原则**：
- 和 KV Cache tab 对称 —— 只算总量，不涉及并行策略（TP/PP/EP）
- 并行策略和 per-GPU 计算留给未来的 Deploy tab
- 纯静态页面，无构建工具，与现有架构一致

**Tab 结构**：

```
● LLM Memory Calculator    [KV Cache]  [Weights]  [Compare]     🌙
```

- KV Cache → index.html（现有，不改）
- Weights → weights.html（新增）
- Compare → compare.html（现有，不改）

**品牌名变更**：`KV Cache Calculator` → `LLM Memory Calculator`（三个页面同步改）

---

## 2. 页面布局

与 index.html 对称的双栏布局：

```
┌─── Form Card ──────────────┐  ┌─── Result Area ──────────────────────┐
│                             │  │                                       │
│  Select model               │  │  Total Parameters   671.0 B          │
│  ┌─────────────────────┐    │  │  1,342.0 GiB (BF16)                  │
│  │ 🔍 Search models... │    │  │                                       │
│  │  ▼ DeepSeek          │    │  │  [GiB] [GB] [MiB]                    │
│  │    V4 Pro      ✓     │    │  │                                       │
│  │  ▼ GLM               │    │  │  Metrics                              │
│  │  ...                  │    │  │  Attn  45.2 GiB · FFN  1,267.8 GiB  │
│  └─────────────────────┘    │  │  Embed  29.0 GiB                      │
│                             │  │                                       │
│  ── Selected ──             │  │  Weight Breakdown (ibar)              │
│  [● DeepSeek V4 Pro ×]     │  │  ████████████████████████████████████ │
│                             │  │  ██ Attn  ██ Dense  ██ Shared        │
│  Weight precision           │  │  ██ Experts  ██ Embed                │
│  [BF16] [FP8] [FP4]        │  │                                       │
│                             │  │  Layer Patterns                       │
│                             │  │  [Dense L0  ][MoE L1-L60 ×60]       │
│                             │  │                                       │
│                             │  │  Breakdown ▼                          │
│                             │  │  Attention per layer    1.2B          │
│                             │  │  Dense FFN per layer    3.4B          │
│                             │  │  Shared expert          1.8B          │
│                             │  │  MoE experts          664.6B          │
│                             │  │  Embeddings            29.0B          │
│                             │  │  Total params         671.0B          │
│                             │  │                                       │
│                             │  │  Note                                 │
│                             │  │  Source: huggingface.co/...           │
└─────────────────────────────┘  └───────────────────────────────────────┘
```

### 2.1 Form Card 控件

| 控件 | 类型 | 说明 |
|---|---|---|
| Model picker | 搜索式 Picker | 复用现有组件，单选模式 |
| Selected tag | Tag | 复用现有组件 |
| Weight precision | Segmented control | BF16 / FP8 / FP4（独立于 KV precision） |

**注意**：Weights 页面没有 Context length、Batch size、Indexer precision、Draft toggle、Linear toggle。权重是模型固有属性，不随推理参数变化。

### 2.2 Result Area

| 区域 | 说明 |
|---|---|
| Total Parameters | 参数量（如 671.0 B），**不依赖 precision** |
| Total Weight Size | 字节数（如 1,342.0 GiB），**依赖 precision** |
| Unit selector | GiB / GB / MiB，与 KV Cache 一致 |
| Metrics | Attn / FFN / Embed 各自大小，紧凑排列 |
| Weight Breakdown (ibar) | 水平色条，展示各组件占比 |
| Layer Patterns | 层模式图，展示 dense/MoE 层的构成比例 |
| Breakdown | 详细 key-value 列表（可折叠） |
| Note | 数据来源说明 |
| Source link | HuggingFace config 链接（复用现有） |

---

## 3. 数据模型变更 (data.js)

### 3.1 新增 weight_precision_options

```javascript
const MODEL_DATA = {
  // ... existing fields ...

  weight_precision_options: [
    { id: "bf16",       label: "BF16",  bytes_per_element: 2 },
    { id: "fp8_int8",   label: "FP8",   bytes_per_element: 1 },
    { id: "fp4_int4",   label: "FP4",   bytes_per_element: 0.5 }
  ],
```

### 3.2 每个 model 新增 weight_fields

`weight_fields` 是一个新对象，包含计算权重所需但现有 `fields` 中缺失的字段。
已有字段（如 `num_hidden_layers`, `num_key_value_heads`, `head_dim` 等）不重复。

#### 通用字段（所有模型都需要）

| 字段 | 类型 | 说明 | 示例 |
|---|---|---|---|
| `hidden_size` | number | 隐藏层维度 | 7168 |
| `num_attention_heads` | number | Q head 数 | 128 |
| `vocab_size` | number | 词表大小 | 129280 |
| `tie_word_embeddings` | boolean | 是否共享 embed 和 lm_head | false |
| `ffn_type` | string | "swiglu" 或 "gelu" | "swiglu" |

#### FFN 字段（Dense 模型）

| 字段 | 类型 | 说明 | 示例 |
|---|---|---|---|
| `intermediate_size` | number | Dense FFN 中间维度 | 18432 |

#### MoE 字段

| 字段 | 类型 | 说明 | 示例 |
|---|---|---|---|
| `num_experts` | number | 总 expert 数（含 shared） | 256 |
| `n_routed_experts` | number | 路由 expert 数 | 256 |
| `n_shared_experts` | number | 共享 expert 数 | 1 |
| `moe_intermediate_size` | number | 每个 expert 的中间维度 | 2048 |
| `shared_expert_intermediate_size` | number | shared expert 的中间维度（若不同） | 2048 |
| `first_k_dense_replace` | number | 前几层使用 dense FFN | 3 |
| `moe_layer_freq` | array | 每层是否为 MoE（0=dense, 1=MoE） | [0,1,1,...] |

#### MLA 专用字段

| 字段 | 类型 | 说明 | 示例 |
|---|---|---|---|
| `q_lora_rank` | number | Q 压缩维度 | 1536 |
| `qk_nope_head_dim` | number | QK 非 RoPE 维度 | 128 |

注：`kv_lora_rank`, `qk_rope_head_dim`, `v_head_dim` 已在现有 `fields` 中。

#### DeepSeek V4 专用字段

| 字段 | 类型 | 说明 | 示例 |
|---|---|---|---|
| `q_lora_rank` | number | Q 压缩维度 | 1536 |
| `o_lora_rank` | number | O 压缩维度 | 1024 |
| `o_groups` | number | O 分组数 | 16 |

#### Mixed Full + Sliding 专用字段

| 字段 | 类型 | 说明 | 示例 |
|---|---|---|---|
| `intermediate_size` | number | Dense FFN 中间维度 | 16384 |
| `swa_num_attention_heads` | number | 滑动窗口 Q head 数 | 64 |

注：`v_head_dim`, `swa_v_head_dim` 等已在现有 `fields` 中。

#### Linear Attention 专用字段

（已有 `linear_num_key_heads`, `linear_value_head_dim` 等在 `fields` 中）
额外需要：

| 字段 | 类型 | 说明 | 示例 |
|---|---|---|---|
| `linear_intermediate_size` | number | Linear 层 FFN 中间维度 | 可选 |

### 3.3 完整 weight_fields 示例

```javascript
// DeepSeek V4 Pro
{
  id: "deepseek-v4-pro",
  // ... existing fields ...
  weight_fields: {
    hidden_size: 7168,
    num_attention_heads: 128,
    vocab_size: 129280,
    tie_word_embeddings: false,
    ffn_type: "swiglu",
    q_lora_rank: 1536,
    qk_rope_head_dim: 64,
    o_lora_rank: 1024,
    o_groups: 16,
    n_routed_experts: 384,
    n_shared_experts: 1,
    moe_intermediate_size: 3072,
    num_experts_per_tok: 6,
    // 没有 intermediate_size / first_k_dense_replace → 所有层都是 MoE
  }
}

// DeepSeek V3
{
  id: "deepseek-v3",
  // ... existing fields ...
  weight_fields: {
    hidden_size: 7168,
    num_attention_heads: 128,
    vocab_size: 129280,
    tie_word_embeddings: false,
    ffn_type: "swiglu",
    q_lora_rank: 1536,
    intermediate_size: 18432,
    first_k_dense_replace: 3,
    n_routed_experts: 256,
    n_shared_experts: 1,
    moe_intermediate_size: 2048,
    num_experts_per_tok: 8,
  }
}

// Qwen3-8B (standard GQA, dense)
{
  id: "qwen3-8b",
  // ... existing fields ...
  weight_fields: {
    hidden_size: 4096,
    num_attention_heads: 32,
    vocab_size: 151936,
    tie_word_embeddings: false,
    ffn_type: "swiglu",
    intermediate_size: 12288,
  }
}

// Qwen3-235B (standard GQA, MoE)
{
  id: "qwen3-235b-a22b",
  // ... existing fields ...
  weight_fields: {
    hidden_size: 4096,
    num_attention_heads: 64,
    vocab_size: 151936,
    tie_word_embeddings: false,
    ffn_type: "swiglu",
    intermediate_size: 12288,
    n_routed_experts: 128,
    n_shared_experts: 0,
    moe_intermediate_size: 1536,
    num_experts_per_tok: 8,
    // mlp_only_layers: [] → 所有层都是 MoE
    // 但没有 first_k_dense_replace → 可能 Qwen3 的 MoE 没有前 N 层 dense
  }
}

// Llama 3.1 8B (standard GQA, dense)
{
  id: "llama-3.1-8b",
  // ... existing fields ...
  weight_fields: {
    hidden_size: 4096,
    num_attention_heads: 32,
    vocab_size: 128256,
    tie_word_embeddings: false,
    ffn_type: "swiglu",
    intermediate_size: 14336,
  }
}

// Gemma 4 26B-A4B (mixed full+sliding, MoE)
{
  id: "gemma-4-26b-a4b",
  // ... existing fields ...
  weight_fields: {
    hidden_size: 2816,
    num_attention_heads: 16,
    vocab_size: 262144,
    tie_word_embeddings: true,
    ffn_type: "gelu",          // Gemma uses GeGLU
    intermediate_size: 2112,
    n_routed_experts: 128,
    n_shared_experts: 0,
    moe_intermediate_size: 704,
    num_experts_per_tok: 8,
    // layer_types array determines full/sliding + dense/MoE per layer
  }
}

// MiMo-V2.5 (mixed full+sliding, MoE)
{
  id: "mimo-v2.5",
  // ... existing fields ...
  weight_fields: {
    hidden_size: 4096,
    num_attention_heads: 64,
    swa_num_attention_heads: 64,
    vocab_size: 152576,
    tie_word_embeddings: false,
    ffn_type: "swiglu",
    intermediate_size: 16384,
    n_routed_experts: 256,
    n_shared_experts: 0,
    moe_intermediate_size: 2048,
    num_experts_per_tok: 8,
    moe_layer_freq: [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  }
}

// Qwen3.5-397B-A17B (linear+full hybrid, MoE)
{
  id: "qwen3.5-397b-a17b",
  // ... existing fields ...
  weight_fields: {
    hidden_size: 4096,
    num_attention_heads: 32,
    vocab_size: 248320,
    tie_word_embeddings: false,
    ffn_type: "swiglu",
    n_routed_experts: 512,
    n_shared_experts: 0,
    moe_intermediate_size: 1024,
    shared_expert_intermediate_size: 1024,
    num_experts_per_tok: 10,
  }
}

// MiniMax M3 (MSA GQA, MoE)
{
  id: "minimax-m3",
  // ... existing fields ...
  weight_fields: {
    hidden_size: 4096,
    num_attention_heads: 48,
    vocab_size: ...,
    tie_word_embeddings: false,
    ffn_type: "swiglu",
    // MoE fields - need to check actual config
    n_routed_experts: ...,
    moe_intermediate_size: ...,
  }
}
```

**注意**：MiniMax M2 系列和 Cohere Command R 系列的 weight_fields 需要从实际 config 补全。
部分模型的 MoE 配置需要逐一从 HuggingFace config 确认。

---

## 4. 计算逻辑 (js/calc_weight.js)

新建 `js/calc_weight.js`，纯函数，无 DOM 依赖。与 `calc.js` 同级。

### 4.1 函数签名

```javascript
/**
 * 计算模型权重总量和构成
 *
 * @param {Object} model - MODEL_DATA 中的模型条目
 * @param {number} wtPrecB - Weight 精度字节数 (2=BF16, 1=FP8, 0.5=FP4)
 * @returns {{
 *   totalParams: number,        // 总参数量
 *   totalBytes: number,         // 总权重字节数
 *   attnParams: number,         // Attention 总参数
 *   ffnDenseParams: number,     // Dense FFN 总参数
 *   ffnSharedParams: number,    // Shared Expert 总参数
 *   ffnExpertParams: number,    // Routed Experts 总参数
 *   embedParams: number,        // Embedding 总参数
 *   breakdown: Array,           // 详细分解列表
 *   formulas: Array,            // 公式展示行
 *   formulaTitle: string,       // 公式标题
 *   patterns: Array,            // Layer pattern ibar 数据
 *   legendTypes: Array,         // 图例类型
 * }}
 */
function calcWeight(model, wtPrecB) { ... }
```

### 4.2 各架构 Attention 权重公式

#### 4.2.1 Standard GQA（Qwen3, Llama, MiniMax M2, Cohere）

```
attn_per_layer = W_q + W_k + W_v + W_o

W_q = h × (n_q × d_h)
W_k = h × (h_kv × d_h)
W_v = h × (h_kv × d_h)
W_o = (n_q × d_h) × h

// 当 d_v ≠ d_h 时（如 MiMo）：
W_v = h × (h_kv × d_v)
W_o = (n_q × d_v) × h
```

默认 `d_v = d_h`，除非模型有 `v_head_dim` 字段。

#### 4.2.2 MLA（DeepSeek V3, R1, Kimi K2.5/K2.6）

```
attn_per_layer = W_q_a + W_q_b + W_kv_a + W_kv_b + W_o

W_q_a  = h × q_lora_rank                           // Q down projection
W_q_b  = q_lora_rank × (n_q × qk_head_dim)         // Q up projection
W_kv_a = h × (kv_lora_rank + qk_rope_head_dim)      // KV down + RoPE key
W_kv_b = kv_lora_rank × (n_q × (qk_nope + d_v))    // KV up projection (per-head)
W_o    = (n_q × d_v) × h                            // Output projection

其中 qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
```

**W_kv_b 是 MLA 中最大的 attention 权重矩阵**。
它将压缩 latent 展开到每个 Q head 的 K 和 V，所以是 `kv_lora_rank × n_q × (qk_nope + d_v)`。

#### 4.2.3 DSA+MLA（DeepSeek V3.2, GLM-5）

与 MLA 相同的 attention 权重公式。Indexer 不增加额外权重
（indexer head 是从 KV latent 中的额外投影，权重已包含在 W_kv_b 中）。

#### 4.2.4 DeepSeek V4 Hybrid

V4 的 attention 结构与 MLA 不同：

```
attn_per_layer = W_q_a + W_q_b + W_k + W_v + W_o_down + W_o_up

W_q_a    = h × q_lora_rank                         // Q down
W_q_b    = q_lora_rank × (n_q × head_dim)          // Q up (head_dim=512, 非 MLA 式拆分)
W_k      = h × head_dim                             // K projection (1 KV head)
W_v      = h × head_dim                             // V projection (1 KV head)
W_o_down = (n_q × head_dim) × (o_lora_rank × o_groups)  // O down (grouped LoRA)
W_o_up   = (o_lora_rank × o_groups) × h                  // O up
```

**注意**：V4 的 O projection 使用 grouped LoRA，参数量远小于标准 O projection。
标准 O: `(n_q × head_dim) × h = 128 × 512 × 7168 = 469M`
LoRA O: `(n_q × head_dim) × (o_lora_rank × o_groups) + (o_lora_rank × o_groups) × h`
     `= 65536 × (1024 × 16) + (1024 × 16) × 7168`
     `= 65536 × 16384 + 16384 × 7168`
     `= 1,073,741,824 + 117,440,512`
     `= 1,191,182,336`

Hmm, 这反而比标准 O 更大。需要确认 V4 的 O projection 实现。

**决策**：V4 的 O projection 逻辑需要从实际 modeling 代码确认。
V1 先按简化版本实现（标准 O projection），标注为 "待确认"，
后续从 HuggingFace 源码修正。

#### 4.2.5 Mixed Full + Sliding GQA（Gemma 4, Cohere Command R7B, MiMo）

Full attention 层和 Sliding attention 层使用不同的参数：

```
full_attn_per_layer = W_q_f + W_k_f + W_v_f + W_o_f

W_q_f = h × (n_q_f × d_f)         // n_q_f = 全局 Q head 数
W_k_f = h × (h_f × d_f)           // h_f = 全局 KV head 数
W_v_f = h × (h_f × d_vf)          // d_vf = 全局 V head dim
W_o_f = (n_q_f × d_vf) × h

sliding_attn_per_layer = W_q_s + W_k_s + W_v_s + W_o_s

W_q_s = h × (n_q_s × d_s)         // n_q_s = 滑动窗口 Q head 数
W_k_s = h × (h_s × d_s)           // h_s = 滑动窗口 KV head 数
W_v_s = h × (h_s × d_vs)          // d_vs = 滑动窗口 V head dim
W_o_s = (n_q_s × d_vs) × h
```

**注意**：
- 对于 Gemma 4：`n_q_f` 需要从 `num_attention_heads` 计算（不一定等于 `num_key_value_heads`）
  - `num_attention_heads` 是滑动窗口的 head 数
  - 全局 full attention 有自己的 `num_global_key_value_heads`
  - 但没有 `num_global_attention_heads` → 推测 full attention 也用 `num_attention_heads` 个 Q head
  - 需要从实际代码确认

- 对于 MiMo：`swa_num_attention_heads` 明确给出滑动窗口 Q head 数，
  `num_attention_heads` 是 full attention 的 Q head 数

#### 4.2.6 Qwen Linear + Full Hybrid

Full attention 层使用 Standard GQA 公式。
Linear attention 层的 Q/K/V 投影权重：

```
linear_attn_per_layer = W_q_l + W_k_l + W_v_l + W_o_l

W_q_l = h × (n_q × d_h)           // 与 full attention 相同（假设共享 Q 投影）
W_k_l = h × (h_kl × d_kl)         // linear key heads × key dim
W_v_l = h × (h_vl × d_vl)         // linear value heads × value dim
W_o_l = (n_q × d_h) × h           // 与 full attention 相同（假设共享 O 投影）
```

**注意**：Linear attention 层的 Conv/Recurrent state 参数量极小，忽略不计。

#### 4.2.7 MSA GQA（MiniMax M3）

与 Standard GQA 相同的 attention 公式。
Sparse index 分支的 K_idx projection 是额外权重：

```
W_k_idx = h × (h_idx × d_idx)     // sparse index key projection
```

仅存在于 sparse layers。

### 4.3 FFN 权重公式

#### 4.3.1 Dense FFN（SwiGLU: 3 矩阵）

```
dense_ffn_per_layer = 3 × h × intermediate_size

// gate_proj: h × intermediate_size
// up_proj:   h × intermediate_size
// down_proj: intermediate_size × h
```

#### 4.3.2 Dense FFN（GeGLU/GELU: 2 矩阵）

```
dense_ffn_per_layer = 2 × h × intermediate_size

// up_proj:   h × intermediate_size
// down_proj: intermediate_size × h
```

#### 4.3.3 MoE FFN

```
shared_expert_per_layer = n_shared × ffn_mats × h × shared_expert_intermediate_size
expert_per_layer        = n_routed × ffn_mats × h × moe_intermediate_size

moe_ffn_per_layer = shared_expert_per_layer + expert_per_layer
```

其中 `ffn_mats = 3`（SwiGLU）或 `2`（GELU）。

**注意**：
- Router/gate 的权重（`n_routed × h`）占比 <0.01%，忽略
- `shared_expert_intermediate_size` 若未指定，默认 = `moe_intermediate_size`
- 若 `n_shared_experts = 0` 或未指定，shared expert 部分为 0

### 4.4 Embedding 权重

```
if tie_word_embeddings:
  embed_params = V × h               // only embed_tokens
else:
  embed_params = 2 × V × h           // embed_tokens + lm_head
```

### 4.5 层分类逻辑

每层由 (attention_type, ffn_type) 二元组定义：

| 模型 | 层分类逻辑 |
|---|---|
| Standard GQA dense | 所有层: (gqa, dense_ffn) |
| Standard GQA MoE (Qwen3-235B) | 前N层: (gqa, dense_ffn)，其余: (gqa, moe_ffn) |
| MLA dense+MoE (DeepSeek V3) | 前N层: (mla, dense_ffn)，其余: (mla, moe_ffn) |
| MLA MoE only (Kimi) | 所有层: (mla, moe_ffn) |
| DSA+MLA MoE (DeepSeek V3.2) | 所有层: (dsa_mla, moe_ffn) |
| DeepSeek V4 | 所有层: (v4_attn, moe_ffn) |
| Mixed (Gemma 4) | full_attn+MoE / sliding_attn+MoE |
| Mixed (MiMo) | full_attn+dense / full_attn+MoE / sliding_attn+dense / sliding_attn+MoE |
| Linear+Full (Qwen3.5) | full_attn+MoE / linear_attn+MoE |
| MSA (MiniMax M3) | full_attn+MoE / sparse_attn+MoE |

**分类规则**：
1. 若有 `first_k_dense_replace` → 前N层用 dense FFN，其余用 MoE
2. 若有 `moe_layer_freq` 数组 → 0=dense, 1=MoE
3. 若有 `layer_types` 数组 → 根据类型映射
4. 若有 `n_routed_experts` 但无 dense/moe 分层 → 所有层 MoE
5. 若无 `n_routed_experts` → 所有层 dense

### 4.6 总量汇总

```
total_params = Σ_layers(attn_params + ffn_params) + embed_params

attnParams    = Σ_layers(attn_per_layer_for_this_type)
ffnDenseParams  = Σ_dense_layers(dense_ffn_per_layer)
ffnSharedParams = Σ_moe_layers(shared_expert_per_layer)
ffnExpertParams = Σ_moe_layers(expert_per_layer)
embedParams     = embed_params

totalBytes = totalParams × wtPrecB
```

### 4.7 Ibar / Layer Patterns 数据

与 KV Cache 的 patterns 结构对齐：

```javascript
patterns = [
  {
    segs: [
      { type: 'attn', ratio: attnBytes / totalPerLayer },
      { type: 'ffn-dense', ratio: ffnDenseBytes / totalPerLayer },
    ],
    count: 3,                // 前N层
    label: 'dense FFN',
    bytes: totalPerLayer,
  },
  {
    segs: [
      { type: 'attn', ratio: attnBytes / totalPerLayer },
      { type: 'ffn-shared', ratio: ffnSharedBytes / totalPerLayer },
      { type: 'ffn-expert', ratio: ffnExpertBytes / totalPerLayer },
    ],
    count: 58,               // MoE 层
    label: 'MoE FFN',
    bytes: totalPerLayer,
  },
]
```

### 4.8 新增 ibar 颜色

```css
:root {
  /* Weight ibar colors */
  --bar-attn:       #4263eb;   /* 蓝 - Attention */
  --bar-ffn-dense:  #f59e0b;   /* 琥珀 - Dense FFN */
  --bar-ffn-shared: #e67700;   /* 橙 - Shared Expert */
  --bar-ffn-expert: #e03131;   /* 红 - Routed Experts */
  --bar-embed:      #9c36b5;   /* 紫 - Embeddings */
}
```

### 4.9 新增 SYMBOL_NAMES 条目

```javascript
// Weight-specific symbol names (追加到现有 SYMBOL_NAMES)
var WEIGHT_SYMBOL_NAMES = {
  h: 'hidden_size', V: 'vocab_size',
  n_q: 'num_attention_heads',
  d_v: 'v_head_dim',
  q_r: 'q_lora_rank', o_r: 'o_lora_rank',
  qk: 'qk_head_dim', qk_nope: 'qk_nope_head_dim',
  I: 'intermediate_size', I_m: 'moe_intermediate_size',
  I_s: 'shared_expert_intermediate_size',
  N_e: 'n_routed_experts', N_s: 'n_shared_experts',
  L_d: 'dense_ffn_layers', L_m: 'moe_ffn_layers',
};
```

### 4.10 公式展示

#### Standard GQA + Dense FFN

```
Attn  = h × n_q × d_h + 2 × h × h_kv × d_h + n_q × d_h × h
FFN   = 3 × h × I
W_t   = L × (Attn + FFN) + Embed
```

#### Standard GQA + MoE

```
Attn   = h × n_q × d_h + 2 × h × h_kv × d_h + n_q × d_h × h
FFN_d  = 3 × h × I × L_d
FFN_s  = N_s × 3 × h × I_s × L_m
FFN_e  = N_e × 3 × h × I_m × L_m
W_t    = Attn × L + FFN_d + FFN_s + FFN_e + Embed
```

#### MLA + MoE

```
Attn   = h × q_r + q_r × n_q × qk + h × (d_c + d_r) + d_c × n_q × (qk_nope + d_v) + n_q × d_v × h
FFN_d  = 3 × h × I × L_d
FFN_s  = N_s × 3 × h × I_s × L_m
FFN_e  = N_e × 3 × h × I_m × L_m
W_t    = Attn × L + FFN_d + FFN_s + FFN_e + Embed
```

其他架构公式类似，遵循相同模式。

---

## 5. 页面实现 (weights.html)

### 5.1 文件结构

```
llm-mem-calculator/
├── index.html            # KV Cache（现有，仅改 nav 链接）
├── weights.html          # Weights（新增）
├── compare.html          # Compare（现有，仅改 nav 链接）
├── css/
│   └── main.css          # 共享样式（新增 weight ibar 颜色变量）
├── js/
│   ├── data.js           # 模型数据（新增 weight_fields, weight_precision_options）
│   ├── calc.js           # KV 计算引擎（不改）
│   ├── calc_weight.js    # Weight 计算引擎（新增）
│   ├── calculator.js     # KV 页面逻辑（不改）
│   ├── weights.js        # Weights 页面逻辑（新增）
│   └── compare.js        # Compare 页面逻辑（不改）
```

### 5.2 weights.html 结构

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <!-- 标准 meta, 与 index.html 对齐 -->
  <title>Model Weights Calculator — LLM Parameter & Memory Estimator</title>
  <!-- 字体 + main.css -->
</head>
<body>
<nav class="nav">
  <div class="nav-inner">
    <span class="nav-brand"><span class="nav-brand-dot"></span> LLM Memory Calculator</span>
    <div class="nav-tabs">
      <a href="index.html" class="nav-tab">KV Cache</a>
      <a href="weights.html" class="nav-tab active">Weights</a>
      <a href="compare.html" class="nav-tab">Compare</a>
    </div>
    <!-- github link + theme toggle + cheatsheet (复用) -->
  </div>
</nav>
<div class="shell">
  <div class="grid">
    <div class="form-card">
      <!-- Model picker (复用组件) -->
      <!-- Selected tag (复用组件) -->
      <!-- Weight precision segmented control -->
    </div>
    <div class="result-area">
      <!-- Total Parameters (大数字) -->
      <!-- Total Weight Size -->
      <!-- Unit selector -->
      <!-- Metrics compact -->
      <!-- Formula section -->
      <!-- Breakdown -->
      <!-- Note -->
      <!-- Source link -->
    </div>
  </div>
</div>
<script src="js/data.js"></script>
<script src="js/calc_weight.js"></script>
<script src="js/weights.js"></script>
</body>
</html>
```

### 5.3 weights.js 逻辑

与 calculator.js 对称，但更简单（无 tokens/batch/indexer/draft/linear 控件）：

```javascript
// 状态
var selectedModelId = 'deepseek-v4-pro';
var wtPrecValue = 'bf16';

// DOM 引用
var $modelPicker = document.getElementById('modelPicker');
var $selectedTags = document.getElementById('selectedTags');
var $wtPrecSeg = document.getElementById('wtPrecSeg');
var $totalParams = document.getElementById('totalParams');
var $totalValue = document.getElementById('totalValue');
var $totalUnit = document.getElementById('totalUnit');
var $metricsCompact = document.getElementById('metricsCompact');
var $formulaSection = document.getElementById('formulaSection');
var $formulaTitle = document.getElementById('formulaTitle');
var $formulaBody = document.getElementById('formulaBody');
var $breakdownGrid = document.getElementById('breakdownGrid');
var $noteSection = document.getElementById('noteSection');
var $sourceLink = document.getElementById('sourceLink');
var $themeToggle = document.getElementById('themeToggle');

// 初始化
initTheme();
$themeToggle.addEventListener('click', toggleTheme);

// Model picker (复用 calculator.js 的 buildPicker 模式)
// ...

// Precision control
initSegControl($wtPrecSeg, function(val) {
  wtPrecValue = val;
  calculate();
});

function getWtPrecBytes() {
  return MODEL_DATA.weight_precision_options.find(function(p) {
    return p.id === wtPrecValue;
  }).bytes_per_element;
}

function calculate() {
  var model = getModel();
  if (!model) { /* clear results */ return; }

  var wtPrecB = getWtPrecBytes();
  var result = calcWeight(model, wtPrecB);

  // 渲染 total params
  $totalParams.textContent = formatParams(result.totalParams);

  // 渲染 total bytes
  $totalValue.textContent = formatTotal(result.totalBytes);
  $totalUnit.textContent = getUnitLabel();

  // 渲染 metrics
  var html = '';
  html += '<span class="metric-item">Attention <span class="metric-val">' +
           formatMetric(result.attnParams * wtPrecB) + '</span></span>';
  html += '<span class="metric-sep">·</span>';
  html += '<span class="metric-item">FFN <span class="metric-val">' +
           formatMetric((result.ffnDenseParams + result.ffnSharedParams + result.ffnExpertParams) * wtPrecB) +
           '</span></span>';
  html += '<span class="metric-sep">·</span>';
  html += '<span class="metric-item">Embed <span class="metric-val">' +
           formatMetric(result.embedParams * wtPrecB) + '</span></span>';
  $metricsCompact.innerHTML = html;

  // 渲染 formula section
  // (与 calculator.js 的 formula 渲染逻辑对称)

  // 渲染 breakdown
  $breakdownGrid.innerHTML = result.breakdown.map(function(item) {
    var tip = item.tip ? ' ' + tipIcon(item.tip) : '';
    return '<div class="breakdown-row">' +
      '<span class="label">' + item.label + tip + '</span>' +
      '<span class="val">' + item.value + '</span>' +
    '</div>';
  }).join('');

  // Note & Source
  $noteSection.textContent = 'Parameter counts derived from official Hugging Face model configs. ' +
    'Excludes LayerNorm, biases, and router weights (<0.1% of total).';
  $sourceLink.href = model.source_url;
  $sourceLink.textContent = 'Source: ' + model.source_url;
}

// 格式化参数量
function formatParams(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + ' B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + ' M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + ' K';
  return n.toString();
}

// 初始化
buildPicker('');
renderTag();
calculate();
```

---

## 6. 导航变更

三个页面的 nav 统一改为：

```html
<nav class="nav">
  <div class="nav-inner">
    <span class="nav-brand"><span class="nav-brand-dot"></span> LLM Memory Calculator</span>
    <div class="nav-tabs">
      <a href="index.html" class="nav-tab">KV Cache</a>
      <a href="weights.html" class="nav-tab">Weights</a>
      <a href="compare.html" class="nav-tab">Compare</a>
    </div>
    <!-- ... -->
  </div>
</nav>
```

- 品牌名：`KV Cache Calculator` → `LLM Memory Calculator`
- Tab 标签：`Calculator` → `KV Cache`（更具体）
- 新增 `Weights` tab
- 每个页面的 active tab 不同

---

## 7. CSS 变更新增

### 7.1 Weight ibar 颜色变量

```css
:root {
  --bar-attn:       #4263eb;
  --bar-ffn-dense:  #f59e0b;
  --bar-ffn-shared: #e67700;
  --bar-ffn-expert: #e03131;
  --bar-embed:      #9c36b5;
}

[data-theme="dark"] {
  --bar-attn:       #5c7cfa;
  --bar-ffn-dense:  #fbbf24;
  --bar-ffn-shared: #f59e0b;
  --bar-ffn-expert: #ef4444;
  --bar-embed:      #c084fc;
}
```

### 7.2 Weight 页面新增样式

```css
/* Total Parameters 大数字 */
.total-params {
  font-size: 2rem;
  font-family: var(--mono);
  font-weight: 700;
  color: var(--accent2);
  line-height: 1.1;
}

.total-params-label {
  font-size: 0.8rem;
  color: var(--text3);
  margin-top: 2px;
}
```

其余样式（form-card, result-area, ibar, breakdown 等）完全复用现有 CSS。

---

## 8. 实施步骤

| # | 步骤 | 产出 | 影响文件 |
|---|---|---|---|
| 1 | **补充 data.js 的 weight_fields** | 每个 model 新增 weight_fields 对象 | `js/data.js` |
| 2 | **新增 weight_precision_options** | MODEL_DATA 新增字段 | `js/data.js` |
| 3 | **实现 calc_weight.js** | 纯计算函数，6 种架构各一个分支 | `js/calc_weight.js` (新) |
| 4 | **创建 weights.html** | HTML 骨架 | `weights.html` (新) |
| 5 | **实现 weights.js** | 页面交互逻辑 | `js/weights.js` (新) |
| 6 | **CSS 新增** | ibar 颜色变量 + total-params 样式 | `css/main.css` |
| 7 | **导航变更** | 三个页面统一 nav | `index.html`, `compare.html`, `weights.html` |
| 8 | **品牌名变更** | 标题、meta、JSON-LD 同步 | 三个 HTML 文件 |
| 9 | **验证** | 用 3-4 个模型验证参数量与 HuggingFace 报告一致 | 手动 |

### 8.1 步骤 1 的数据采集

需要从 HuggingFace config 逐一确认每个模型的 weight_fields。
部分模型可能缺少 `intermediate_size`（纯 MoE 模型可能没有）。

**优先级**：
1. DeepSeek V4 Pro / Flash — 最复杂，验证 MLA+MoE+hybrid
2. DeepSeek V3 — 经典 MLA+MoE
3. Qwen3-235B — 标准 GQA+MoE
4. Llama 3.1 8B — 标准 GQA+dense（最简单，做 baseline）
5. 其余模型按家族批量补全

### 8.2 验证方法

1. **参数总量验证**：与 HuggingFace 模型页面的 "Parameters" 数字对比
2. **与现有工具交叉验证**：用 `safetensors` 元数据确认总参数量
3. **逐组件验证**：Attention / FFN / Embedding 分别与模型代码对比

---

## 9. 已知限制 & 未来工作

| 限制 | 说明 | 处理方式 |
|---|---|---|
| LayerNorm / Bias 忽略 | 占比 <0.1% | 在 Note 中说明 |
| Router 权重忽略 | 占比 <0.01% | 在 Note 中说明 |
| DeepSeek V4 O projection | LoRA 实现细节待确认 | V1 先用简化公式，标注待确认 |
| 混合精度权重 | 部分模型 attention 和 FFN 用不同精度 | V1 不支持，留给 Deploy tab |
| MLA Absorption | absorption 改变的是推理时矩阵，不改变存储的权重 | Weights tab 展示存储权重（不含 absorption） |
| 多模态模型 | Qwen3.5、Gemma 4、MiMo 有 vision encoder | V1 只计算 text model 权重，vision 部分标注为 "excluded" |
| Activation memory | 不在 Weights tab 范围内 | 留给 Deploy tab |
| tie_word_embeddings | 需要精确的 flag | 从 config.json 的 `tie_word_embeddings` 字段获取 |

---

## 10. 不做的事

| 项目 | 原因 |
|---|---|
| 并行策略 (TP/PP/EP) | 留给 Deploy tab |
| Per-GPU 计算 | 留给 Deploy tab |
| Absorption | Weights 展示存储权重，absorption 不改变存储 |
| Activation memory | 留给 Deploy tab |
| Vision encoder 权重 | V1 只算 text model，预留字段 |
| 安全张量精确解析 | 参数量从 config.json 推算，足够精确 |
| Compare 页面的 Weight 对比 | 无比较价值，不做 |
| V4 O projection LoRA | V1 简化实现，后续修正 |
| DeepSeek V4 Indexer projection | 细节待确认，V1 先忽略 |

---

## 11. 开发计划

### 总体思路：分 5 个阶段，每阶段可独立验证

```
Phase 1: 计算引擎 + 典型模型验证（纯 JS，无 UI）
   ↓
Phase 2: 页面骨架 + 交互逻辑（weights.html + weights.js）
   ↓
Phase 3: 可视化（ibar + formula + layer patterns）
   ↓
Phase 4: 导航变更 + 品牌名变更
   ↓
Phase 5: 剩余模型批量补全
```

### Phase 1: 计算引擎 + 典型模型验证

**目标**：`calc_weight.js` 核心逻辑写完，用 5 个典型模型验证参数量正确。

**产出**：
- `js/calc_weight.js` — 纯函数，无 DOM
- `js/data.js` — 5 个典型模型新增 `weight_fields`

**5 个典型模型**（覆盖所有架构类型）：

| # | 模型 | 架构 | 验证重点 |
|---|---|---|---|
| 1 | Llama 3.1 8B | standard_gqa + dense_ffn | 最简单，baseline |
| 2 | Qwen3-235B-A22B | standard_gqa + moe_ffn | GQA + MoE |
| 3 | DeepSeek V3 | mla + dense_ffn + moe_ffn | MLA attention + 混合 dense/MoE |
| 4 | DeepSeek V4 Pro | v4_attn + moe_ffn | V4 hybrid attention + 纯 MoE |
| 5 | Qwen3.5-397B-A17B | linear+full + moe_ffn | Linear attention + MoE |

**子步骤**：

1.1 补全 5 个典型模型的 `weight_fields`（从 HuggingFace config 采集）
1.2 实现 `calc_weight.js` 的 `standard_gqa` 分支 + `dense_ffn` 逻辑
1.3 用 Llama 3.1 8B 验证 → 参数量应约 8.03B
1.4 实现 `moe_ffn` 逻辑
1.5 用 Qwen3-235B 验证 → 参数量应约 235B（总）/ 22B（active）
1.6 实现 `mla_attn` 逻辑
1.7 用 DeepSeek V3 验证 → 参数量应约 671B（总）/ 37B（active）
1.8 实现 `v4_attn` 逻辑（简化版 O projection）
1.9 用 DeepSeek V4 Pro 验证 → 与公开数据对比
1.10 实现 `linear_attn` 逻辑
1.11 用 Qwen3.5-397B-A17B 验证
1.12 实现 `mixed_full_sliding` + `msa_gqa` 的 weight 逻辑（先空壳，Phase 5 补全）

**验证方式**：
- 参数总量与 HuggingFace 模型页的 "Parameters" 数字对比
- Attention / FFN / Embedding 分量与模型代码或 safetensors metadata 交叉验证
- 在浏览器 console 调用 `calcWeight(model, 2)` 检查返回值

**完成标准**：
- 5 个典型模型的 `totalParams` 与公开数字偏差 <1%
- 所有分支无 console 报错
- `calcWeight` 返回结构完整（含 breakdown, formulas, patterns, legendTypes）

---

### Phase 2: 页面骨架 + 交互逻辑

**目标**：weights.html 可运行，数字正确，但可视化简陋。

**产出**：
- `weights.html` — HTML 骨架
- `js/weights.js` — 交互逻辑

**子步骤**：

2.1 创建 `weights.html`，复制 index.html 的 nav + shell 结构
2.2 修改 form-card：去掉 tokens/batch/indexer/draft/linear，只保留 model picker + weight precision
2.3 修改 result-area：替换为 total-params + total-bytes + metrics + breakdown
2.4 实现 `weights.js`：
    - Model picker 交互（复用 calculator.js 的 buildPicker 模式）
    - Weight precision 切换
    - `calculate()` 主函数：调用 `calcWeight()` → 渲染 DOM
    - Theme toggle（复用）
2.5 实现参数量格式化（`formatParams`: B/M/K）
2.6 实现字节数格式化（复用 `formatTotal` / `formatMetric`）
2.7 实现单位切换（GiB / GB / MiB）
2.8 实现 breakdown 渲染（复用 calculator.js 的模式）

**完成标准**：
- 选择模型 → 显示正确的参数总量和权重字节数
- 切换 precision → 数字实时更新
- 5 个典型模型都能正确展示
- Dark mode 正常

---

### Phase 3: 可视化

**目标**：ibar + formula + layer patterns 完整展示。

**产出**：
- `css/main.css` — 新增 weight ibar 颜色变量
- `js/calc_weight.js` — 补全 formulas / patterns / legendTypes 数据
- `js/weights.js` — formula section 渲染逻辑

**子步骤**：

3.1 CSS 新增 weight ibar 颜色变量（`--bar-attn`, `--bar-ffn-dense`, 等）
3.2 在 `calc_weight.js` 中为每个架构分支生成 formulas 数据（短符号公式行）
3.3 在 `calc_weight.js` 中为每个架构分支生成 patterns 数据（layer patterns ibar）
3.4 在 `weights.js` 中实现 formula section 渲染（复用 calculator.js 的 pill 渲染逻辑）
3.5 在 `weights.js` 中实现 layer patterns 渲染（复用 calculator.js 的 pattern 渲染逻辑）
3.6 新增 `WEIGHT_SYMBOL_NAMES` 映射表
3.7 新增 WEIGHT_BAR_COLOR_MAP / WEIGHT_BAR_HEX_MAP / WEIGHT_LEGEND_LABEL_MAP
3.8 实现 cheatsheet 更新（新增 weight 相关符号）

**完成标准**：
- 5 个典型模型都有完整的 ibar 分布图
- Formula 区展示正确的短符号公式
- Layer patterns 展示 dense/MoE 层分布
- Hover pill 显示正确的 tooltip
- 亮色/暗色主题下颜色正确

---

### Phase 4: 导航变更 + 品牌名变更

**目标**：三个页面统一导航和品牌名。

**产出**：
- `index.html` — nav 变更
- `compare.html` — nav 变更
- `weights.html` — nav 变更（Phase 2 已做，确认一致）
- 三个 HTML 文件的 `<title>` / meta / JSON-LD 更新

**子步骤**：

4.1 `index.html`：
    - 品牌名 `KV Cache Calculator` → `LLM Memory Calculator`
    - Tab `Calculator` → `KV Cache`
    - 新增 `Weights` tab 链接
    - `<title>` 保持不变（KV Cache Calculator — ...已经够具体）
4.2 `compare.html`：
    - 同上品牌名和 tab 变更
4.3 `weights.html`：
    - 确认 nav 与其他页面一致
4.4 更新 JSON-LD 的 `name` 字段
4.5 更新 Open Graph / Twitter Card 的 title（如需要）

**完成标准**：
- 三个页面 nav 一致，active tab 正确
- 品牌名统一为 `LLM Memory Calculator`
- Tab 间跳转正常

---

### Phase 5: 剩余模型批量补全

**目标**：所有现有模型都补全 weight_fields。

**产出**：
- `js/data.js` — 剩余 15+ 模型新增 weight_fields

**模型列表**（按家族分组）：

| 家族 | 模型 | 特殊处理 |
|---|---|---|
| DeepSeek | V4 Flash | 同 V4 Pro，层数和 expert 数不同 |
| DeepSeek | V3.2 | DSA+MLA + MoE |
| DeepSeek | R1 | 同 V3 架构 |
| GLM | GLM-5, 5.1 | DSA+MLA + MoE |
| Kimi | K2.5, K2.6 | MLA + dense（无 MoE）|
| Qwen3.5 | 27B, 9B, 4B, 2B, 0.8B, 35B-A3B, 122B-A10B | Linear+Full + MoE（小模型可能无 MoE）|
| Qwen3 | 32B, 30B-A3B, 14B, 8B, 4B, 1.7B, 0.6B | Standard GQA + dense |
| Qwen2.5 | 72B, 32B, 14B, 7B, Coder-32B | Standard GQA + dense |
| Llama | 3.1-70B, 3.3-70B | Standard GQA + dense |
| Gemma | 4 E2B, E4B, 26B-A4B, 31B | Mixed + MoE |
| Cohere | Command R v01, R+, R7B, A 03-2025, A+ 05-2026 | Mixed 或 Standard |
| MiMo | V2.5, V2.5-Pro | Mixed + MoE |
| MiniMax | M2, M2.1, M2.5, M2.7, M3 | Standard GQA 或 MSA |

**子步骤**：

5.1 从 HuggingFace config 逐一采集 weight_fields
5.2 逐一验证 calcWeight 返回值
5.3 特别关注小模型：部分 Qwen3.5 小模型可能没有 MoE（纯 dense FFN）
5.4 Kimi K2.5/K2.6 确认是否为纯 dense（无 MoE）

**完成标准**：
- 所有模型的 totalParams 与公开数字偏差 <1%
- 无 console 报错

---

### 风险 & 缓解

| 风险 | 缓解 |
|---|---|
| 模型 config 缺少必要字段（如 `intermediate_size`） | 从模型文档或 safetensors 推算 |
| MoE 模型的前 N 层 dense 数量不明确 | 从 config 的 `first_k_dense_replace` 或 `moe_layer_freq` 确定 |
| 混合精度模型（部分权重 FP8，部分 BF16） | V1 统一精度计算，混合精度留给 Deploy tab |
| Vision 模型的 text/vision 参数混在一起 | V1 只算 text model，vision 部分预留 `vision_weight_fields` |
| 参数量验证偏差 >1% | 逐矩阵对比，找到偏差来源，修正公式 |

### 时间估算

| Phase | 预估工作量 |
|---|---|
| Phase 1 | 核心：5 个架构分支 + 5 个模型验证 |
| Phase 2 | 页面骨架 + 交互（大量复用 calculator.js） |
| Phase 3 | 可视化（复用 ibar/formula 渲染） |
| Phase 4 | 导航变更（机械操作） |
| Phase 5 | 批量补全（机械操作 + 逐个验证） |
