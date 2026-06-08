# Deploy Tab — Feature Spec

## 1. Overview

Deploy Tab 是 Weights Tab 的进阶版，将 **KV Cache + Weights + 并行策略** 统一计算，
回答用户最核心的问题：**每张 GPU 需要多少显存？**

**核心原则**：
- 从总量到单卡 —— 引入 TP/PP/EP/DP，计算 per-GPU 显存
- 吸收 KV Cache 和 Weights 两个 Tab 的计算结果，不重复实现
- Prefill / Decode 分离部署是可选的高级功能
- 纯静态页面，无构建工具

**Tab 结构**：

```
● LLM Memory Calculator    [KV Cache]  [Weights]  [Deploy]  [Compare]     🌙
```

---

## 2. 核心问题

Deploy Tab 回答以下问题：

| 问题 | 答案来源 |
|---|---|
| 模型总权重多少？ | calc_weight.js（已有） |
| 模型 KV Cache 多少？ | calc.js（已有） |
| TP/PP/EP/DP 下每卡权重多少？ | **新增：权重切分逻辑** |
| TP/PP 下每卡 KV Cache 多少？ | **新增：KV 切分逻辑** |
| MLA Absorption 是否开启？ | **新增：影响权重和 KV 分布** |
| Prefill 和 Decode 用不同并行策略？ | **新增：分离计算** |
| 每卡总显存占用？ | weight_per_gpu + kv_per_gpu |
| 需要多少张什么型号的卡？ | per_gpu_bytes vs GPU VRAM |

---

## 3. 并行策略模型

### 3.1 基本规则

| 策略 | 切分对象 | 切分方式 | 对 per-GPU 的影响 |
|---|---|---|---|
| **TP** | Attention 权重 + Dense/Shared Expert FFN 权重 + KV Cache | 按头/列/行切分 | weight ÷ tp, kv ÷ tp |
| **PP** | 层 | 按 stage 分配层 | 只计算本 stage 的层 |
| **EP** | Routed Expert FFN 权重 | 按 expert ID 分配 | (n_routed / ep) × per_expert |
| **DP** | 无切分（全复制） | 每卡完整副本 | per-gpu 不变，总卡数 × dp |

**每卡显存公式**：

```
weight_per_gpu = 
  (layers_in_this_pp_stage) × (
    attn_weight_per_layer / tp          // Attention 按 TP 切
    + shared_expert_weight / tp         // Shared Expert 按 TP 切
    + (n_routed / ep) × per_expert_weight  // Routed Expert 按 EP 切
    + dense_ffn_weight / tp             // Dense FFN 按 TP 切
  )
  + embed_weight / tp                   // Embedding 按 TP 切（简化假设）

kv_per_gpu =
  layers_in_this_pp_stage × (
    kv_per_layer / tp                   // KV Cache 按 TP 切
    + idx_per_layer / tp_idx            // Indexer 可能按不同 TP 切
  )

mem_per_gpu = weight_per_gpu + kv_per_gpu
```

### 3.2 TP 切分 Attention 权重的细节

**Standard GQA**：W_q, W_k, W_v 按 KV head 切，W_o 按行切。简化为 `attn_weight / tp`。

**MLA**：W_kv_b 按 Q head 切（`n_q / tp`），W_q 同理，W_o 按行切。
但 **W_kv_a（down projection）** 是 `h × d_c`，不能按 head 切，需要特殊处理：

| 矩阵 | Shape | TP 切分方式 |
|---|---|---|
| W_q_a | h × q_lora_rank | 列并行：每卡 h × (q_lora_rank / tp)，需 all-reduce |
| W_q_b | q_lora_rank × (n_q × qk) | 按 Q head：每卡 q_lora_rank × (n_q/tp × qk) |
| W_kv_a | h × (d_c + d_r) | 列并行：每卡 h × ((d_c + d_r) / tp)，需 all-reduce |
| W_kv_b | d_c × (n_q × (qk_nope + d_v)) | 按 Q head：每卡 d_c × (n_q/tp × (qk_nope + d_v)) |
| W_o | (n_q × d_v) × h | 行并行：每卡 (n_q/tp × d_v) × h |

**简化处理**：所有 attention 权重统一 `/ tp`。误差来自 W_kv_a 等列并行矩阵
在 tp 不能整除时的余量，以及列并行需要额外通信缓冲。
V1 用 `÷ tp` 简化，在 note 中说明这是上界估计。

### 3.3 EP 切分 MoE Expert 的细节

```
expert_weight_per_gpu = ceil(n_routed / ep) × per_expert_weight
```

注意用 `ceil` 而非 `floor`——不能整除时，部分 GPU 多承载一个 expert。
V1 暂时用 `ceil`，未来可支持自定义 expert 分配。

**Shared Expert 不受 EP 影响**，按 TP 切分。

### 3.4 PP 切分层的细节

PP 切分按 stage 分配层。不同 stage 的权重不同（dense vs MoE 层）。

**层分配策略**：

| 策略 | 说明 | 优缺点 |
|---|---|---|
| 均匀分配 | 每个stage分配 `L / pp` 层 | 简单，但可能不均 |
| 按权重平衡 | 尽量让每个stage的总权重相等 | 更优，但计算复杂 |
| 自定义 | 用户指定每个 stage 的层范围 | 最灵活，但交互复杂 |

V1 实现 **均匀分配**。每个 stage 包含连续的 `ceil(L/pp)` 层。

```
stage_i_layers = layers[i * ceil(L/pp) ... min((i+1) * ceil(L/pp) - 1, L-1)]
```

对于有 `first_k_dense_replace` 的模型（如 DeepSeek V3：前3层 dense + 后58层 MoE），
PP=2 时 stage 0 会包含前3层 dense + 部分 MoE 层，stage 1 只有 MoE 层。
均匀分配自然处理了这种情况，只是 stage 间不均衡——这正好通过 ibar 可视化暴露出来。

### 3.5 KV Cache 在并行下的切分

**Standard GQA**：KV cache 按 TP 切，`kv_per_layer / tp`。

**MLA**：KV cache 存储 latent (d_c + d_r)，按 TP 切分时每卡存 `n_q/tp` 份 latent。
简化为 `kv_per_layer / tp`。

**DeepSeek V4 Hybrid**：这更复杂——
- Sliding window KV：所有层参与，`sliding_kv / tp`
- Compressed KV：ratio>0 的层，`compressed_kv / tp`
- Indexer：**通常 TP=1 运行**（indexer 不按 TP 切），所以 `idx_bytes` **不除以 tp**

**这是一个关键的精确度问题**。Indexer 在实际部署中通常 TP=1，
意味着 indexer 的 KV cache 是全量复制到每张卡，而非按 TP 切分。

V1 处理方式：
- 默认 `idx_per_layer / tp`（与 KV cache 统一）
- 增加 "Indexer TP" 独立控件，默认 = TP，用户可手动设为 1
- 对于 deepseek_v4_hybrid / dsa_mla / msa_gqa 模型，显示提示：
  "Indexer 通常以 TP=1 运行，请根据实际部署调整"

### 3.6 MLA Absorption

**Absorption 不改变存储的权重**（weights tab 不受影响），
但它改变了 **推理时实际使用的矩阵**，从而影响 per-GPU 的激活内存和计算量。

在 Deploy Tab 中，absorption 的作用是：
1. **减少 decode 阶段的 KV cache 访存量**（W_kvb_v 与 W_o 融合，减少一次矩阵乘）
2. **不改变权重存储量**（融合矩阵预计算后仍然存着）

所以 V1 的 Deploy Tab，absorption **不影响 weight_per_gpu 的计算**。
它影响的是计算效率（不在本工具范围）。

如果未来要计算 activation memory，absorption 才会改变 per-layer 的中间激活大小。
V1 不涉及，在 spec 中预留字段。

---

## 4. Prefill vs Decode 分离部署

### 4.1 背景

大规模 MoE 模型（DeepSeek V3/V4, Qwen3.5）的实际部署中，
prefill 实例和 decode 实例使用不同的 GPU 集群和不同的并行策略：

| | Prefill | Decode |
|---|---|---|
| 计算特征 | 计算密集 | 访存密集 |
| TP | 高（如 TP=8） | 低（如 TP=1~2） |
| EP | 可与 TP 相同 | 独立（如 EP=8） |
| Batch size | 小（1~4） | 大（128~1024） |
| MLA Absorption | 不用 | 可能使用 |

**结果**：同一模型的 prefill GPU 和 decode GPU 的显存占用完全不同。

### 4.2 UI 设计

采用 **Serving Mode** 开关：

```
── Serving Mode ──────────────────
○ Unified (单集群)
● Disaggregated (分离部署)

  ── Prefill ──
  TP  [8]   EP  [8]

  ── Decode ──
  TP  [2]   EP  [8]
  ☑ MLA Absorption
```

**Unified 模式**：一组 GPU 同时处理 prefill 和 decode，取 max(prefill_mem, decode_mem)。

**Disaggregated 模式**：两组 GPU 分别计算，结果并排展示。

### 4.3 计算逻辑

```javascript
function calcDeployPerGPU(model, options) {
  // options = {
  //   wtPrecB, kvPrecB, idxPrecB,
  //   tokens, batch,
  //   tp, pp, ep, dp,
  //   idx_tp,  // indexer TP (默认 = tp)
  //   mode: 'unified' | 'disaggregated',
  //   prefill: { tp, ep },
  //   decode: { tp, ep, absorption },
  // }

  if (options.mode === 'unified') {
    return calcUnifiedDeploy(model, options);
  } else {
    return {
      prefill: calcUnifiedDeploy(model, { ...options, tp: options.prefill.tp, ep: options.prefill.ep }),
      decode:  calcUnifiedDeploy(model, { ...options, tp: options.decode.tp, ep: options.decode.ep }),
    };
  }
}

function calcUnifiedDeploy(model, options) {
  const { tp, pp, ep } = options;
  
  // 1. 权重切分
  const weightResult = calcWeight(model, options.wtPrecB);
  const weightPerGPU = calcWeightPerGPU(weightResult, tp, pp, ep);
  
  // 2. KV Cache 切分
  const kvResult = calcKvCache(model, options.tokens, options.kvPrecB, options.idxPrecB, {
    includeDraft: options.includeDraft,
    includeLinear: options.includeLinear,
    seqs: options.batch,
  });
  const kvPerGPU = calcKVPerGPU(kvResult, tp, options.idx_tp, pp, options.batch);
  
  // 3. 总计
  return {
    weightPerGPU,
    kvPerGPU,
    totalPerGPU: weightPerGPU + kvPerGPU,
    // breakdown details...
  };
}
```

---

## 5. 页面布局

### 5.1 完整布局

```
┌─── Form Card ────────────────────────┐  ┌─── Result Area ─────────────────────────┐
│                                       │  │                                          │
│  Select model                         │  │  Per-GPU Memory                          │
│  ┌─────────────────────────────────┐  │  │  38.2 GiB                               │
│  │ 🔍 Search models...             │  │  │  ████████████████████████████████████    │
│  │  ▼ DeepSeek                      │  │  │  ██ Attn ██ ShrExp ██ Expert ██ KV      │
│  │    V4 Pro                  ✓     │  │  │                                          │
│  └─────────────────────────────────┘  │  │  [GiB] [GB] [MiB]                        │
│  [● DeepSeek V4 Pro ×]               │  │                                          │
│                                       │  │  GPU Fit                                 │
│  ── Precision ──                      │  │  38.2 / 80.0 GiB (H100) ✅ 47.8%        │
│  Weight  [BF16][FP8][FP4]            │  │  ████████████░░░░░░░░░░░░░░░░             │
│  KV      [BF16][FP8][FP4]            │  │                                          │
│  Indexer [BF16][FP8][FP4]            │  │  ── Weight Breakdown ──                   │
│                                       │  │  Attn weights     4.2 GiB                │
│  ── Inference Config ──              │  │  Shared expert    2.8 GiB                │
│  Context length                      │  │  Experts (48/384) 28.5 GiB               │
│  [1,024] [1K][4K][128K][1M]         │  │  Embedding        1.7 GiB                │
│                                       │  │  ────────────────────                    │
│  Batch size                          │  │  Weight total     37.2 GiB               │
│  [1][4][8][32][128]                  │  │  KV Cache          1.0 GiB               │
│                                       │  │  ────────────────────                    │
│  ☑ Include draft KV                  │  │  Total            38.2 GiB               │
│  ☐ Include linear KV                 │  │                                          │
│                                       │  │  ── Layer Distribution ──                │
│  ── Parallelism ──                   │  │  (ibar per-PP-stage)                     │
│  TP  [8]                              │  │  Stage 0 (L0-L30):                       │
│  PP  [1]                              │  │  [Attn][ShrExp][Expert] × 30             │
│  EP  [8]                              │  │  Stage 1 (L31-L60):                      │
│  DP  [1]                              │  │  [Attn][ShrExp][Expert] × 30             │
│                                       │  │                                          │
│  Indexer TP [8]  ← 仅 MoE 模型显示    │  │  ── GPU Count Estimate ──                │
│                                       │  │  H100 80GiB:  8 cards (TP×PP×DP)         │
│  ── Serving Mode ──                  │  │  A100 80GiB:  8 cards                    │
│  ○ Unified                           │  │  A100 40GiB:  ⚠️ OOM                     │
│  ● Disaggregated                     │  │                                          │
│                                       │  │  ── Disaggregated View ──                │
│    Prefill TP  [8]  EP  [8]          │  │  ┌──────────────┐ ┌──────────────┐      │
│    Decode  TP  [2]  EP  [8]          │  │  │ Prefill GPU  │ │ Decode GPU   │      │
│    ☐ MLA Absorption                  │  │  │ 52.1 GiB     │ │ 28.7 GiB     │      │
│                                       │  │  │ ██████████░ │ │ ██████░░░░░  │      │
│  ── GPU Target ──                    │  │  │ H100 ✅ 65%  │ │ H100 ✅ 36%  │      │
│  [H100 80GiB ▼]                      │  │  └──────────────┘ └──────────────┘      │
│                                       │  │                                          │
│                                       │  │  Note                                    │
│                                       │  │  Source                                  │
└───────────────────────────────────────┘  └──────────────────────────────────────────┘
```

### 5.2 Form Card 控件详解

| 控件 | 类型 | 默认值 | 条件显示 | 说明 |
|---|---|---|---|---|
| Model picker | 搜索式 Picker | deepseek-v4-pro | 始终 | 复用 |
| Weight precision | Segmented | BF16 | 始终 | |
| KV precision | Segmented | FP8 | 始终 | |
| Indexer precision | Segmented | FP8 | 有 indexer 的模型 | |
| Context length | 输入 + 预设 | 1,024 | 始终 | 复用 calculator.js |
| Batch size | 按钮 + 自定义 | 1 | 始终 | |
| Include draft KV | Toggle | off | 有 draft 的模型 | |
| Include linear KV | Toggle | off | linear 模型 | |
| **TP** | 数字输入 | 8 | 始终 | **新增** |
| **PP** | 数字输入 | 1 | 始终 | **新增** |
| **EP** | 数字输入 | 8 | 有 MoE 的模型 | **新增** |
| **DP** | 数字输入 | 1 | 始终 | **新增** |
| **Indexer TP** | 数字输入 | = TP | 有 indexer 的模型 | **新增** |
| **Serving mode** | Radio | Unified | 始终 | **新增** |
| Prefill TP/EP | 数字输入 | TP=8, EP=8 | Disaggregated | **新增** |
| Decode TP/EP | 数字输入 | TP=2, EP=8 | Disaggregated | **新增** |
| MLA Absorption | Toggle | off | MLA 模型 + Disaggregated | **新增** |
| GPU type | 下拉 | H100 80GiB | 始终 | **新增** |

### 5.3 条件显示规则

与 KV Cache tab 类似，多个控件根据模型类型条件显示：

```
if (model has n_routed_experts) → 显示 EP
if (model formula in [deepseek_v4_hybrid, dsa_mla, msa_gqa]) → 显示 Indexer TP
if (model formula in [mla, dsa_mla, deepseek_v4_hybrid]) → 显示 MLA Absorption
if (model has mtp/draft layers) → 显示 Include draft KV
if (model formula === qwen_linear_full_hybrid) → 显示 Include linear KV
```

### 5.4 GPU Type 下拉

```javascript
var GPU_OPTIONS = [
  { id: 'h100_80',  label: 'H100 80GiB',  vram: 80 * 1024**3 },
  { id: 'h100_40',  label: 'H100 40GiB',  vram: 40 * 1024**3 },
  { id: 'a100_80',  label: 'A100 80GiB',  vram: 80 * 1024**3 },
  { id: 'a100_40',  label: 'A100 40GiB',  vram: 40 * 1024**3 },
  { id: 'h200_140', label: 'H200 141GiB', vram: 141 * 1024**3 },
  { id: 'b200_180', label: 'B200 180GiB', vram: 180 * 1024**3 },
  { id: 'l40s_48',  label: 'L40S 48GiB',  vram: 48 * 1024**3 },
  { id: 'l4_24',    label: 'L4 24GiB',    vram: 24 * 1024**3 },
];
```

---

## 6. 计算逻辑 (js/calc_deploy.js)

新建 `js/calc_deploy.js`，依赖 `calc.js` 和 `calc_weight.js`。

### 6.1 函数签名

```javascript
/**
 * 计算部署配置下每 GPU 的显存占用
 *
 * @param {Object} model - MODEL_DATA 中的模型条目
 * @param {Object} options - 配置项
 * @param {number} options.wtPrecB - Weight 精度字节数
 * @param {number} options.kvPrecB - KV 精度字节数
 * @param {number} options.idxB - Indexer 精度字节数
 * @param {number} options.tokens - Context length
 * @param {number} options.batch - Batch size
 * @param {number} options.tp - Tensor parallelism
 * @param {number} options.pp - Pipeline parallelism
 * @param {number} options.ep - Expert parallelism
 * @param {number} options.dp - Data parallelism
 * @param {number} options.idxTp - Indexer TP (默认 = tp)
 * @param {boolean} options.includeDraft
 * @param {boolean} options.includeLinear
 * @param {string}  options.mode - 'unified' | 'disaggregated'
 * @param {Object}  options.prefill - { tp, ep } (仅 disaggregated)
 * @param {Object}  options.decode - { tp, ep, absorption } (仅 disaggregated)
 * @returns {Object} 部署结果
 */
function calcDeploy(model, options) { ... }
```

### 6.2 返回结构

```javascript
// Unified 模式
{
  mode: 'unified',
  weightPerGPU: number,         // bytes
  kvPerGPU: number,             // bytes
  totalPerGPU: number,          // bytes
  totalGPUs: number,            // tp × pp × dp
  
  // Weight breakdown per GPU
  weightBreakdown: {
    attnPerGPU: number,
    sharedExpertPerGPU: number,
    routedExpertPerGPU: number,
    denseFfnPerGPU: number,
    embedPerGPU: number,
  },
  
  // KV breakdown per GPU
  kvBreakdown: {
    kvPerGPU: number,
    idxPerGPU: number,
  },
  
  // Per-PP-stage details
  stages: [
    {
      stageIndex: 0,
      layerRange: 'L0-L30',
      layerCount: 30,
      denseLayers: 3,
      moeLayers: 27,
      weightPerGPU: number,
      kvPerGPU: number,
      totalPerGPU: number,
      ibar: [ { type, bytes } ],
    },
    ...
  ],
  
  // GPU fit
  gpuFit: {
    gpuId: 'h100_80',
    vram: number,
    usage: number,            // totalPerGPU / vram
    fits: boolean,
  },
  
  // Formula rows (per-GPU 版本)
  formulas: Array,
  formulaTitle: string,
  
  // Ibar data
  ibarSegments: [ { type, bytes, label } ],
  patterns: Array,
  legendTypes: Array,
}

// Disaggregated 模式
{
  mode: 'disaggregated',
  prefill: { ... unified structure ... },
  decode: { ... unified structure ... },
  totalGPUs: { prefill: n1, decode: n2 },
}
```

### 6.3 权重切分核心逻辑

```javascript
function calcWeightPerGPU(weightResult, tp, pp, ep, model) {
  const f = model.weight_fields;
  const nRouted = f.n_routed_experts || 0;
  const nShared = f.n_shared_experts || 0;
  const layers = model.fields.num_hidden_layers;
  
  // ── Per-layer weight ──
  const attnPerLayer = weightResult.attnParams / layers;   // 平均每层 attn 参数
  const denseFfnPerLayer = weightResult.ffnDenseParams / (/* dense layers count */);
  const sharedExpertPerLayer = weightResult.ffnSharedParams / (/* moe layers count */);
  const expertPerLayer = weightResult.ffnExpertParams / (/* moe layers count */);
  
  // ── Per-GPU per-layer ──
  const attnPerGPU = attnPerLayer / tp;
  const denseFfnPerGPU = denseFfnPerLayer / tp;
  const sharedExpertPerGPU = sharedExpertPerLayer / tp;
  const routedExpertPerGPU = (nRouted > 0)
    ? Math.ceil(nRouted / ep) * perExpertParams
    : 0;
  const embedPerGPU = weightResult.embedParams / tp;
  
  // ── PP stage assignment ──
  // 每个PP stage分配的层
  const layersPerStage = Math.ceil(layers / pp);
  
  // ── Total per-GPU ──
  // 注意：不同PP stage的层可能不同（dense vs MoE）
  // 需要逐stage计算
  const stages = [];
  for (let s = 0; s < pp; s++) {
    const startLayer = s * layersPerStage;
    const endLayer = Math.min((s + 1) * layersPerStage - 1, layers - 1);
    const stageLayerCount = endLayer - startLayer + 1;
    
    // 统计本stage的dense/moe层数
    let denseCount = 0, moeCount = 0;
    for (let l = startLayer; l <= endLayer; l++) {
      if (isMoeLayer(model, l)) moeCount++;
      else denseCount++;
    }
    
    const stageWeightPerGPU = 
      stageLayerCount * attnPerGPU +
      denseCount * denseFfnPerGPU +
      moeCount * (sharedExpertPerGPU + routedExpertPerGPU) +
      // Embedding 只在最前或最后stage，这里简化为均匀分配到第一个stage
      (s === 0 ? embedPerGPU : 0);
    
    stages.push({
      stageIndex: s,
      layerRange: 'L' + startLayer + '-L' + endLayer,
      layerCount: stageLayerCount,
      denseLayers: denseCount,
      moeLayers: moeCount,
      weightPerGPU: stageWeightPerGPU,
    });
  }
  
  // 取最大stage作为瓶颈
  const maxStage = stages.reduce((a, b) => a.weightPerGPU > b.weightPerGPU ? a : b);
  
  return {
    weightPerGPU: maxStage.weightPerGPU * wtPrecB,
    stages: stages,
    weightBreakdown: {
      attnPerGPU: attnPerGPU * wtPrecB,
      sharedExpertPerGPU: sharedExpertPerGPU * wtPrecB,
      routedExpertPerGPU: routedExpertPerGPU * wtPrecB,
      denseFfnPerGPU: denseFfnPerGPU * wtPrecB,
      embedPerGPU: embedPerGPU * wtPrecB,
    },
  };
}
```

### 6.4 KV Cache 切分核心逻辑

```javascript
function calcKVPerGPU(kvResult, tp, idxTp, pp, batch) {
  const totalLayers = /* from model */;
  const layersPerStage = Math.ceil(totalLayers / pp);
  
  // Per-GPU KV = (本stage的layers) × (kv_per_layer / tp) × batch
  // 简化：假设均匀分布
  const kvPerLayer = kvResult.kvBytes / totalLayers;
  const idxPerLayer = kvResult.idxBytes / totalLayers;
  
  const kvPerGPU = layersPerStage * (kvPerLayer / tp + idxPerLayer / idxTp) * batch;
  // 注意：kvResult.kvBytes 已经是单 seq 的量，需要 × batch
  
  return {
    kvPerGPU: kvPerGPU,
    kvBreakdown: {
      kvPerGPU: layersPerStage * kvPerLayer / tp * batch,
      idxPerGPU: layersPerStage * idxPerLayer / idxTp * batch,
    },
  };
}
```

**注意**：kvResult.kvBytes 是单序列的 KV cache 量。
`calcKvCache` 已经考虑了 `seqs` 参数，但那是总量。
在 Deploy 中需要按 TP/PP 切分，所以应该用单序列的值，再 × batch / TP。

实际上 `calcKvCache` 的 `seqs` 参数返回的是 `seqs × (kvBytes + idxBytes)`，
而 Deploy 需要的是 `(seqs / tp) × per_layer_kv × layers_per_stage`。
所以 Deploy 应该调用 `calcKvCache` 时 `seqs=1`，然后自行切分。

### 6.5 层分类辅助函数

```javascript
/**
 * 判断第 l 层是否为 MoE 层
 */
function isMoeLayer(model, layerIndex) {
  const wf = model.weight_fields;
  const f = model.fields;
  
  // 1. moe_layer_freq 数组（MiMo, MiniMax M2）
  if (wf.moe_layer_freq) {
    return wf.moe_layer_freq[layerIndex] === 1;
  }
  
  // 2. layer_types 数组（Qwen3.5, Gemma 4）
  // 需要从 model.fields 或 weight_fields 中获取
  if (f.layer_types) {
    const type = f.layer_types[layerIndex];
    // layer_types 通常是 "linear_attention", "full_attention", "sliding_attention"
    // 但不区分 dense/MoE，需要从其他字段推断
  }
  
  // 3. first_k_dense_replace（DeepSeek V3, GLM-5）
  if (wf.first_k_dense_replace) {
    return layerIndex >= wf.first_k_dense_replace;
  }
  
  // 4. 无 n_routed_experts → 全部 dense
  if (!wf.n_routed_experts || wf.n_routed_experts <= 1) {
    return false;
  }
  
  // 5. 有 n_routed_experts 但无 dense/MoE 分层 → 全部 MoE
  return true;
}
```

---

## 7. 可视化设计

### 7.1 Per-GPU Memory Ibar

最重要的可视化——展示每张卡的显存构成：

```
Per-GPU Memory (TP=8, PP=1, EP=8)

[████████████████████████████████████████] 38.2 GiB
 ██ Attn (4.2G)  ██ Shared (2.8G)  ██ Expert (28.5G)
 ██ KV Cache (1.0G)  ██ Embed (1.7G)
```

### 7.2 GPU Fit Bar

展示占用量与 GPU VRAM 的关系：

```
H100 80GiB
[████████████████░░░░░░░░░░░░░░░░░░░░░░░] 38.2 / 80.0 GiB  ✅ 47.8%

A100 40GiB  
[███████████████████████████████████░░░░] 38.2 / 40.0 GiB  ⚠️ 95.5%

A100 80GiB × 4 卡
[████████████████████░░░░░░░░░░░░░░░░░░░] 38.2 / 80.0 GiB  ✅ 47.8%
```

颜色规则：
- usage < 70% → 绿色 ✅
- 70% ≤ usage < 90% → 橙色 ⚠️
- usage ≥ 90% → 红色 ❌

### 7.3 PP Stage 分布图

当 PP > 1 时，展示每个 stage 的显存分布：

```
PP=2, TP=8, EP=8

Stage 0 (L0-L30: 3 dense + 27 MoE)
[████████████████████████░░░] 42.1 GiB
 ██ Attn ██ ShrExp ██ Expert ██ DenseFFN ██ Embed ██ KV

Stage 1 (L31-L60: 30 MoE)
[███████████████████░░░░░░░] 38.5 GiB
 ██ Attn ██ ShrExp ██ Expert ██ KV

⚠️ Stage 0 is the bottleneck (3 dense FFN layers + embedding)
```

### 7.4 Disaggregated 并排对比

```
Disaggregated Deployment

┌── Prefill (TP=8, EP=8) ──┐  ┌── Decode (TP=2, EP=8) ──┐
│  52.1 GiB / 80 GiB        │  │  28.7 GiB / 80 GiB       │
│  ████████████████░░░░░░░  │  │  █████████░░░░░░░░░░░░░  │
│  ✅ 65.1%                 │  │  ✅ 35.9%                 │
│                            │  │                            │
│  Attn      8.4 GiB        │  │  Attn      2.1 GiB        │
│  Shared    5.6 GiB        │  │  Shared    5.6 GiB        │
│  Expert   36.2 GiB        │  │  Expert   19.8 GiB        │
│  Embed     1.7 GiB        │  │  Embed     1.7 GiB        │
│  KV        0.2 GiB        │  │  KV        0.5 GiB        │
│                            │  │                            │
│  8 GPUs (TP×PP×DP)        │  │  16 GPUs (TP×PP×DP)       │
└────────────────────────────┘  └────────────────────────────┘
```

### 7.5 Ibar 颜色映射

Deploy tab 使用与 Weights tab 相同的颜色体系，额外增加 KV Cache 颜色：

| 类型 | 颜色 | CSS 变量 |
|---|---|---|
| Attention Weights | 蓝 | `--bar-attn` |
| Dense FFN | 琥珀 | `--bar-ffn-dense` |
| Shared Expert | 橙 | `--bar-ffn-shared` |
| Routed Expert | 红 | `--bar-ffn-expert` |
| Embedding | 紫 | `--bar-embed` |
| KV Cache | 青 | `--bar-kv` (新增) |
| Indexer KV | 紫红 | `--bar-indexer` (复用) |

```css
:root {
  --bar-kv: #0c8599;        /* 青 - KV Cache */
}
[data-theme="dark"] {
  --bar-kv: #15aabf;
}
```

---

## 8. 公式展示

Deploy tab 的公式展示 per-GPU 版本，比 KV Cache 和 Weights tab 更具体：

### 8.1 示例：DeepSeek V4 Pro, TP=8, EP=8, PP=1

```
── Per-GPU Weight ──

Attn/tp       = (W_q + W_kv + W_o) / 8                    = 4.2 GiB
Shared/tp     = 1 × 3 × h × I_s / 8                       = 2.8 GiB
Expert/ep     = ceil(384/8) × 3 × h × I_m                  = 28.5 GiB
Embed/tp      = V × h / 8                                   = 1.7 GiB
Weight_total  = Attn/tp + Shared/tp + Expert/ep + Embed/tp  = 37.2 GiB

── Per-GPU KV Cache ──

KV/tp         = L × (KV_sw + KV_cmp) / 8 × T × p × B      = 0.8 GiB
Idx/tp_idx    = L_4 × ⌊T/4⌋ × d_idx × p_idx × B / 8      = 0.2 GiB
KV_total      = KV/tp + Idx/tp_idx                           = 1.0 GiB

── Total ──

Mem/GPU       = Weight_total + KV_total                      = 38.2 GiB
```

### 8.2 公式变量

Deploy 新增的短符号：

| 符号 | 说明 |
|---|---|
| `tp` | Tensor parallelism |
| `pp` | Pipeline parallelism |
| `ep` | Expert parallelism |
| `dp` | Data parallelism |
| `tp_idx` | Indexer tensor parallelism |
| `B` | Batch size（注意：与 KV Cache 的 B = sequences 对齐） |
| `V_gpu` | GPU VRAM |

---

## 9. 数据模型变更

### 9.1 data.js 新增

```javascript
const MODEL_DATA = {
  // ... existing ...
  
  weight_precision_options: [...],  // Weights tab 已新增
  
  gpu_options: [
    { id: 'h100_80',  label: 'H100 80GiB',  vram: 85899345920 },
    { id: 'h200_140', label: 'H200 141GiB', vram: 151393783808 },
    { id: 'b200_180', label: 'B200 180GiB', vram: 193273528320 },
    { id: 'a100_80',  label: 'A100 80GiB',  vram: 85899345920 },
    { id: 'a100_40',  label: 'A100 40GiB',  vram: 42949672960 },
    { id: 'h100_40',  label: 'H100 40GiB',  vram: 42949672960 },
    { id: 'l40s_48',  label: 'L40S 48GiB',  vram: 51539607552 },
    { id: 'l4_24',    label: 'L4 24GiB',    vram: 25769803776 },
  ],
  
  // 每个 model 新增 deploy_defaults（可选，覆盖通用默认值）
  // 用于根据模型特征提供合理的默认并行配置
};
```

### 9.2 deploy_defaults 示例

```javascript
// DeepSeek V4 Pro
deploy_defaults: {
  tp: 8, pp: 1, ep: 8, dp: 1,
  idx_tp: 8,
  gpu: 'h100_80',
  mode: 'unified',
}

// Llama 3.1 8B
deploy_defaults: {
  tp: 1, pp: 1, ep: 1, dp: 1,
  idx_tp: 1,
  gpu: 'a100_40',
  mode: 'unified',
}
```

### 9.3 层分类数据

每个模型需要提供逐层分类信息，用于 PP 切分时准确计算。

三种方式（按优先级）：

1. **moe_layer_freq 数组**（已有于 MiMo, MiniMax M2 的 config）
2. **first_k_dense_replace**（已有于 DeepSeek V3, GLM-5 的 config）
3. **推导**：有 n_routed_experts → 全部 MoE；无 → 全部 dense

这些数据已经在 `weight_fields` 中，无需额外字段。

---

## 10. 文件结构

```
llm-mem-calculator/
├── index.html            # KV Cache
├── weights.html          # Weights
├── deploy.html           # Deploy (新增)
├── compare.html          # Compare
├── css/
│   └── main.css          # 新增 deploy ibar 颜色变量
├── js/
│   ├── data.js           # 新增 gpu_options, deploy_defaults
│   ├── calc.js           # KV 计算引擎（不改）
│   ├── calc_weight.js    # Weight 计算引擎（不改）
│   ├── calc_deploy.js    # Deploy 计算引擎（新增）
│   ├── calculator.js     # KV 页面（不改）
│   ├── weights.js        # Weights 页面（不改）
│   ├── deploy.js         # Deploy 页面（新增）
│   └── compare.js        # Compare 页面（不改）
```

---

## 11. 开发计划

### 总体思路：分 4 个阶段

```
Phase 1: 计算引擎（calc_deploy.js 核心逻辑）
   ↓
Phase 2: 页面骨架 + Unified 模式交互
   ↓
Phase 3: 可视化（ibar + GPU fit + PP stage）
   ↓
Phase 4: Disaggregated 模式
```

### Phase 1: 计算引擎

**目标**：`calc_deploy.js` 核心逻辑，Unified 模式可工作。

**子步骤**：

1.1 实现 `calcWeightPerGPU()` — 权重切分
1.2 实现 `calcKVPerGPU()` — KV 切分
1.3 实现 `isMoeLayer()` — 层分类辅助函数
1.4 实现 `calcDeploy()` Unified 模式 — 组合以上函数
1.5 用 DeepSeek V4 Pro (TP=8, EP=8) 验证
1.6 用 DeepSeek V3 (TP=8, EP=8) 验证
1.7 用 Llama 3.1 8B (TP=1) 验证
1.8 用 Qwen3-235B (TP=4, EP=4) 验证
1.9 PP>1 场景验证（DeepSeek V3, PP=2）

**验证方式**：
- 对比公开部署文档中的 per-GPU 显存数字
- 权重切分结果 × 精度 应与 Weights tab 总量 / 并行度 一致（误差 <5%）
- KV 切分结果应与 KV Cache tab 总量 / TP 一致

**完成标准**：
- 4 个典型模型的 per-GPU 计算结果合理
- PP>1 时 stage 间不均衡可正确展示

---

### Phase 2: 页面骨架 + Unified 模式

**目标**：deploy.html 可运行，Unified 模式数字正确。

**产出**：
- `deploy.html` — HTML 骨架
- `js/deploy.js` — 交互逻辑

**子步骤**：

2.1 创建 `deploy.html`，复制 weights.html 结构
2.2 新增 Form Card 控件：
    - KV precision + Indexer precision
    - Context length + Batch size
    - Draft/Linear toggles
    - TP/PP/EP/DP 数字输入
    - Indexer TP
    - Serving mode radio (Unified only)
    - GPU type dropdown
2.3 实现 `deploy.js`：
    - Model picker 交互
    - 所有控件的事件监听
    - `calculate()` 主函数
    - 条件显示逻辑（EP 仅 MoE, Indexer TP 仅 DSA 等）
    - GPU fit 判断
2.4 渲染 per-GPU 数字和 breakdown

**完成标准**：
- 选择模型 → 设置并行参数 → 显示正确的 per-GPU 显存
- GPU fit bar 正常显示
- 条件显示控件正确切换
- Dark mode 正常

---

### Phase 3: 可视化

**目标**：ibar + GPU fit bar + PP stage 分布完整展示。

**子步骤**：

3.1 CSS 新增 `--bar-kv` 颜色变量
3.2 实现 per-GPU memory ibar（复用现有 ibar 渲染）
3.3 实现 GPU fit bar（带阈值颜色的进度条）
3.4 实现 PP stage 分布图（PP>1 时显示）
3.5 实现 formula section（per-GPU 公式）
3.6 实现 GPU count estimate 展示
3.7 新增 DEPLOY_SYMBOL_NAMES 映射表

**完成标准**：
- 所有可视化正确渲染
- 亮色/暗色主题正确
- PP>1 时各 stage 正确展示
- GPU fit 颜色阈值正确（绿/橙/红）

---

### Phase 4: Disaggregated 模式

**目标**：支持 prefill/decode 分离部署的计算和展示。

**子步骤**：

4.1 新增 Disaggregated 控件（Prefill TP/EP, Decode TP/EP, Absorption toggle）
4.2 实现 `calcDeploy()` Disaggregated 分支
4.3 实现并排对比 UI（Prefill GPU vs Decode GPU）
4.4 实现 total GPU count 展示
4.5 Absorption 对 decode 侧的影响（V1: 不影响，标注 "coming soon"）

**完成标准**：
- 切换到 Disaggregated → 两组独立结果并排显示
- Prefill 和 Decode 参数独立调整
- Absorption toggle 显示但不影响计算（标注 "coming soon"）

---

## 12. 已知限制 & 未来工作

| 限制 | 说明 | 处理方式 |
|---|---|---|
| 权重切分简化为 `/tp` | MLA 的 W_kv_a 列并行、O projection 行并行等细节未精确建模 | Note 中说明是上界估计 |
| Activation memory | 推理时的中间激活内存未计算 | V1 不含，留给未来 |
| MLA Absorption 影响 | 不改变权重存储，但影响激活内存 | V1 不影响计算，标注 "coming soon" |
| Expert 不均匀分配 | `ceil(n/ep)` 是粗略估计 | V1 均匀，未来支持自定义 |
| PP 层分配策略 | V1 均匀分配，不考虑权重平衡 | 未来支持按权重平衡 |
| 通信开销 | TP/EP 的 all-reduce/all-to-all 通信缓冲区未计入 | V1 不含 |
| CUDA 内核碎片 | 实际显存有 5-10% 的碎片开销 | V1 不含，在 Note 中提醒用户预留 |
| 混合精度权重 | 部分模型 attention BF16 + expert FP8 | V1 统一精度 |
| Custom expert mapping | 不同 GPU 可能承载不同数量的 expert | V1 均匀 |
| vLLM/SGLang 特定优化 | 不同 serving 框架的显存策略不同 | V1 不区分框架 |

---

## 13. 不做的事

| 项目 | 原因 |
|---|---|
| 推理吞吐量/延迟计算 | 超出本工具范围 |
| 激活内存精确计算 | 需要模型计算图，过于复杂 |
| 通信开销估算 | 依赖具体硬件拓扑 |
| Serving 框架适配 | vLLM/SGLang/TF 各有优化策略 |
| 自动推荐并行配置 | 过于复杂，V1 让用户手动配置 |
| Multi-query / speculative decoding 额外 KV | V1 不含 |
| 量化策略推荐 | 比较不同量化方案的超出了 V1 范围 |
