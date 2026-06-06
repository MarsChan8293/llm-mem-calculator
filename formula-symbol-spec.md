# Formula Symbol Spec — Pill 变量名短符号化

## 1. 动机

当前公式区的 pill 使用长文本变量名（如 `precision_bytes`、`kv_lora_rank`），导致：

- 单行公式在窄屏频繁换行，**公式结构被破坏**
- `precision_bytes` 一个 pill 占 ~120px，整行 GQA 公式超过 700px
- 用户一眼看不出公式的数学结构，信息密度低

改为短数学符号后：

- GQA 公式：`2 × L × h_kv × d_h × T × b`（~300px，一行放得下）
- 符号与 MLA / DeepSeek 论文记法一致，目标用户无需额外学习
- hover tooltip 保留，符号负责紧凑，tooltip 负责解释

**不引入 KaTeX/MathJax**，纯文本 + CSS `<sub>` 渲染，零依赖。

## 2. 符号映射表

### 2.1 通用变量（跨公式共用）

| 原始字段名 | 符号 | Tooltip 第一行 | 符号说明 |
|---|---|---|---|
| `layers` / `total_layers` / `num_hidden_layers` | `L` | `layers = 61` | 层数，所有公式通用 |
| `tokens` | `T` | `tokens = 1,024` | 序列长度，用户输入 |
| `sequences` | `S` | `sequences = 1` | 并发序列数，用户输入 |
| `precision_bytes` | `b` | `precision = 1 B` | KV 精度字节数，用户选择 |
| `indexer_precision_bytes` | `b_idx` | `indexer precision = 0.5 B` | Indexer 精度字节数 |

### 2.2 GQA 公式变量

| 原始字段名 | 符号 | 说明 |
|---|---|---|
| `kv_heads` | `h_kv` | KV head 数 |
| `head_dim` | `d_h` | Head 维度 |

### 2.3 MLA / DSA-MLA 公式变量

| 原始字段名 | 符号 | 说明 |
|---|---|---|
| `kv_lora_rank` | `d_c` | KV 压缩维度（MLA 论文记法） |
| `qk_rope_head_dim` | `d_r` | RoPE 维度（MLA 论文记法） |
| `index_head_dim` | `d_idx` | Indexer head 维度 |

### 2.4 DeepSeek V4 Hybrid 公式变量

| 原始字段名 | 符号 | 说明 |
|---|---|---|
| `sliding_window` | `W` | 滑动窗口大小 |
| `compress_ratio` | `r` | 压缩比 |
| `total_layers` | `L` | 与通用 L 相同 |
| `head_dim` | `d_h` | 与 GQA 相同 |
| `index_head_dim` | `d_idx` | 与 DSA-MLA 相同 |
| `ratio4_layers` | `L_4` | ratio=4 的层数 |
| `ratio128_layers` | `L_128` | ratio=128 的层数 |
| `ratio0_layers` | `L_0` | ratio=0 的层数 |

### 2.5 Mixed Full + Sliding GQA 公式变量

| 原始字段名 | 符号 | 说明 |
|---|---|---|
| `full_layers` | `L_f` | Full attention 层数 |
| `sliding_layers` | `L_s` | Sliding attention 层数 |
| `full_kv_heads` / `global_kv_heads` | `h_f` | Full attention KV heads |
| `sliding_kv_heads` | `h_s` | Sliding attention KV heads |
| `full_head_dim` / `global_head_dim` | `d_f` | Full attention head dim |
| `sliding_head_dim` | `d_s` | Sliding attention head dim |
| `full_v_head_dim` / `global_v_head_dim` | `d_vf` | Full attention V head dim |
| `sliding_v_head_dim` | `d_vs` | Sliding attention V head dim |
| `sliding_window` | `W` | 滑动窗口大小 |

### 2.6 Qwen Linear + Full Hybrid 公式变量

| 原始字段名 | 符号 | 说明 |
|---|---|---|
| `full_attention_layers` | `L_f` | Full attention 层数 |
| `linear_attention_layers` | `L_l` | Linear attention 层数 |
| `kv_heads` | `h_kv` | Full attention KV heads |
| `head_dim` | `d_h` | Full attention head dim |
| `lin_key_heads` / `linear_num_key_heads` | `h_kl` | Linear key heads |
| `lin_val_heads` / `linear_num_value_heads` | `h_vl` | Linear value heads |
| `lin_key_dim` / `linear_key_head_dim` | `d_kl` | Linear key head dim |
| `lin_val_dim` / `linear_value_head_dim` | `d_vl` | Linear value head dim |
| `conv_kernel_dim` / `linear_conv_kernel_dim` | `k_c` | Conv kernel 维度 |

### 2.7 结果变量（pill-result 类）

| 原始名称 | 符号 | 说明 |
|---|---|---|
| `kv_bytes` | `KV` | KV cache 字节数 |
| `total_bytes` | `Total` | 总字节数 |
| `indexer_bytes` | `Idx` | Indexer 字节数 |
| `sliding_kv_bytes` | `KV_sw` | 滑动窗口 KV 字节数 |
| `compressed_kv_bytes` | `KV_cmp` | 压缩 KV 字节数 |
| `full_kv_bytes` | `KV_f` | Full attention KV 字节数 |
| `sliding_kv_bytes` (mixed) | `KV_s` | Sliding KV 字节数 |
| `linear_conv_state_bytes` | `S_conv` | Conv state 字节数 |
| `linear_recurrent_state_bytes` | `S_rec` | Recurrent state 字节数 |

## 3. 各公式渲染效果

### 3.1 standard_gqa

**现在**：
```
kv_bytes = 2 × layers × kv_heads × head_dim × tokens × precision_bytes
total_bytes = sequences × kv_bytes
```

**改为**：
```
KV = 2 × L × h_kv × d_h × T × b
Total = S × KV
```

Tooltip 示例（hover `L`）：`layers = 61`
Tooltip 示例（hover `KV`）：`7.68 MiB\nMain DeepSeek V3 KV cache before applying precision.`

### 3.2 mla

**现在**：
```
kv_bytes = layers × (kv_lora_rank + qk_rope_head_dim) × tokens × precision_bytes
total_bytes = sequences × kv_bytes
```

**改为**：
```
KV = L × (d_c + d_r) × T × b
Total = S × KV
```

### 3.3 dsa_mla

**现在**：
```
kv_bytes       = layers × (kv_lora_rank + qk_rope_head_dim) × tokens × precision_bytes
indexer_bytes  = layers × index_head_dim × tokens × indexer_precision_bytes
total_bytes    = sequences × (kv_bytes + indexer_bytes)
```

**改为**：
```
KV   = L × (d_c + d_r) × T × b
Idx  = L × d_idx × T × b_idx
Total = S × (KV + Idx)
```

### 3.4 deepseek_v4_hybrid

**现在**：
```
sliding_kv_bytes   = total_layers × sliding_window × head_dim × precision_bytes
compressed_kv_bytes = Σ over ratio>0 layers: floor(tokens / compress_ratio) × head_dim × precision_bytes
kv_bytes           = sliding_kv_bytes + compressed_kv_bytes
indexer_bytes      = ratio4_layers × floor(tokens / 4) × index_head_dim × indexer_precision_bytes
total_bytes        = sequences × (kv_bytes + indexer_bytes)
```

**改为**：
```
KV_sw  = L × W × d_h × b
KV_cmp = Σ_{r>0} ⌊T / r⌋ × d_h × b
KV     = KV_sw + KV_cmp
Idx    = L_4 × ⌊T / 4⌋ × d_idx × b_idx
Total  = S × (KV + Idx)
```

> 注意：`Σ_{r>0}` 和 `⌊T / r⌋` 中的下标和向下取整用纯文本近似。
> - `Σ_{r>0}` → 文本写 `Σ(r>0)`，不依赖 LaTeX
> - `⌊T / r⌋` → 文本写 `⌊T/r⌋`，使用 Unicode 字符 U+230A / U+230B

### 3.5 mixed_full_sliding_gqa

**现在**：
```
full_kv_bytes    = full_layers × full_kv_heads × (full_head_dim + full_v_head_dim) × tokens × precision_bytes
sliding_kv_bytes = sliding_layers × sliding_kv_heads × (sliding_head_dim + sliding_v_head_dim) × min(tokens, sliding_window) × precision_bytes
kv_bytes         = full_kv_bytes + sliding_kv_bytes
total_bytes      = sequences × kv_bytes
```

**改为**：
```
KV_f = L_f × h_f × (d_f + d_vf) × T × b
KV_s = L_s × h_s × (d_s + d_vs) × min(T, W) × b
KV   = KV_f + KV_s
Total = S × KV
```

### 3.6 qwen_linear_full_hybrid

**现在**：
```
full_kv_bytes               = 2 × full_attention_layers × kv_heads × head_dim × tokens × precision_bytes
linear_conv_state_bytes     = sequences × linear_layers × conv_kernel_dim × (2 × lin_key_heads × lin_key_dim + lin_val_heads × lin_val_dim) × 2
linear_recurrent_state_bytes = sequences × linear_layers × lin_val_heads × lin_key_dim × lin_val_dim × 4
total_bytes                 = full_kv_bytes + linear_conv_state_bytes + linear_recurrent_state_bytes
```

**改为**：
```
KV_f   = 2 × L_f × h_kv × d_h × T × b
S_conv = S × L_l × k_c × (2 × h_kl × d_kl + h_vl × d_vl) × 2
S_rec  = S × L_l × h_vl × d_kl × d_vl × 4
Total  = KV_f + S_conv + S_rec
```

## 4. Tooltip 格式规范

### 4.1 参数 pill (pill-param)

Tooltip 格式：`{全名字段} = {数值}`

示例：
- hover `L` → `layers = 61`
- hover `h_kv` → `kv_heads = 128`
- hover `d_c` → `kv_lora_rank = 512`
- hover `d_r` → `qk_rope_head_dim = 64`
- hover `W` → `sliding_window = 128`

### 4.2 输入 pill (pill-input)

Tooltip 格式：`{描述} = {数值}`

示例：
- hover `T` → `tokens = 1,024`
- hover `S` → `sequences = 1`
- hover `b` → `precision = 1 B`
- hover `b_idx` → `indexer precision = 0.5 B`

### 4.3 结果 pill (pill-result)

Tooltip 格式：`{格式化字节数}\n{公式描述}`

示例：
- hover `KV` → `7.68 MiB\nMain DeepSeek V3 KV cache before applying precision.`
- hover `Total` → `7.68 MiB\nCombined DeepSeek V3 cache payload for all concurrent sequences.`
- hover `Idx` → `480 KiB\nIndexer cache for DSA sparse attention lookup.`

### 4.4 特殊符号的 Tooltip

- hover `Σ(r>0)` → `Sum over layers where compress_ratio > 0`
- hover `⌊T/r⌋` → `Floor of tokens divided by compress_ratio`

## 5. 实现方案

### 5.1 改动范围

只需修改 `js/calc.js` 中每个公式分支的 `formulas` 数组构建逻辑：

- `expr` 字符串中的变量名替换为短符号
- `values` 对象的 key 替换为短符号（与 expr 保持一致）
- `name` 字段替换为短符号（结果变量）

**不需要改的**：
- HTML 结构不变
- CSS 不变（pill 样式不变）
- `calculator.js` 中的替换逻辑不变（它按 `values` 的 key 做 regex 替换，key 换了符号自然会匹配）
- Breakdown 区域不变（仍然用自然语言）
- Compare 页面不变

### 5.2 calculator.js 中的 inputNames 判断

当前代码：
```js
var inputNames = { tokens: 1, sequences: 1, precision_bytes: 1, indexer_precision_bytes: 1 };
```

需改为：
```js
var inputNames = { T: 1, S: 1, b: 1, b_idx: 1 };
```

### 5.3 calculator.js 中的 `_bytes` 后缀判断

当前代码用 `match.indexOf('_bytes') !== -1` 判断是否为结果 pill，决定 tooltip 格式。

改为：检查 pill 的 CSS class 即可（`pill-result` 类已经在 DOM 上），或者维护一个结果变量名集合：
```js
var resultNames = { KV: 1, Total: 1, Idx: 1, KV_sw: 1, KV_cmp: 1, KV_f: 1, KV_s: 1, S_conv: 1, S_rec: 1 };
```

### 5.4 下标渲染

短符号中的下标（如 `h_kv`、`d_c`、`L_f`）用 HTML `<sub>` 标签渲染，而非纯文本下划线：

- pill 内容：`h<sub>kv</sub>`、`d<sub>c</sub>`、`L<sub>f</sub>`
- `<sub>` 标签在 pill 内不影响 regex 替换（因为替换发生在 `expr` 字符串上，`<sub>` 是在最终 HTML 输出时注入的）

实现方式：在 `calculator.js` 的 pill 生成逻辑中，对 pill 文本做一次后处理，将 `_` 分隔的下标转为 `<sub>`：

```js
function formatSymbol(text) {
  // h_kv → h<sub>kv</sub>, d_c → d<sub>c</sub>, L_f → L<sub>f</sub>
  return text.replace(/^([A-Za-z]+)_(.+)$/, '$1<sub>$2</sub>');
}
```

此函数仅作用于 pill 内的显示文本，不影响 `data-tooltip` 中的键名（tooltip 仍用全名+数值）。

### 5.5 特殊符号

- `×` 保留（已使用 Unicode `\u00d7`）
- `Σ(r>0)` → 纯文本 `Σ(r\u003e0)`
- `⌊T/r⌋` → 纯文本 `⌊T/r⌋`（U+230A / U+230B）
- `⌊T/4⌋` → 纯文本 `⌊T/4⌋`

这些特殊符号不经过 regex 替换（不在 `values` 的 key 中），直接写在 `expr` 字符串里。

### 5.6 Tooltip 中全名字段映射

`calc.js` 中需维护一个符号→全名的映射表，供 tooltip 使用：

```js
var SYMBOL_NAMES = {
  L: 'layers', T: 'tokens', S: 'sequences', b: 'precision', b_idx: 'indexer precision',
  h_kv: 'kv_heads', d_h: 'head_dim',
  d_c: 'kv_lora_rank', d_r: 'qk_rope_head_dim', d_idx: 'index_head_dim',
  W: 'sliding_window', r: 'compress_ratio',
  L_4: 'ratio4_layers', L_128: 'ratio128_layers', L_0: 'ratio0_layers',
  L_f: 'full_attention_layers', L_s: 'sliding_attention_layers', L_l: 'linear_attention_layers',
  h_f: 'full_kv_heads', h_s: 'sliding_kv_heads',
  d_f: 'full_head_dim', d_s: 'sliding_head_dim',
  d_vf: 'full_v_head_dim', d_vs: 'sliding_v_head_dim',
  h_kl: 'linear_key_heads', h_vl: 'linear_value_heads',
  d_kl: 'linear_key_head_dim', d_vl: 'linear_value_head_dim',
  k_c: 'conv_kernel_dim'
};
```

Tooltip 生成逻辑改为：`SYMBOL_NAMES[symbol] + ' = ' + formattedValue`

## 6. 不做的事

| 项目 | 原因 |
|---|---|
| 引入 KaTeX / MathJax | 杀鸡用牛刀，短符号纯文本+`<sub>` 已够用 |
| 改 Breakdown 区域 | Breakdown 用自然语言更适合（"Layers"、"KV heads"） |
| 改 Compare 页面 | Compare 不显示公式，不受影响 |
| 改 `data.js` 中的字段名 | 字段名是 HuggingFace config 的原始名，保持不变 |
| 给所有下标加样式化 | `h_kv` 的 `kv` 用 `<sub>`，但 `b_idx` 的 `idx` 也用 `<sub>` — 统一规则，全部下标化 |

## 7. 验收标准

1. 所有公式行的 pill 文本使用短符号
2. Hover 任何参数 pill 能看到 `{全名} = {数值}`
3. Hover 任何结果 pill 能看到 `{格式化字节数}\n{公式描述}`
4. 单行 GQA 公式在 768px 宽度下不换行
5. 下标（`kv`、`c`、`r`、`f`、`s`、`idx` 等）用 `<sub>` 渲染
6. 无新增 JS/CSS 依赖
7. Breakdown 区域不受影响
