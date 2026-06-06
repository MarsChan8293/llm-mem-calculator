# KV Cache Layer Bar Spec

## 1. Overview

在 Calculator 页面的 formula 区域内，为每个公式行右侧添加 inline proportional bar，可视化该行结果变量对应的 KV cache 组件构成。同时在公式行下方添加 "Layer patterns" 区域，展示 unique layer 类型及其重复次数。

核心价值：公式告诉用户"怎么算"，bar 告诉用户"长什么样"，两者左右对齐、颜色统一。

## 2. Color System

### 2.1 语义色板

6 种语义类型，跨架构统一。同色同义，用户看任何模型的 bar 都能直接读出"KV cache 由什么组成"。

| ID | 语义 | Hex | 含义 |
|---|---|---|---|
| `full` | Full KV | `#4263eb` | 存全部 T tokens 的完整 KV cache |
| `window` | Window KV | `#0c8599` | 只存 W tokens 的滑动窗口，到顶不增长 |
| `compressed` | Compressed KV | `#ae3ec9` | 低秩压缩 / 稀疏压缩后的 KV cache |
| `rope` | RoPE | `#d9480f` | 位置编码分量（MLA 的 d_r） |
| `indexer` | Indexer | `#e03131` | 稀疏注意力索引 cache |
| `fixed` | Fixed State | `#2b8a3e` | 与 T 无关的固定状态（linear conv / recurrent） |

### 2.2 子类型深浅区分

同一语义类型下有子类型时，用深浅两色区分：

| 语义 | 深色 | 浅色 | 用途 |
|---|---|---|---|
| Fixed | `#2b8a3e` (conv) | `#40c057` (recurrent) | Linear+Full 的 S_conv vs S_rec |
| Full | `#4263eb` (K) | `#5c7cfa` (V) | GQA / Full attention 的 K vs V 分拆 |

子类型深浅区分仅在 bar 内部使用，legend 只显示语义级别的 6 色。

### 2.3 暗色主题适配

使用 CSS 变量，暗色模式下亮度微调（与现有 pill 颜色系统一致）：

| 变量 | Light | Dark |
|---|---|---|
| `--bar-full` | `#4263eb` | `#5c7cfa` |
| `--bar-window` | `#0c8599` | `#15aabf` |
| `--bar-compressed` | `#ae3ec9` | `#c084fc` |
| `--bar-rope` | `#d9480f` | `#f59e0b` |
| `--bar-indexer` | `#e03131` | `#f03e3e` |
| `--bar-fixed` | `#2b8a3e` | `#40c057` |
| `--bar-fixed-alt` | `#40c057` | `#69db7c` |
| `--bar-full-alt` | `#5c7cfa` | `#818cf8` |

## 3. Formula Row Layout

### 3.1 结构

每个 formula-row 从原来的两段 (LHS + RHS) 扩展为四段：

```
┌──────────────────────────────────────────────────────────┐
│ [pill] =  [expression...]   [████████ ██]  3.44 MB       │
│  LHS       RHS              ibar          ibar-val       │
└──────────────────────────────────────────────────────────┘
```

- **LHS**：结果变量 pill（不变）
- **RHS**：公式表达式（不变）
- **ibar**：新增，proportional bar，宽度按比例反映各组件字节数
- **ibar-val**：新增，该行结果值的格式化字节数

### 3.2 CSS 变更

`.formula-row` 从 `display: flex; align-items: baseline;` 改为 `display: flex; align-items: center;`（bar 需要 vertical center 对齐）。

新增 CSS：

```css
.ibar {
  display: flex;
  height: 10px;
  gap: 1px;
  flex-shrink: 0;
  margin-left: 16px;
  min-width: 4px;
}
.ibar .seg {
  height: 100%;
  border-radius: 1.5px;
  min-width: 1px;
  transition: width 0.4s ease;
}
.ibar-val {
  font-size: 0.72rem;
  color: var(--text3);
  font-family: var(--mono);
  margin-left: 8px;
  white-space: nowrap;
  flex-shrink: 0;
}
```

### 3.3 Bar 宽度计算

所有 bar 的宽度基于**同一行内各组件的相对比例**：

1. 计算该行每个语义组件的字节数
2. 找到最大组件的字节数，映射为 `maxSegWidth = 120px`
3. 其余组件按比例缩放：`segWidth = (componentBytes / maxComponentBytes) * 120px`
4. 最小宽度 `1px`（即使极小也可见）

**跨行不统一 scale**——每行独立 scale。原因：KV_sw 和 Total 量级可能差 10 倍，统一 scale 会让 KV_sw 的 bar 几乎不可见。每行独立 scale 让每行的内部比例清晰可读。

### 3.4 各架构映射

#### Standard GQA

| 行 | bar 组件 |
|---|---|
| KV | `[full]` |
| Total | `[full]` |

#### MLA

| 行 | bar 组件 |
|---|---|
| KV | `[compressed] + [rope]` |
| Total | `[compressed] + [rope]` |

#### DSA+MLA

| 行 | bar 组件 |
|---|---|
| KV | `[compressed] + [rope]` |
| Idx | `[indexer]` |
| Total | `[compressed] + [rope] + [indexer]` |

#### DeepSeek V4 Hybrid

| 行 | bar 组件 |
|---|---|
| KV_sw | `[window]` |
| KV_cmp | `[compressed]` |
| KV | `[window] + [compressed]` |
| Idx | `[indexer]` |
| Total | `[window] + [compressed] + [indexer]` |

#### Mixed Full + Sliding

| 行 | bar 组件 |
|---|---|
| KV_f | `[full]` |
| KV_s | `[window]` |
| KV | `[full] + [window]` |
| Total | `[full] + [window]` |

#### Linear + Full Hybrid

| 行 | bar 组件 |
|---|---|
| KV_f | `[full]` |
| S_conv | `[fixed]` (深色 `#2b8a3e`) |
| S_rec | `[fixed]` (浅色 `#40c057`) |
| Total | `[full] + [fixed #2b8a3e] + [fixed #40c057]` |

### 3.5 ibar-val 格式化

使用现有的 `fmtBytes()` 函数，与 breakdown 区域一致。

对于 `Total` 行，值为 `seqs * (kvBytes + idxBytes)`；其余行为单序列值。

## 4. Layer Patterns

### 4.1 位置与结构

在公式行列表下方（最后一个 formula-row 之后），新增一个轻量区域：

```html
<div class="pattern-section">
  <div class="pattern-title">Layer patterns</div>
  <div class="pattern-row">
    <div class="pattern-bar">[segs...]</div>
    <span class="pattern-count">×21</span>
    <span class="pattern-label">r = 4</span>
  </div>
  <!-- more rows -->
</div>
```

### 4.2 CSS

```css
.pattern-section {
  margin-top: 10px;
  padding: 8px 12px;
  background: var(--bg);
  border-radius: var(--radius-md);
  border: 1px solid var(--surface2);
}
.pattern-title {
  font-size: 9px;
  color: var(--text3);
  font-family: var(--mono);
  letter-spacing: 0.5px;
  text-transform: uppercase;
  margin-bottom: 5px;
}
.pattern-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 2px;
  font-size: 10px;
  font-family: var(--mono);
  color: var(--text2);
}
.pattern-bar {
  display: flex;
  height: 7px;
  gap: 1px;
}
.pattern-bar .seg {
  height: 100%;
  border-radius: 1.5px;
  min-width: 1px;
}
.pattern-count {
  color: var(--text3);
  font-size: 9px;
}
.pattern-label {
  color: var(--text3);
  font-size: 9px;
  min-width: 40px;
}
```

### 4.3 各架构 pattern 行

只展示 **unique 层类型**，标注重复次数。颜色与 bar 语义色一致。

#### Standard GQA

| pattern | bar | count | label |
|---|---|---|---|
| 1 | `[full-K] + [full-V]` | ×L | all layers |

#### MLA

| pattern | bar | count | label |
|---|---|---|---|
| 1 | `[compressed] + [rope]` | ×L | all layers |

#### DSA+MLA

| pattern | bar | count | label |
|---|---|---|---|
| 1 | `[compressed] + [rope] + [indexer]` | ×L | all layers |

注：DSA+MLA 的所有层结构相同，只有 1 种 pattern。

#### DeepSeek V4 Hybrid

| pattern | bar | count | label |
|---|---|---|---|
| ratio=4 | `[window] + [compressed] + [indexer]` | ×(count) | r = 4 |
| ratio=128 | `[window] + [compressed]` | ×(count) | r = 128 |
| ratio=0 | `[window]` (用 `#c8cdd8` 灰色表示空/无) | ×(count) | r = 0 |

ratio=0 层的 pattern bar 使用灰色 `#c8cdd8`（light）/ `#3a3f52`（dark），因为这类层只有 sliding window，在 per-token 视角下贡献为 0（amortized）。灰色传达"存在但可忽略"。

#### Mixed Full + Sliding

| pattern | bar | count | label |
|---|---|---|---|
| full attention | `[full-K] + [full-V]` | ×L_f | full attn |
| sliding attention | `[window-K] + [window-V]` | ×L_s | sliding attn |

Full 和 Sliding 都拆分 K/V 显示，使用 full/window 语义色的深浅变体。

#### Linear + Full Hybrid

| pattern | bar | count | label |
|---|---|---|---|
| full attention | `[full-K] + [full-V]` | ×L_f | full attn |
| linear attention | `[fixed] + [fixed-alt]` | ×L_l | linear attn |

Linear pattern 的两个 seg 用 fixed 色的深浅区分 conv vs recurrent。

### 4.4 Pattern bar 宽度

Pattern bar 使用**固定参考 scale**，同一模型内所有 pattern 行共享同一 scale，以便直观对比不同 pattern 的宽度差异。

- 找到所有 pattern 中字节总量最大的那个，映射为 `maxPatternWidth = 120px`
- 其余按比例缩放
- 最小宽度 2px

## 5. Legend

### 5.1 位置

Pattern section 下方，紧跟 pattern 行。

### 5.2 显示规则

只显示当前模型实际使用的语义色，不显示未使用的。例如：
- GQA 只显示：Full
- V4 Hybrid 显示：Window, Compressed, Indexer
- Linear+Full 显示：Full, Fixed

### 5.3 CSS

```css
.layer-bar-legend {
  display: flex;
  gap: 10px;
  margin-top: 6px;
  flex-wrap: wrap;
}
.lbleg {
  display: flex;
  align-items: center;
  gap: 3px;
  font-size: 8px;
  color: var(--text3);
}
.lbleg .s {
  width: 8px;
  height: 5px;
  border-radius: 1px;
}
```

### 5.4 Legend 文案映射

| 语义 | Legend 文案 |
|---|---|
| `full` | Full KV |
| `window` | Window |
| `compressed` | Compressed |
| `rope` | RoPE |
| `indexer` | Indexer |
| `fixed` | Fixed state |

## 6. calc.js 变更

### 6.1 formulas 数组扩展

每个 formula 对象新增两个字段：

```js
{
  name: 'KV_sw',
  tip: '...',
  expr: '...',
  values: { ... },
  resultValue: 12345,
  // 新增 ↓
  bar: [
    { type: 'window', bytes: 12345 }
  ],
  ibarVal: '2.81 MB'
}
```

- `bar`: 数组，每项 `{ type: 语义ID, bytes: 字节数 }`，描述该行 bar 由哪些组件构成
- `ibarVal`: 格式化后的字节数字符串，直接渲染

### 6.2 patterns 数组扩展

`calcKvCache` 返回值新增 `patterns` 字段：

```js
return {
  kvBytes, idxBytes, perTokenBytes,
  breakdown, formulas, formulaTitle,
  // 新增 ↓
  patterns: [
    {
      segs: [{ type: 'window', width: 0.8 }, { type: 'compressed', width: 0.15 }, { type: 'indexer', width: 0.05 }],
      count: 21,
      label: 'r = 4',
      bytes: 12345  // 用于 scale 计算
    },
    // ...
  ],
  legendTypes: ['window', 'compressed', 'indexer']  // 当前模型使用的语义类型列表
};
```

- `segs[].width`: 该 seg 在 pattern bar 中的相对宽度比例（0~1），由调用方换算为像素
- `legendTypes`: 用于按需显示 legend

### 6.3 各架构的 patterns 定义

#### standard_gqa
```js
patterns: [{
  segs: [{ type: 'full', ratio: 0.5 }, { type: 'full-alt', ratio: 0.5 }],
  count: layers,
  label: 'all layers',
  bytes: kvBytes / layers
}]
legendTypes: ['full']
```

#### mla
```js
patterns: [{
  segs: [{ type: 'compressed', ratio: kvLoraRank / (kvLoraRank + qkRopeHd) }, { type: 'rope', ratio: qkRopeHd / (kvLoraRank + qkRopeHd) }],
  count: layers,
  label: 'all layers',
  bytes: kvBytes / layers
}]
legendTypes: ['compressed', 'rope']
```

#### dsa_mla
```js
patterns: [{
  segs: [
    { type: 'compressed', ratio: kvLoraRank / (kvLoraRank + qkRopeHd + idxHd) },
    { type: 'rope', ratio: qkRopeHd / (kvLoraRank + qkRopeHd + idxHd) },
    { type: 'indexer', ratio: idxHd / (kvLoraRank + qkRopeHd + idxHd) }
  ],
  count: layers,
  label: 'all layers',
  bytes: (kvBytes + idxBytes) / layers
}]
legendTypes: ['compressed', 'rope', 'indexer']
```

#### deepseek_v4_hybrid
```js
patterns: [
  {
    segs: [{ type: 'window', ratio: ... }, { type: 'compressed', ratio: ... }, { type: 'indexer', ratio: ... }],
    count: ratio4Layers,
    label: 'r = 4',
    bytes: windowB + compressB_r4 + idxB_r4
  },
  {
    segs: [{ type: 'window', ratio: ... }, { type: 'compressed', ratio: ... }],
    count: ratio128Layers,
    label: 'r = 128',
    bytes: windowB + compressB_r128
  },
  {
    segs: [{ type: 'window-empty', ratio: 1 }],
    count: ratio0Layers,
    label: 'r = 0',
    bytes: 0
  }
]
legendTypes: ['window', 'compressed', 'indexer']
```

注：ratio=0 的 `type` 使用 `window-empty`，渲染时使用灰色。

#### mixed_full_sliding_gqa
```js
patterns: [
  {
    segs: [{ type: 'full', ratio: globalHd / (globalHd + fullVHd) }, { type: 'full-alt', ratio: fullVHd / (globalHd + fullVHd) }],
    count: fullLayers,
    label: 'full attn',
    bytes: fullBytes / fullLayers
  },
  {
    segs: [{ type: 'window', ratio: slidingHd / (slidingHd + slidingVHd) }, { type: 'window-alt', ratio: slidingVHd / (slidingHd + slidingVHd) }],
    count: slidingLayers,
    label: 'sliding attn',
    bytes: slidingBytes / slidingLayers
  }
]
legendTypes: ['full', 'window']
```

#### qwen_linear_full_hybrid
```js
patterns: [
  {
    segs: [{ type: 'full', ratio: 0.5 }, { type: 'full-alt', ratio: 0.5 }],
    count: fullLayers,
    label: 'full attn',
    bytes: fullBytes / fullLayers
  },
  {
    segs: [{ type: 'fixed', ratio: linConvBytes / (linConvBytes + linRecurrentBytes) }, { type: 'fixed-alt', ratio: linRecurrentBytes / (linConvBytes + linRecurrentBytes) }],
    count: linearLayers,
    label: 'linear attn',
    bytes: linBytes / linearLayers
  }
]
legendTypes: ['full', 'fixed']
```

## 7. calculator.js 变更

### 7.1 formula 渲染

现有公式行渲染逻辑中，在每个 `.formula-row` 末尾追加 ibar 和 ibar-val：

```js
// 现有代码生成 formulaRow HTML 后...
var ibarHtml = '<div class="ibar">';
f.bar.forEach(function(seg) {
  var colorVar = getBarColor(seg.type);  // 映射到 CSS class
  var widthPx = Math.max(1, (seg.bytes / maxBytesInRow) * 120);
  ibarHtml += '<div class="seg ' + colorVar + '" style="width:' + widthPx + 'px"></div>';
});
ibarHtml += '</div>';
ibarHtml += '<div class="ibar-val">' + f.ibarVal + '</div>';
```

### 7.2 patterns 渲染

在 formulaSection 末尾追加 pattern section：

```js
if (result.patterns && result.patterns.length > 0) {
  var patternHtml = '<div class="pattern-section">';
  patternHtml += '<div class="pattern-title">Layer patterns</div>';
  result.patterns.forEach(function(p) {
    patternHtml += '<div class="pattern-row"><div class="pattern-bar">';
    p.segs.forEach(function(seg) {
      var widthPx = Math.max(1, seg.ratio * maxPatternWidth);
      patternHtml += '<div class="seg ' + getBarColor(seg.type) + '" style="width:' + widthPx + 'px"></div>';
    });
    patternHtml += '</div>';
    patternHtml += '<span class="pattern-count">×' + p.count + '</span>';
    patternHtml += '<span class="pattern-label">' + p.label + '</span>';
    patternHtml += '</div>';
  });
  // Legend
  if (result.legendTypes && result.legendTypes.length > 1) {
    patternHtml += '<div class="layer-bar-legend">';
    result.legendTypes.forEach(function(t) {
      patternHtml += '<div class="lbleg"><div class="s" style="background:' + getBarHex(t) + '"></div>' + getLegendLabel(t) + '</div>';
    });
    patternHtml += '</div>';
  }
  patternHtml += '</div>';
  // 追加到 formulaSection
}
```

### 7.3 辅助函数

```js
function getBarColor(type) {
  var map = {
    'full': 'seg-full', 'full-alt': 'seg-full-alt',
    'window': 'seg-window', 'window-alt': 'seg-window-alt',
    'compressed': 'seg-compressed', 'rope': 'seg-rope',
    'indexer': 'seg-indexer',
    'fixed': 'seg-fixed', 'fixed-alt': 'seg-fixed-alt',
    'window-empty': 'seg-empty'
  };
  return map[type] || 'seg-full';
}

function getBarHex(type) {
  var map = {
    'full': '#4263eb', 'full-alt': '#5c7cfa',
    'window': '#0c8599', 'window-alt': '#15aabf',
    'compressed': '#ae3ec9', 'rope': '#d9480f',
    'indexer': '#e03131',
    'fixed': '#2b8a3e', 'fixed-alt': '#40c057'
  };
  return map[type] || '#4263eb';
}

function getLegendLabel(type) {
  var map = {
    'full': 'Full KV', 'window': 'Window', 'compressed': 'Compressed',
    'rope': 'RoPE', 'indexer': 'Indexer', 'fixed': 'Fixed state'
  };
  return map[type] || type;
}
```

## 8. CSS 新增 class

```css
/* Bar segment colors — light */
.seg-full      { background: var(--bar-full, #4263eb); }
.seg-full-alt  { background: var(--bar-full-alt, #5c7cfa); }
.seg-window    { background: var(--bar-window, #0c8599); }
.seg-window-alt{ background: #15aabf; }
.seg-compressed{ background: var(--bar-compressed, #ae3ec9); }
.seg-rope      { background: var(--bar-rope, #d9480f); }
.seg-indexer   { background: var(--bar-indexer, #e03131); }
.seg-fixed     { background: var(--bar-fixed, #2b8a3e); }
.seg-fixed-alt { background: var(--bar-fixed-alt, #40c057); }
.seg-empty     { background: #c8cdd8; }

/* Dark theme overrides */
[data-theme="dark"] .seg-window-alt { background: #15aabf; }
[data-theme="dark"] .seg-empty      { background: #3a3f52; }
```

## 9. 不做的事

| 项目 | 原因 |
|---|---|
| 改变 pill 颜色系统 | pill 颜色按语义角色（param/input/result），bar 颜色按 cache 组件，两个维度互补不冗余 |
| 引入 Chart.js / D3 等库 | 纯 HTML+CSS 的 inline bar 足够，零依赖 |
| 3D / 交互式可视化 | 草图验证过，2D inline bar + pattern section 已充分传达信息 |
| 在 Compare 页面添加 | Compare 页面不显示公式，不属于本次 scope |
| Bar 宽度跨行统一 scale | 量级差异会导致小值 bar 不可见，每行独立 scale 更可读 |

## 10. 验收标准

1. 每个 formula-row 右侧有 ibar + ibar-val，颜色与语义对应
2. ibar 各 seg 宽度按比例反映该行各组件字节数
3. ibar-val 显示格式化字节数，与 breakdown 一致
4. Pattern section 只展示 unique 层类型 + 重复次数
5. Legend 只显示当前模型使用的语义色
6. 跨架构同色同义：Full 蓝、Window 青、Compressed 紫、RoPE 橙、Indexer 红、Fixed 绿
7. 暗色模式下 bar 颜色亮度适配
8. GQA / MLA 等单 pattern 架构的 bar 不显冗余（1 个 pattern + ×L 标注）
9. 参数变化时 bar 宽度实时更新（与公式同步重算）
10. 无新增外部依赖
