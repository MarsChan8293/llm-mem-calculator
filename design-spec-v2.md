# KV Cache Calculator — Design Spec v2

## 1. 设计原则

1. **工程师工具感** — 精密、高效、数据密度高，不是消费级 App
2. **暗色优先** — 目标用户是 GPU/推理工程师，暗色是默认环境；亮色作为备选
3. **交互减负** — 少下拉、少点击、少表单行；多用即时反馈的控件
4. **信息层次** — 关键数字大而突出，辅助信息小而收敛

## 2. 主题系统

### 2.1 令牌定义

```css
:root {
  /* ── 语义令牌（不变） ── */
  --bg:            var(--bg-light);
  --surface:       var(--surface-light);
  --surface2:      var(--surface2-light);
  --border:        var(--border-light);
  --text:          var(--text-light);
  --text2:         var(--text2-light);
  --text3:         var(--text3-light);
  --accent:        #6366f1;       /* Indigo — 两主题共用 */
  --accent-light:  var(--accent-light-light);
  --accent-border: var(--accent-border-light);
  --accent-ring:   rgba(99,102,241,0.25);
  --accent2:       #818cf8;
  --green:         #22c55e;
  --green-light:   var(--green-light-light);
  --orange:        #f59e0b;
  --orange-light:  var(--orange-light-light);
  --orange-dark:   #d97706;
  --red:           #ef4444;
  --radius-sm:     6px;
  --radius-md:     8px;
  --radius-lg:     12px;
  --radius-pill:   999px;
}

/* ── 亮色原始值 ── */
:root {
  --bg-light:            #ffffff;
  --surface-light:       #f8f9fb;
  --surface2-light:      #eef0f4;
  --border-light:        #e2e5eb;
  --accent-light-light:  #eef2ff;
  --accent-border-light: #c7d2fe;
  --green-light-light:   #dcfce7;
  --orange-light-light:  #fef3c7;
  --text-light:          #0f172a;
  --text2-light:         #475569;
  --text3-light:         #94a3b8;
}

/* ── 暗色原始值 ── */
[data-theme="dark"] {
  --bg:            #0c0e14;
  --surface:       #161923;
  --surface2:      #1e2231;
  --border:        #2a2f3e;
  --text:          #e2e8f0;
  --text2:         #94a3b8;
  --text3:         #64748b;
  --accent-light:  rgba(99,102,241,0.12);
  --accent-border: rgba(99,102,241,0.25);
  --accent-ring:   rgba(99,102,241,0.35);
  --green-light:   rgba(34,197,94,0.12);
  --orange-light:  rgba(245,158,11,0.12);
}
```

### 2.2 主色变更

| 元素 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| 强调色 | `#3b5bdb` 蓝 | `#6366f1` Indigo | 更现代，暗色下发光感更好 |
| 绿色 | `#2b8a3e` 深绿 | `#22c55e` 亮绿 | 暗色下可见度大幅提升 |
| 橙色 | `#d9480f` 烧橙 | `#f59e0b` 琥珀 | 暗色下更醒目，亮色下更柔和 |
| 红色 | `#e03131` | `#ef4444` | 对齐 Tailwind 色板，暗色下更亮 |
| 圆角 | 8px 一刀切 | 6/8/12/999px 四级 | pill 用全圆角，卡片用 12px 更现代 |

### 2.3 主题切换

- 导航栏右侧加 🌙/☀️ 切换按钮
- `data-theme="dark"` 属性挂在 `<html>` 上
- 默认跟随 `prefers-color-scheme`，用户手动切换后存 `localStorage`
- 切换时 body 加 `transition: background 0.2s, color 0.2s`

## 3. 布局

### 3.1 页面框架

```
┌─────────────────────────────────────────────────────────────┐
│  ● KV Cache Calculator          [Calculator] [Compare]  🌙 │  ← 导航栏：固定顶部
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────┐  ┌────────────────────────────────┐  │
│   │                 │  │                                │  │
│   │   左侧面板       │  │     右侧结果区                  │  │
│   │   (380px)       │  │     (1fr)                      │  │
│   │                 │  │                                │  │
│   └─────────────────┘  └────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

- 左栏从 320px → **380px**，给模型选择器更多空间
- 导航栏 **固定顶部**（`position: sticky`），滚动时始终可见
- 去掉页面内的 `<h1>` 和 `<p class="subtitle">`，信息已由导航栏品牌名承载

## 4. Calculator 页面 — 交互重设计

### 4.1 当前问题

8 个垂直堆叠的表单控件，像填报表：

```
Model family    [▼ DeepSeek      ]
Model           [▼ V4 Pro        ]
Tokens          [  1024          ]
Sequences       [  1             ]
KV precision    [▼ FP8 / INT8    ]
Indexer prec.   [▼ FP4 / INT4    ]
□ Include draft KV cache
□ Include linear attention KV
```

**痛点**：
- 选模型要两步（family → model），但 90% 用户直接知道要哪个模型
- Precision 下拉只有 3 个选项，用下拉太重
- 每次切换 family 后 model 重置，打断心流
- 8 行控件视觉上很闷

### 4.2 新交互方案

```
┌──────────────────────────────────┐
│ 🔍 Search or select a model...   │  ← 搜索式模型选择器（复用 Compare 的 picker）
│                                  │
│  ▼ DeepSeek                      │
│    V4 Pro     ✓                  │     ← 选中模型自动折叠 picker
│  ▼ GLM                           │
│  ...                             │
├──────────────────────────────────┤
│                                  │
│  Context length                  │
│  ┌──────────────────────────┐    │
│  │  1,024               ✎   │    │  ← 大号输入，带千位分隔
│  └──────────────────────────┘    │
│                                  │
│  Batch size                      │
│  ┌──────┐ ┌──────┐ ┌──────┐    │
│  │  1   │ │  4   │ │  8   │    │  ← 快捷按钮 + 自定义输入
│  └──────┘ └──────┘ └──────┘    │
│                                  │
│  KV precision                    │
│  ┌──────┐ ┌──────┐ ┌──────┐    │
│  │ BF16 │ │ FP8  │ │ FP4  │    │  ← Segmented control，非下拉
│  └──────┘ └──────┘ └──────┘    │
│                                  │
│  Indexer precision               │     ← 仅 DSA 模型显示
│  ┌──────┐ ┌──────┐ ┌──────┐    │
│  │ BF16 │ │ FP8  │ │ FP4  │    │
│  └──────┘ └──────┘ └──────┘    │
│                                  │
│  ┌──────────────────────────┐    │
│  │ ○ Draft KV    ○ Linear  │    │  ← Toggle switch，非 checkbox
│  └──────────────────────────┘    │
└──────────────────────────────────┘
```

### 4.3 控件替换规则

| 控件 | 旧 | 新 | 原因 |
|---|---|---|---|
| 模型选择 | Family 下拉 + Model 下拉 | 搜索式 Picker（复用 Compare 组件） | 一步直达，支持搜索，与 Compare 交互统一 |
| Context length | 普通数字输入 | 大号输入框 + 千位分隔显示 | 核心参数，视觉强调 |
| Batch size | 普通数字输入 | 快捷按钮 (1/4/8/32) + 自定义输入 | 常用值一键选，减少输入 |
| KV precision | 下拉 3 项 | Segmented control (BF16 / FP8 / FP4) | 3 个选项用下拉太浪费，按钮组一目了然 |
| Indexer precision | 下拉 3 项 | Segmented control | 同上 |
| Draft / Linear | Checkbox | Toggle switch | 更现代，更紧凑 |
| Family 下拉 | 独立控件 | **删除** — Picker 按 family 分组即可 | 消除冗余 |

### 4.4 Segmented Control 规范

```
┌──────┐ ┌──────┐ ┌──────┐
│ BF16 │ │ FP8  │ │ FP4  │    ← 未选中：透明底，文字色 --text2
└──────┘ └──────┘ └──────┘    ← 选中：--accent 底，白色字
```

- 3 个选项水平排列，圆角 `var(--radius-md)`
- 选中项：`background: var(--accent); color: #fff; font-weight: 600`
- 未选中：`background: transparent; color: var(--text2)`
- hover 未选中项：`background: var(--surface2)`
- 整组外框：`border: 1px solid var(--border); border-radius: var(--radius-md); padding: 3px`

### 4.5 Toggle Switch 规范

```
○ Draft KV          ● Linear KV
  off                  on
```

- 尺寸：36px × 20px，圆形滑块 16px
- off 状态：`background: var(--surface2); 滑块靠左`
- on 状态：`background: var(--accent); 滑块靠右`
- 带标签文字，标签在右侧

### 4.6 Context Length 输入框

- 字号：`1.25rem`，字体 `var(--mono)`，加粗
- 宽度 100%
- 值显示千位分隔（`1,024`），实际 value 保持数字
- 底部滑动条（range slider）同步：0 → model.max_position_embeddings
  - 滑动条分段标记：0, 128K, 256K, 512K, 1M
  - 输入框和滑动条双向绑定

## 5. Compare 页面 — 交互优化

### 5.1 改动较小，主要是视觉升级

- Picker 和 Tags 样式适配暗色
- Chart 区域：暗色下网格线更微妙（`var(--border)` 透明度降低）
- 导出按钮：升级为有图标（↓ PNG / 📋 Copy）

### 5.2 新增：模型快捷预设

在 Picker 上方加一排快捷按钮：

```
[ Top 5 Models ]  [ DeepSeek Family ]  [ All Qwen ]
```

点击即选中对应模型组，免去逐个点击。

## 6. 右侧结果区 — Calculator

### 6.1 Total Cache Size

**现在**：小蓝底框 + 0.00796 GiB

**新设计**：
- 去掉蓝底框，改为纯数字大字
- 字号 `2rem`，字体 `var(--mono)`，颜色 `var(--accent2)`
- 下方小字显示 GB 换算和 per-token 大小
- 单位选择器（GiB / GB / MiB）— segmented control

### 6.2 Metrics 行

**现在**：3 个等宽 metric box

**新设计**：
- 去掉边框，改为紧凑的水平排列
- `KV Cache  7.68 MiB    Indexer  480 KiB    Per Token  8.15 KiB`
- 用 `·` 分隔，更紧凑

### 6.3 公式区

保持 pill 设计，适配暗色：
- pill-param: 暗色下 `background: rgba(99,102,241,0.12); color: #818cf8`
- pill-input: 暗色下 `background: rgba(245,158,11,0.12); color: #fbbf24`
- pill-result: 暗色下 `background: rgba(34,197,94,0.12); color: #4ade80`

### 6.4 Breakdown

**现在**：双列网格 + 斑马纹

**新设计**：
- 改为单列 key-value 列表
- 每行：`Label ···· Value`，用 dot leader 填充
- 去掉斑马纹，用细分隔线
- Tooltip 保持

## 7. 导航栏

### 7.1 新设计

```
● KV Cache Calculator          [ Calculator ] [ Compare ]     🌙
```

- 背景：`var(--surface)` + 底部 `1px solid var(--border)`
- 品牌名：`font-weight: 700; font-size: 0.95rem`
- ● 品牌名前的圆点：`var(--accent)` 色，8px，暗示"在线/运行中"
- Tab 按钮：pill 式
  - 未选中：`padding: 6px 16px; border-radius: var(--radius-pill); color: var(--text2)`
  - 选中：`background: var(--accent); color: #fff`
  - hover 未选中：`background: var(--surface2)`
- 主题切换：圆形按钮 `24px`，里面 🌙 或 ☀️ 图标（SVG，不用 emoji）

## 8. 字体

| 用途 | 旧 | 新 | 原因 |
|---|---|---|---|
| 正文 | Inter | **DM Sans** | 几何感更强，辨识度高，不如 Inter 泛滥 |
| 等宽 | JetBrains Mono | JetBrains Mono（保持） | 已是最好的代码字体 |
| 标题 | Inter Bold | **JetBrains Mono Bold** | 工程师工具用等宽做标题更有气质 |

### 引入

```html
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
```

### CSS

```css
--sans: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
--mono: 'JetBrains Mono', 'Fira Code', monospace;
```

## 9. 动效

### 9.1 页面加载

- 卡片 `opacity: 0 → 1` + `translateY(8px) → 0`，stagger 100ms
- 数字从 0 跳到目标值（countUp），持续 400ms

### 9.2 交互反馈

- Segmented control 切换：滑块 `transition: left 0.15s ease`
- Toggle switch：`transition: background 0.2s, transform 0.2s`
- 主题切换：`transition: background 0.2s, color 0.2s, border-color 0.2s`
- Pill hover：微微放大 `scale(1.05)` + tooltip 显现

### 9.3 不要做的

- ❌ 不做页面滚动视差
- ❌ 不做粒子/3D 背景
- ❌ 不做 loading 骨架屏（页面足够轻）

## 10. 实施优先级

| 阶段 | 内容 | 影响范围 |
|---|---|---|
| **P0** | 暗色主题 + 主题切换 + 色彩令牌 | CSS :root + 少量 HTML |
| **P0** | Segmented control 替换下拉 | calculator.js + compare.js + CSS |
| **P0** | 搜索式模型选择器替换双下拉 | calculator.js + CSS |
| **P1** | Toggle switch 替换 checkbox | CSS + calculator.js |
| **P1** | 导航栏 pill 式 Tab + 主题切换按钮 | CSS + HTML |
| **P1** | Context length slider | calculator.js + CSS |
| **P2** | Batch size 快捷按钮 | calculator.js + CSS |
| **P2** | 结果区重设计（大数字 + 单位切换 + 紧凑 metrics） | calculator.js + CSS |
| **P2** | 字体替换 DM Sans | HTML link + CSS |
| **P3** | 页面加载动效 | CSS / JS |
| **P3** | Compare 快捷预设 | compare.js |
