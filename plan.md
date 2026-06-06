# KV Cache Calculator — Compare Feature & Refactoring Plan

## 1. 目标

在现有 KV Cache Calculator 基础上增加模型对比功能，生成折线图并支持导出 PNG。同时为 GitHub Pages 部署和未来模型扩展做好架构准备。

## 2. 部署方案：GitHub Pages

- 单仓库部署，`main` 分支根目录即为站点根目录
- 无需构建工具，纯静态文件
- 访问地址：`https://<username>.github.io/llm-mem-calculator/`
- 文件结构扁平，GitHub Pages 直接 serve

## 3. 项目重构

### 3.1 文件拆分

当前 `index.html` 约 1015 行，CSS/JS/HTML 全部内联。为可维护性和未来扩展，拆分为：

```
llm-mem-calculator/
├── index.html            # Calculator 页面（精简为 HTML 骨架 + 引用）
├── compare.html          # Compare 页面（新增）
├── css/
│   └── main.css          # 共享样式（现有 <style> 提取）
├── js/
│   ├── data.js           # 模型数据定义（DATA 对象提取）
│   ├── calc.js           # 计算引擎（公式逻辑提取为纯函数）
│   ├── calculator.js     # Calculator 页面交互逻辑
│   └── compare.js        # Compare 页面逻辑（新增）
└── plan.md
```

### 3.2 data.js — 模型数据模块

```js
// 集中管理所有模型数据，新增模型只需在此文件追加
const MODEL_DATA = {
  precision_options: [...],
  indexer_precision_options: [...],
  models: [...]
};
```

未来扩展：
- 新增模型：追加一个对象到 `models` 数组即可
- 新增公式类型：在 `calc.js` 中增加对应公式函数，模型数据中引用 formula 名

### 3.3 calc.js — 计算引擎

提取为纯函数，无 DOM 依赖：

```js
/**
 * 计算单个模型在指定参数下的 KV cache 结果
 * @returns { kvBytes, idxBytes, perTokenBytes, breakdown, formulas, formulaTitle }
 */
function calcKvCache(model, tokens, precB, idxB, options) { ... }

/**
 * 计算单个模型在多个 token 点的 KV cache GB 值（用于绘图）
 * @returns Array<{ tokens, gb }>
 */
function calcKvCacheSeries(model, tokenPoints, precB, idxB, options) { ... }
```

### 3.4 calculator.js — 现有计算器交互

从 index.html 的 `<script>` 中提取 DOM 交互逻辑，调用 `calc.js`。

### 3.5 compare.js — 新增对比功能

绘图 + 交互逻辑，调用 `calc.js`。

## 4. Compare 页面设计

### 4.1 页面结构

- 顶部导航栏：`Calculator` | `Compare` 两个链接，当前页高亮
- 主体两栏布局（与 Calculator 风格一致）

```
┌──────────────────────────────────────────────────┐
│  KV Cache Calculator    [Calculator] [Compare]   │
├───────────────┬──────────────────────────────────┤
│  Controls     │                                  │
│  ───────────  │         Chart Area               │
│  🔍 Search... │                                  │
│  DeepSeek  ▼  │     ┌──────────────────┐         │
│   V4 Pro  [+] │     │   折线图          │         │
│   V4 Flash[+] │     │                  │         │
│  GLM       ▼  │     └──────────────────┘         │
│   GLM-5   [+] │                                  │
│  ...          │     [Download PNG] [Copy]        │
│  ───────────  │                                  │
│  Selected(3): │                                  │
│  ■ V4 Flash × │                                  │
│  ■ M2      × │                                  │
│  ■ Qwen3.5 × │                                  │
│  ───────────  │                                  │
│  Precision:   │                                  │
│  [FP8/INT8 ▼] │                                  │
│  Sequences:   │                                  │
│  [1        ]  │                                  │
│  □ Draft KV   │                                  │
│  □ Linear KV  │                                  │
│  ───────────  │                                  │
│  Chart Title: │                                  │
│  [__________] │                                  │
│  X Max:       │                                  │
│  [1M       ]  │                                  │
└───────────────┴──────────────────────────────────┘
```

### 4.2 模型选择 — Tag/Chip 选择器

采用搜索 + 分组列表 + 已选 Tag 的三段式交互：

**① 搜索框**
- 输入关键词实时过滤模型列表（模糊匹配 model label）
- 清空搜索恢复完整列表

**② 分组模型列表**
- 按 family 分组，family 标题可点击展开/折叠（默认展开）
- 每个模型右侧 `[+]` 按钮，点击选中
- 已选模型 `[+]` 变为 `[✓]`，再次点击取消
- 选中数量达上限 10 时，未选模型的 `[+]` 置灰不可点击
- 搜索时隐藏不匹配的 family 和模型

**③ 已选 Tag 区域**
- 显示在列表下方，标题 `Selected (n/10):`
- 每个已选模型渲染为彩色 Tag：`■ 模型名 ×`
  - `■` 色块颜色与对应折线颜色一致（按 PALETTE 顺序分配）
  - `×` 点击移除该模型
- Tag 排列顺序 = 选中顺序，决定折线颜色分配
- 拖拽 Tag 可调整顺序（调整后折线颜色跟随重分配，图表实时重绘）

**交互流程：**
1. 搜索/浏览 → 点击 `[+]` → Tag 出现 + 折线出现
2. 点击 Tag `×` → Tag 消失 + 折线消失
3. 拖拽 Tag → 颜色重分配 + 图表重绘
4. 达 10 个上限 → 提示 "最多选择 10 个模型"

### 4.3 参数配置

| 参数 | 说明 |
|---|---|
| KV Precision | 下拉选择（复用 precision_options） |
| Sequences | 批次大小（默认 1） |
| Draft KV | 仅当所选模型支持时显示 |
| Linear KV | 仅当所选模型包含 Qwen hybrid 时显示 |
| Chart Title | 可编辑，默认 "KV Cache Comparison" |
| X Axis Max | 序列长度上限，下拉：100K / 200K / 500K / 1M / 2M |

### 4.4 绘图 — Chart.js

通过 CDN 引入：`<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>`

**样式配置（对标 spec + 整体风格协调）：**

| Spec 要求 | 实现 |
|---|---|
| 白底画布 | `backgroundColor: '#ffffff'` |
| 浅灰虚线网格 | `grid: { color: '#e2e5eb', borderDash: [4, 4] }` （复用 CSS --border 色） |
| 仅左/下边框 | `border: { display: false }` on top/right |
| 实线无圆点无填充 | `pointRadius: 0`, `fill: false`, `borderWidth: 2` |
| 左上角图例白底黑框 | `plugins.legend: { position: 'top', align: 'start' }` + 自定义 legend box |
| 标题顶部居中 | `plugins.title: { display: true, align: 'center', font: { weight: 'bold' } }` |
| X 轴 K/M 单位 | tick callback：`>= 1M` 显示 `xM`，`>= 1K` 显示 `xK` |
| Y 轴 GB 单位 | tick callback：整数显示 `x GB` |

**配色方案（与现有 UI 蓝调协调）：**

选用饱和度适中、彼此可区分的色板：

```js
const PALETTE = [
  '#3b5bdb',  // 藏青蓝（主色 --accent）
  '#e03131',  // 暗红
  '#2b8a3e',  // 草绿（--green）
  '#e67700',  // 橘橙（--yellow）
  '#9c36b5',  // 粉紫
  '#0c8599',  // 青蓝
  '#c2255c',  // 玫红
  '#5c940d',  // 橄榄绿
  '#d9480f',  // 焦橙
  '#6741d9',  // 靛紫
];
```

按模型选中顺序依次分配颜色。

### 4.5 导出

- **Download PNG**：`chart.toBase64Image('image/png', 1.0)` → 创建 `<a download="kv-cache-compare.png">`
- **Copy to Clipboard**：`canvas.toBlob()` → `navigator.clipboard.write()` （不支持时降级提示）

### 4.6 交互细节

- 任何参数变化 → 实时重绘图表
- 模型选择变化 → 实时重绘
- 悬浮折线显示 tooltip（模型名 + tokens + GB 值）
- 不支持所选精度的模型自动跳过并提示

## 5. 共享样式 (css/main.css)

从 index.html `<style>` 提取，两个页面共用：

- CSS 变量（颜色、字体、圆角）
- 基础重置
- 导航栏样式（新增）
- 表单控件样式
- 结果卡片样式

新增导航栏：

```css
.nav { display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
.nav a { padding: 10px 20px; font-weight: 600; color: var(--text2); text-decoration: none; border-bottom: 2px solid transparent; }
.nav a:hover { color: var(--accent); }
.nav a.active { color: var(--accent); border-bottom-color: var(--accent); }
```

## 6. 实施步骤

1. **拆分文件**：将 index.html 的 CSS/JS 提取为独立文件，确认功能无损
2. **提取计算引擎**：calc.js 纯函数化，calculator.js 调用 calc.js
3. **创建 compare.html**：页面骨架 + 导航栏
4. **实现 compare.js**：模型选择 → 参数配置 → 调用 calc.js 生成数据 → Chart.js 绘图
5. **导出功能**：Download PNG + Copy to Clipboard
6. **GitHub Pages 配置**：确认仓库设置，验证部署

## 7. 不做的事

- ❌ 不引入构建工具/打包器
- ❌ 不添加不存在于 DATA 中的模型
- ❌ 不做服务端逻辑
- ❌ 不做响应式图表（固定合理宽度即可）
