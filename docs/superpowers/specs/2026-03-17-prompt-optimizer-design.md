# Prompt Optimizer Agent 设计文档

## 1. 问题

用户用自由文本描述爬取需求，当前 TaskParser 做"文本→结构体"的机械翻译，不做策略推理。导致：

- traversal_hints 遗漏（用户说"下一页"但 LLM 未映射到 `next_button`）
- 字段缺失（进详情页但没加 URL 字段，框架无法自动填充）
- mode 误判（capture vs extract 边界模糊）
- 样本格式不规范（字段名不一致、值不是 HTML 原文）
- 遍历策略不优（用 next_button 逐页点击而非 pagination_url 直接跳转）

根本原因：**框架有 4 层隐性知识（能力、策略、避坑、效率），用户不可能知道，TaskParser 也不具备。**

## 2. 方案

新增一个 **PromptOptimizer Agent**（策略专家），插入在用户输入和 TaskParser 之间：

```
之前:  用户(模糊) ──→ TaskParser(猜) ──→ ParsedTask(可能错)
现在:  用户(模糊) ──→ PromptOptimizer(推理) ──→ 完美指令 ──→ TaskParser(零歧义翻译) ──→ ParsedTask(对)
```

### 核心定位

- **不是文本修补器**，是策略编排器
- **不替代 TaskParser**，而是消除 TaskParser 的猜测空间
- **纯推理**，不访问目标网页，基于用户描述 + 框架知识做决策
- **对用户透明**：展示推理过程，用户确认后才执行

### 执行时机

前端拦截：用户在 Web UI 输入后、发送到后端之前，先调优化接口。用户通过弹窗看到推理过程和优化结果，确认/编辑/跳过后再提交执行。

## 3. 知识体系

PromptOptimizer 通过 system prompt 硬编码 4 层框架知识：

### 第 1 层：框架能力图谱

| 维度 | 选项 |
|------|------|
| 模式 | extract（批量提取） / capture（捕获少量值） |
| 遍历 | load_more / next_button / pagination / sub_pages（可组合） |
| 提取 | 有样本→AutoScraper (秒级) / 无样本→browser-use Agent (分钟级) |
| 特殊 | `_source_url` 自动填充 URL 字段、`_fill_missing_url_fields` 样本推断 URL 构造、`url_pattern` 模板 |

### 第 2 层：策略知识

| 网站类型 | 推荐策略 |
|----------|---------|
| 新闻列表站 | sub_pages + next_button, 加 URL 字段 |
| 文件仓库(HuggingFace) | sub_pages + load_more, recursive=true |
| 电商列表 | pagination_url (?page={n}), 判断是否需要进详情 |
| 单页长列表 | load_more, 无需 sub_pages |
| 搜索结果 | pagination_url, 通常不需要进详情页 |

### 第 3 层：避坑知识

- 样本值必须和 HTML 原文完全匹配（空格、全半角敏感）
- URL 字段不能从 HTML 内容提取（是地址栏地址）→ 必须依赖 `_source_url`
- traversal_hints 空 → 单页模式，不翻页
- 没有样本 → 走 browser-use Agent 路径，慢 10x
- pagination + sub_pages 但没有 sub_page_url_filter → 可能抓到导航栏链接

### 第 4 层：执行效率知识

- 有样本 >> 无样本（AutoScraper 秒级 vs Agent 分钟级）
- pagination_url >> next_button（直接跳转 vs 逐页点击）
- 列表页已有所有数据 → 不进详情页（省 N 次页面跳转）
- max_pages 设上限 → 防止跑飞
- 2-3 个样本最优

## 4. 决策树

LLM system prompt 中编码的决策规则：

```
1. 模式选择
   "获取/复制/捕获一个值" → capture
   "提取/获取所有/列表/批量" → extract

2. 是否需要 sub_pages？
   用户要的数据在列表页全部可见 → 不需要
   需要点进去才有完整内容 → 需要 sub_pages → 必须加 URL 字段

3. 翻页策略（按优先级）
   用户提到"下一页" → next_button
   URL 有明显页码参数 → 优先 pagination（更快）
   用户提到"加载更多" → load_more
   都没提 → 单页模式

4. 字段策略
   每个字段必须明确列出
   进详情页 → 必须加 URL 字段（样本值留空 ""）

5. 样本优化
   有样本 → AutoScraper（推荐）
   无样本 → 建议用户补充
   URL 字段样本值写 ""

6. 安全护栏
   有翻页 → 必须有 max_pages（默认 20）
   有 sub_pages → 建议加 URL 过滤关键词
```

## 5. 输入输出

### 输入

用户原始自由文本，如：

```
打开 https://www.ahnews.com.cn/df/hss/pc/lay/node_525.html
获取子页面链接
进入子页获取 title、time
如果有"下一页"链接则翻页

样本数据:
{"title":"千年迎客松 广迎八方客","time":"2026-01-23 18:29:17"}
{"title":"以营商"软实力"夯实发展"硬支撑"","time":"2026-01-23 07:18:03"}
```

### 输出

```json
{
  "optimized": "目标页面: https://www.ahnews.com.cn/df/hss/pc/lay/node_525.html\n提取字段: title(标题), time(发布时间), URL(详情页链接)\n遍历方式: 获取子页面链接，进入详情页提取数据；列表页点击"下一页"翻页\n页数限制: 20\n样本数据:\n{\"title\":\"千年迎客松 广迎八方客\",\"time\":\"2026-01-23 18:29:17\",\"URL\":\"\"}\n{\"title\":\"以营商"软实力"夯实发展"硬支撑"\",\"time\":\"2026-01-23 07:18:03\",\"URL\":\"\"}",
  "reasoning": "① 意图: 批量提取新闻数据 → extract 模式\n② 页面结构: 新闻列表页+详情页结构，标题是链接 → 需要 sub_pages\n③ 遍历: 用户说有\"下一页\" → next_button + sub_pages 组合\n④ 字段: 进详情页 → 加 URL 字段，框架用 _source_url 自动填充\n⑤ 提取: 用户提供了样本 → AutoScraper 路径(快)\n⑥ 护栏: 用户未指定页数 → 默认 max_pages=20",
  "changes": [
    "添加 URL 字段（进入详情页时，框架自动填充详情页地址）",
    "明确 sub_pages + next_button 组合遍历策略",
    "设置页数限制 20（防止无限翻页）",
    "样本数据补充 URL 字段（值为空，框架自动填充）"
  ],
  "skippable": false
}
```

## 6. 架构

### 后端

```
POST /api/optimize
  → PromptOptimizer.optimize(instruction)
    → 一次 LLM 调用 (system prompt 含决策树 + 4层知识)
    → 解析 <reasoning> / <optimized> / <changes> 标签
  → 返回 OptimizeResult

POST /api/tasks  (不改)
  → TaskParser.parse(optimized_instruction)
  → pipeline 正常执行
```

### 前端

```
用户输入 → InputBar.onSend
  → POST /api/optimize → 等待响应
  → 弹出 OptimizePreview 弹窗:
      - 推理过程（只读）
      - 优化后指令（可编辑 textarea）
      - 改动列表
      - [跳过优化] [确认执行] 按钮
  → 用户确认 → POST /api/tasks {instruction: 优化后文本}
  → 用户跳过 → POST /api/tasks {instruction: 原始文本}
```

### 数据模型

```python
@dataclass
class OptimizeResult:
    optimized: str        # 标准化指令文本
    reasoning: str        # 推理过程
    changes: list[str]    # 改动摘要
    skippable: bool       # 指令已规范时 True
```

## 7. 文件变更

| 文件 | 动作 | 内容 |
|------|------|------|
| `src/agent_scraper/pipeline/prompt_optimizer.py` | 新增 | PromptOptimizer 类、system prompt、响应解析 |
| `src/agent_scraper/server.py` | 改 | 新增 `POST /api/optimize` 路由 |
| `web/src/components/OptimizePreview.tsx` | 新增 | 策略预览弹窗组件 |
| `web/src/App.tsx` | 改 | handleSend 加入优化步骤 |
| `web/src/types.ts` | 改 | 新增 OptimizeResponse 类型 |
| `tests/test_prompt_optimizer.py` | 新增 | 策略推理测试用例 |

**零侵入**：task_parser.py、orchestrator.py、reasoner.py 及所有 pipeline/extraction 代码不改。

## 8. 测试策略

单测覆盖典型场景：

| 场景 | 输入 | 期望 |
|------|------|------|
| 新闻列表+下一页 | "获取新闻标题和时间，有下一页" | sub_pages + next_button, +URL 字段, max_pages=20 |
| 文件仓库 | "遍历所有文件夹获取文件名和下载链接" | sub_pages + load_more, recursive |
| 单页列表 | "提取表格中所有数据" | 无遍历, extract 模式 |
| capture 模式 | "复制当前页面的下载链接" | capture 模式 |
| 已规范指令 | 用户已写标准格式 | skippable=true, 不改写 |
| 无样本 | 没提供样本数据 | changes 中提示"建议补充样本以提速" |
