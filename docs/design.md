# Agent Scraper 设计文档

## 1. 项目定位

Agent Scraper 是一个 **AI 驱动的通用网页数据提取工具**。用户只需用自然语言描述"去哪里、取什么"，系统自动完成浏览器导航、页面遍历和结构化数据提取。

### 1.1 核心设计理念

- **自然语言驱动**：用户无需编写代码或 CSS 选择器，用中文/英文描述即可
- **AI 与代码分工**：AI 负责"理解"（解析指令、发现规则、生成选择器），代码负责"执行"（遍历页面、点击按钮、提取数据）
- **渐进降级**：多层提取策略（CSS 缓存 → AutoScraper ML → LLM 兜底），兼顾速度与准确率
- **实时可观测**：通过事件回调和 WebSocket 实时推送进度日志

## 2. 解决的问题

| 传统爬虫痛点 | Agent Scraper 方案 |
|---|---|
| 每个网站写不同的解析代码 | LLM 自动理解页面结构，生成 CSS 选择器 |
| 动态加载内容抓不到 | browser-use 控制真实浏览器，支持 JS 渲染 |
| 翻页/加载更多逻辑各不相同 | AI 发现遍历规则，代码统一执行 |
| 修改需求需要改代码 | 修改自然语言指令即可 |

## 3. 设计决策

### 3.1 为什么用 browser-use 而非 requests

项目面对的目标页面大多是 SPA 或含 JS 动态加载的页面（如 HuggingFace），纯 HTTP 请求无法获取渲染后内容。browser-use 封装了 Playwright + LLM Agent，可以像人一样操作浏览器。

### 3.2 为什么 AI 只做"发现"、代码做"执行"

早期方案让 AI Agent 完成全部操作（包括翻页、提取），但存在：
- Token 消耗巨大（每次翻页都要调用 LLM）
- 行为不稳定（LLM 可能遗漏某些页面）
- 速度慢（每步都等 LLM 响应）

当前方案：AI 只在第一页分析一次规则，后续所有页面由代码机械执行，做到 **零 LLM 调用**。

PageIterator 采用 **流式 AsyncGenerator 架构**：逐页 yield HTML → IteratePagesTool 内联调用 Extractor 提取 → 丢弃 HTML → yield 下一页。内存中始终只保留 1 页 HTML，1000 页场景下内存占用从 ~200MB 降至 ~200KB。

### 3.3 三级提取降级策略

```
第 2+ 页 → 缓存的 CSS 选择器（零 LLM，最快）
    ↓ 失败
第 1 页 → AutoScraper ML 匹配（零 LLM）
    ↓ 部分字段缺失
混合模式 → AutoScraper 成功的字段 + CSS Selector 补齐缺失字段
    ↓ 全部失败
全量 CSS → LLM 生成 CSS 选择器兜底
```

### 3.4 为什么用 OpenAI SDK 而非直接用 Anthropic/其他

项目通过 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY` 环境变量配置，兼容所有 OpenAI API 协议的服务（OpenAI、Azure、国内代理、本地 Ollama 等）。这是最通用的 LLM 接入方式。

### 3.5 模块化分层架构

项目按职责域划分为四个子包，依赖方向单向清晰：

- **core/**（基础层）：数据模型 + 共享 LLM 客户端工厂，不依赖其他子包
- **browser/**（浏览器层）：导航 + 页面遍历，仅依赖 core
- **extraction/**（提取层）：规则发现 + 数据提取 + 格式化，仅依赖 core
- **pipeline/**（编排层）：隐式 FSM 编排 + Tool 注册 + 评估重试（演进方向：显式 FSM），依赖所有子包
- **autoscraper/**（独立包）：ML 提取引擎，作为独立顶层包存在

LLM 客户端统一由 `core/llm.py` 工厂创建，避免各模块重复配置。

### 3.6 图片参考功能

Navigator 支持传入 base64 编码的参考截图。用户可在 Web UI 上传截图并用红框标注目标元素，系统将截图通过 `sample_images` 参数传递给多模态 LLM，帮助 Agent 精准定位页面元素。

完整链路：UI (FileReader.readAsDataURL) → REST API → Navigator._convert_images() → browser_use Agent(sample_images=...)

### 3.7 编排模式选型：为什么选 FSM 而非纯 ReAct

#### 问题：纯 ReAct 不适合爬虫 Agent

ReAct（Reasoning + Action）的核心假设是每一步都需要 LLM 推理决定下一步做什么。但爬虫任务的特点与此矛盾：

| 特点 | ReAct 假设 | 爬虫现实 |
|------|-----------|---------|
| 决策复杂度 | 每步都需推理 | 90% 的步骤是确定性的 |
| 分支数量 | 开放式 | 有限（capture / extract / free 三条路径） |
| 执行体量 | 少量步骤 | 可能遍历 1000+ 页 |
| 容错模式 | 自由探索纠错 | 错误模式有限且可枚举 |
| 成本敏感度 | 可接受 | 每页调 LLM 不可接受 |

实测验证：原生 browser-use 让 LLM Agent 负责翻页操作，5 步后仍在打转（点错元素、开新 tab、找不到"下一页"）。这是把确定性任务交给概率模型的典型失败。

#### 结论：宏观 FSM 编排 + 微观节点内 Agent

分层架构是最优平衡点：

- **宏观层（FSM）**：显式状态机编排整体流程，状态转换是确定性的
- **微观层（ReAct）**：在单个 FSM 节点内部保留 browser-use Agent 的自由探索

```
FSM 状态流转:
  PARSING → ROUTING → NAVIGATING → DISCOVERING → ITERATING → EVALUATING → FORMATTING → DONE
                         │                                        │
                    (内部 Agent                              fail │
                     自由探索，                                   ▼
                     max_steps                              RETRYING
                     硬上限)                            clear_cache → ITERATING
                                                        retry_nav  → NAVIGATING
                                                        skip       → FORMATTING
```

**受控自由度原则**：Navigator 内部的 browser-use Agent 可以自由探索（点按钮、等加载、重试），但它必须在 max_steps 内返回结果。FSM 外层只关心：成功了去下一个状态，失败了去 RETRYING。

#### 当前实现状态

当前 `Reasoner` 本质上是一个**隐式 FSM**——用 for 循环 + list + if/else 表达状态转换：

```python
# 计划是程序化生成的（不调 LLM）
plan = _create_default_plan(task)
# 执行是顺序线性的
for step in plan: await tool.execute()
# 重试策略是枚举值（不调 LLM）
plan = _apply_retry_strategy(strategy)
```

**演进方向**：将散落的 if/else 收拢为显式转换表 `TRANSITIONS = {(State, Event) → State}`，实现：
- 所有可能的状态转换一目了然
- 新增状态只需加转换行，不影响现有逻辑
- 非法转换直接报错
- 事件回调天然对齐状态

#### 各模式对比

| 维度 | 纯 ReAct | 当前隐式 FSM | 显式 FSM（目标） |
|------|---------|------------|----------------|
| 确定性 | 低 | 中 | 高（转换表声明式） |
| 可观测性 | 低 | 中（有 events） | 高（状态自带含义） |
| 可扩展性 | 高（但不可控） | 中（加分支要改多处） | 高（加状态 + 转换行） |
| LLM 开销 | 高（每步都调） | 低（3~4 次） | 低（不变） |
| 终止保证 | 无 | 有（max_retries） | 有（状态 + 计数器） |

## 4. 非功能性约束

- **Python ≥ 3.11**：使用了 `type | None` 等新语法
- **浏览器环境**：需要 Playwright 安装 Chromium
- **LLM 依赖**：需要可用的 OpenAI 兼容 API
- **内存限制**：HTML 片段截取有上限（`MAX_HTML_SIZE`），防止超出 LLM 上下文窗口
- **流式处理**：PageIterator 以 AsyncGenerator 逐页 yield HTML，提取后立即丢弃，内存中始终只保留 1 页 HTML + 累积的结构化数据，避免大量页面 HTML 堆积
