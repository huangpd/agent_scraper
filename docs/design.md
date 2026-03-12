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

## 4. 非功能性约束

- **Python ≥ 3.12**：使用了 `type | None` 等新语法
- **浏览器环境**：需要 Playwright 安装 Chromium
- **LLM 依赖**：需要可用的 OpenAI 兼容 API
- **内存限制**：HTML 片段截取 50KB 上限，防止超出 LLM 上下文窗口
