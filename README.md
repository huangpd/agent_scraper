# Agent Scraper

AI 驱动的通用网页数据提取工具。用自然语言描述"去哪里、取什么"，系统自动完成浏览器导航、页面遍历和结构化数据提取。

## 核心特性

- **自然语言驱动** — 无需编写代码或 CSS 选择器，用中文/英文描述即可
- **真实浏览器** — 基于 browser-use + Playwright，支持 JS 渲染和动态加载
- **三级提取降级** — CSS 缓存 → AutoScraper ML → LLM 兜底，兼顾速度与准确率
- **自动遍历** — 支持加载更多、翻页、子页面遍历、下一页按钮
- **图片参考** — 上传截图标注目标元素，多模态 LLM 辅助定位
- **Web UI** — React 前端 + WebSocket 实时日志推送
- **REST API** — FastAPI 后端，支持异步任务管理

## 快速开始

```bash
# 安装
pip install -e ".[dev]"
playwright install chromium

# 配置 .env
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o

# Web UI
python run_server.py
```

## 系统架构

```
用户指令 (自然语言)
       │
       ▼
  TaskParser ──▶ Navigator ──▶ RuleDiscoverer
  (LLM 解析)    (Agent 导航)    (LLM 规则)
                                     │
                                     ▼
                              PageIterator
                              (代码遍历)
                                     │
                              yield 每页 HTML
                                     │
                                     ▼
                              Extractor ──▶ 丢弃 HTML
                              (CSS+ML+LLM)
                                     │
                                     ▼
                               Formatter
                              (格式化输出)
                                     │
                                     ▼
                          ScrapedResult (JSON/CSV)
```

> 流式架构：PageIterator 逐页 yield HTML，Extractor 提取后立即丢弃，内存中始终只保留 1 页 HTML。

## 项目结构

```
src/
├── agent_scraper/              # 核心爬虫引擎（按职责分层）
│   ├── core/                   # 基础层：数据模型 + LLM 客户端工厂
│   ├── pipeline/               # 编排层：ReAct 循环 + 工具注册 + 评估重试
│   ├── browser/                # 浏览器层：导航 + 页面遍历（流式 AsyncGenerator）
│   └── extraction/             # 提取层：规则发现 + 数据提取 + 格式化
├── autoscraper/                # AutoScraper ML 引擎（独立包）
└── server/                     # Web 服务（FastAPI + WebSocket）
tests/                          # 单元测试（全 mock，无需真实环境）
web/                            # 前端（Vite + React + TypeScript）
```


### Web UI

访问 `http://localhost:5173`（开发模式）或 `http://localhost:8000`（生产模式），输入自然语言指令即可。

## 文档

| 文档 | 说明 |
|------|------|
| [架构文档](docs/architecture.md) | 系统总览、模块详解、数据模型、目录结构、依赖关系 |
| [设计文档](docs/design.md) | 项目定位、设计理念、技术决策、非功能性约束 |
| [开发文档](docs/development.md) | 环境搭建、项目结构、开发指南、核心开发场景、调试技巧 |
| [使用手册](docs/user-guide.md) | 快速开始、指令编写、使用示例、准确率优化技巧、常见问题 |

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM 调用 | OpenAI 兼容 API（支持 OpenAI / Azure / 本地 Ollama） |
| 浏览器自动化 | browser-use + Playwright |
| ML 提取 | AutoScraper（基于样本的规则学习） |
| HTML 解析 | BeautifulSoup4 |
| 数据模型 | Pydantic v2 |
| Web 后端 | FastAPI + Uvicorn |
| Web 前端 | React + TypeScript + Vite |
| 测试 | pytest + pytest-asyncio |

## 环境要求

- Python >= 3.11
- Node.js >= 18（前端开发时需要）
- 可用的 OpenAI 兼容 API

## License

MIT
