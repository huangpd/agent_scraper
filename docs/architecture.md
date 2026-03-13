# Agent Scraper 架构文档

## 1. 系统总览

```
用户指令 (自然语言)
       │
       ▼
┌─────────────────────────────────────────────────┐
│                  AgentScraper                    │
│                 (Orchestrator)                   │
│                                                  │
│  ┌───────────┐   ┌────────────┐   ┌───────────┐ │
│  │TaskParser  │──▶│ Navigator  │──▶│   Rule    │ │
│  │(LLM解析)  │   │(browser-use│   │Discoverer │ │
│  │           │   │ Agent导航) │   │(LLM规则)  │ │
│  └───────────┘   └────────────┘   └─────┬─────┘ │
│                                         │       │
│                                         ▼       │
│                  ┌────────────┐   ┌───────────┐ │
│                  │ Extractor  │◀──│   Page    │ │
│                  │(CSS+ML+LLM│   │ Iterator  │ │
│                  │ 三级降级)  │   │(代码遍历) │ │
│                  └──────┬─────┘   └───────────┘ │
│                         │                       │
│                         ▼                       │
│                  ┌────────────┐                  │
│                  │ Formatter  │                  │
│                  │(格式化输出)│                  │
│                  └────────────┘                  │
└─────────────────────────────────────────────────┘
       │
       ▼
  ScrapedResult (JSON/CSV)
```

## 2. 核心模块

### 2.1 数据流 Pipeline

整个系统是一个 **6 步流水线**，每步职责单一：

| 步骤 | 模块 | 输入 | 输出 | 是否调用 LLM |
|------|------|------|------|:---:|
| 1 | TaskParser | 自然语言指令 | ParsedTask | 是 |
| 2 | Navigator | NavigationStep[] | browser + HTML | 是 (Agent) |
| 3 | RuleDiscoverer | HTML + traversal_hints | PageRules | 条件性 |
| 4 | PageIterator | PageRules + browser | HTML[] | 否 |
| 5 | Extractor | HTML[] + ExtractionGoal | dict[str, list] | 首页是 |
| 6 | Formatter | raw_data + goal | ScrapedResult | 否 |

### 2.2 模块详解

#### TaskParser (`pipeline/task_parser.py`)

将自然语言指令解析为结构化任务。

- **输入**：用户自然语言指令（含导航步骤、提取目标、样本数据）
- **输出**：`ParsedTask`（导航步骤 + 提取目标 + 遍历提示）
- **特殊能力**：
  - 自动从指令中提取 JSONL 样本数据
  - 关键词兜底机制：即使 LLM 遗漏遍历意图，关键词匹配也能补全

#### Navigator (`browser/navigator.py`)

控制浏览器完成首次导航，支持图片参考定位。

- **输入**：`NavigationStep[]`（goto / click / wait / input）+ 可选参考截图
- **输出**：`NavigateResult`（browser 实例 + page + 首页 HTML）
- **核心**：使用 browser-use Agent（LLM 驱动的浏览器自动化），浏览器保持存活供后续步骤使用
- **图片参考**：支持传入 base64 截图（红框标注目标元素），通过 `sample_images` 传递给多模态 LLM，提高元素定位准确率
- **Capture 模式**：`navigate_and_capture()` 可在导航同时捕获页面字段值，Agent 通过 done 动作返回 JSON 结果

#### RuleDiscoverer (`extraction/rule_discoverer.py`)

分析页面结构，发现遍历规则。

- **输入**：HTML + traversal_hints
- **输出**：`PageRules`（load_more / pagination / sub_pages / next_button 选择器）
- **优化**：用户未要求遍历时跳过 LLM 调用，直接返回空规则

#### PageIterator (`browser/page_iterator.py`)

纯代码执行遍历，零 AI 调用。

- **输入**：PageRules + browser
- **输出**：所有页面的 HTML 列表
- **支持的遍历模式**：
  - `load_more`：循环点击"加载更多"按钮
  - `sub_pages`：递归进入子页面（最大深度 5）
  - `pagination`：URL 模式翻页（`page/{n}`）
  - `next_button`：点击"下一页"按钮

#### Extractor (`extraction/extractor.py`)

核心提取引擎，三级降级策略。

- **第一优先**：缓存的 CSS 选择器（第 2+ 页复用，零 LLM）
- **第二优先**：AutoScraper ML 匹配（基于样本值的模式学习）
- **第三优先**：LLM 生成 CSS 选择器（兜底）
- **混合模式**：AutoScraper 提取成功的字段 + CSS 补齐缺失字段

#### Formatter (`extraction/formatter.py`)

输出格式化与后处理。

- 字段对齐（截取到最短字段长度）
- 跨页面去重
- URL 模式替换与相对 URL 补全
- 从用户样本自动推断缺失的 URL 构造规则

### 2.3 数据模型 (`core/models.py`)

```
ParsedTask
├── navigation_steps: list[NavigationStep]
│   ├── action: "goto" | "click" | "wait" | "input"
│   ├── target: str
│   ├── description: str
│   └── value: str (input 动作的输入值)
├── extraction_goal: ExtractionGoal
│   ├── fields: dict[str, str]
│   ├── output_format: "json" | "csv"
│   ├── url_pattern: str | None
│   ├── samples: dict[str, list[str]] | None
│   └── traversal_hints: list[str]
└── raw_instruction: str

PageRules
├── load_more_selector: str | None
├── next_button_selector: str | None
├── pagination_url: str | None
├── pagination_max: int | None
├── sub_page_selector: str | None
├── sub_page_url_attr: str
└── sub_page_recursive: bool

ScrapedResult
├── data: list[dict]
├── total_count: int
└── source_url: str
```

## 3. Web 服务层

### 3.1 API 架构

```
FastAPI (server/app.py)
├── POST /api/tasks          → 创建爬取任务（异步执行）
├── GET  /api/tasks/{id}     → 查询任务状态
├── POST /api/tasks/{id}/cancel → 取消任务
├── WS   /ws/{id}            → WebSocket 实时日志
└── /  (静态文件)             → 前端 SPA（生产模式）
```

### 3.2 任务生命周期

```
PENDING → RUNNING → COMPLETED
                  → FAILED
                  → CANCELLED
```

### 3.3 实时通信

- 各模块通过 `logging` 记录日志，`PrintCapture` 拦截输出并转发为 WebSocket `log` 事件
- 结构化事件（`progress`、`result`、`error`）通过 `on_event` 回调推送
- 支持 WebSocket 重连后补发历史事件

## 4. 目录结构

```
browser_use/
├── pyproject.toml                 # 项目配置、依赖、工具设置
├── src/
│   ├── agent_scraper/             # 核心爬虫引擎
│   │   ├── __init__.py            # 公共 API（延迟导入 AgentScraper）
│   │   ├── core/                  # 基础层：模型 + 共享基础设施
│   │   │   ├── __init__.py        # re-export 所有模型
│   │   │   ├── models.py          # Pydantic 数据模型
│   │   │   └── llm.py             # 共享 LLM 客户端工厂
│   │   ├── pipeline/              # 编排层：任务解析 + 流水线调度
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py    # AgentScraper 主类（6 步 Pipeline）
│   │   │   └── task_parser.py     # LLM 指令解析
│   │   ├── browser/               # 浏览器层：导航 + 页面遍历
│   │   │   ├── __init__.py
│   │   │   ├── navigator.py       # Agent 导航 + 值捕获 + 图片参考
│   │   │   └── page_iterator.py   # 页面遍历执行（纯代码）
│   │   └── extraction/            # 提取层：规则发现 + 数据提取 + 格式化
│   │       ├── __init__.py
│   │       ├── rule_discoverer.py # 页面规则发现（LLM）
│   │       ├── extractor.py       # 数据提取（三级降级）
│   │       └── formatter.py       # 输出格式化
│   ├── autoscraper/               # AutoScraper ML 引擎（独立包）
│   │   ├── __init__.py
│   │   ├── auto_scraper.py        # 基于样本的规则学习
│   │   └── utils.py               # 辅助工具函数
│   └── server/                    # Web 服务
│       ├── __init__.py
│       ├── app.py                 # FastAPI + WebSocket
│       └── task_manager.py        # 内存任务管理器
├── docs/                          # 项目文档
├── tests/                         # 单元测试（全 mock，无需真实环境）
├── web/                           # 前端 (Vite + React + TypeScript)
├── run.py                         # CLI 入口
└── run_server.py                  # Web UI 启动脚本
```

### 4.1 依赖方向

```
core ← browser, extraction, pipeline
          ↑                ↑
          └── pipeline ────┘
                ↑
            server/app.py
```

- `core/` 不依赖其他子包
- `browser/`、`extraction/` 只依赖 `core/`
- `pipeline/` 依赖所有子包（编排层）
- `autoscraper/` 是独立顶层包，仅被 `extraction/extractor.py` 使用

## 5. 外部依赖关系

```
agent_scraper
├── openai (AsyncOpenAI)        → LLM 调用（解析、发现、提取）
├── browser-use                 → 浏览器自动化 Agent
│   └── playwright              → 底层浏览器引擎
├── beautifulsoup4              → HTML 解析与 CSS 选择器
├── pydantic                    → 数据模型验证
└── requests                    → AutoScraper HTTP 请求

server
├── fastapi                     → Web 框架
├── uvicorn                     → ASGI 服务器
└── python-dotenv               → 环境变量加载
```
