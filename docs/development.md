# Agent Scraper 开发文档

## 1. 环境搭建

### 1.1 前置条件

- Python ≥ 3.12
- Node.js ≥ 18（前端开发时需要）
- 可用的 OpenAI 兼容 API

### 1.2 安装

```bash
# 克隆项目
git clone <repo-url>
cd browser_use

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# 安装项目（可编辑模式 + 开发依赖）
pip install -e ".[dev]"

# 安装 Playwright 浏览器
playwright install chromium
```

### 1.3 环境变量

创建 `.env` 文件：

```env
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
```

支持任何 OpenAI API 兼容服务：

| 服务 | BASE_URL 示例 |
|------|--------------|
| OpenAI 官方 | `https://api.openai.com/v1` |
| Azure OpenAI | `https://<name>.openai.azure.com/openai/deployments/<deploy>/` |
| 本地 Ollama | `http://localhost:11434/v1` |
| 国内代理 | 视代理商而定 |

## 2. 项目结构

```
src/
├── agent_scraper/       # 核心库（可独立使用）
│   ├── models.py        # 数据模型 → 修改字段/新增模型从这里开始
│   ├── orchestrator.py  # Pipeline 编排 → 调整步骤顺序/新增步骤
│   ├── task_parser.py   # 指令解析 → 修改 Prompt / 新增 action 类型
│   ├── navigator.py     # 浏览器导航 → 修改 Agent 行为
│   ├── rule_discoverer.py # 规则发现 → 新增遍历模式
│   ├── page_iterator.py # 页面遍历 → 新增遍历执行逻辑
│   ├── extractor.py     # 数据提取 → 调整降级策略/Prompt
│   ├── formatter.py     # 输出格式化 → 新增输出格式
│   └── autoscraper/     # ML 引擎 → 一般不需要修改
└── server/              # Web 层（依赖 agent_scraper）
    ├── app.py           # API 路由 → 新增接口
    └── task_manager.py  # 任务管理 → 修改任务生命周期
```

## 3. 开发指南

### 3.1 运行测试

```bash
# 运行全部测试
pytest

# 运行单个文件
pytest tests/test_extractor.py

# 查看详细输出
pytest -v -s

# 生成覆盖率报告
pytest --cov=agent_scraper --cov=server
```

测试全部使用 mock，不依赖真实浏览器或 LLM API。

### 3.2 启动开发服务器

```bash
# CLI 方式运行（直接执行爬取）
python run.py

# Web UI 开发模式（Vite + FastAPI 同时启动）
python run_server.py

# Web UI 生产模式（先 build 前端，再启动 FastAPI）
python run_server.py --prod
```

### 3.3 代码规范

项目使用 Ruff 进行代码检查：

```bash
# 检查
ruff check src/ tests/

# 自动修复
ruff check --fix src/ tests/

# 格式化
ruff format src/ tests/
```

配置在 `pyproject.toml` 中：
- 行长度：100 字符
- 规则：E（错误）、F（pyflakes）、I（import 排序）

## 4. 核心开发场景

### 4.1 新增一种遍历模式

例如新增 "infinite_scroll"（无限滚动）：

**步骤 1**：`models.py` — 在 `PageRules` 中添加字段

```python
class PageRules(BaseModel):
    # ... 已有字段 ...
    infinite_scroll: bool = False  # 是否无限滚动加载
```

**步骤 2**：`task_parser.py` — 更新 `_ensure_traversal_hints`

```python
checks = {
    # ... 已有 ...
    "infinite_scroll": ["无限滚动", "滚动加载", "infinite scroll"],
}
```

**步骤 3**：`rule_discoverer.py` — 更新 Prompt 和过滤逻辑

**步骤 4**：`page_iterator.py` — 在 `iterate()` 中添加执行逻辑

**步骤 5**：添加测试

### 4.2 新增提取输出格式

例如新增 Excel 输出：

**步骤 1**：`formatter.py` — 添加 `to_excel()` 静态方法

**步骤 2**：`models.py` — `ExtractionGoal.output_format` 增加可选值

### 4.3 修改 LLM Prompt

所有 Prompt 定义在各模块文件顶部的常量中：

| Prompt | 文件 | 用途 |
|--------|------|------|
| `PARSE_PROMPT` | `task_parser.py` | 解析自然语言指令 |
| `DISCOVER_PROMPT` | `rule_discoverer.py` | 发现页面遍历规则 |
| `SAMPLE_PROMPT` | `extractor.py` | LLM 采样生成样本 |
| `CSS_SELECTOR_PROMPT` | `extractor.py` | 生成 CSS 选择器 |

修改 Prompt 时注意：
- 保持 JSON 输出格式不变（下游解析依赖固定格式）
- 使用 `{{` 和 `}}` 转义 Python format 字符串中的花括号
- 修改后运行相关测试验证

### 4.4 新增 API 接口

在 `server/app.py` 中添加路由：

```python
@app.get("/api/tasks")
async def list_tasks():
    return [
        {"task_id": tid, "status": info.status.value}
        for tid, info in manager.tasks.items()
    ]
```

## 5. 调试技巧

### 5.1 查看 LLM 交互

每个 LLM 调用的输入输出都通过 `print()` 记录，运行时可在终端直接看到：
- `[TaskParser]`：指令解析结果
- `[RuleDiscoverer]`：发现的规则详情
- `[Extractor]`：提取策略和各步骤结果
- `[PageIterator]`：遍历进度

### 5.2 headless 模式切换

```python
# 开发调试：显示浏览器窗口
scraper = AgentScraper(headless=False)

# 生产部署：无头模式
scraper = AgentScraper(headless=True)
```

### 5.3 HTML 片段截取

`Extractor` 和 `RuleDiscoverer` 都有 50KB 的 HTML 截取上限（`MAX_HTML_SIZE`）。如果目标页面主内容超过此限制，可调大此值，但要注意 LLM 上下文窗口限制。
