# Agent Scraper 使用手册

## 1. 快速开始

### 1.1 安装

```bash
pip install -e ".[dev]"
playwright install chromium
```

### 1.2 配置

创建 `.env` 文件：

```env
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
```

### 1.3 第一次运行

编辑 `run.py` 中的 `instruction` 变量，然后：

```bash
python run.py
```

## 2. 编写指令

Agent Scraper 的核心是**自然语言指令**。一条好的指令包含三个要素：

```
去哪里（导航） + 怎么翻页（遍历） + 取什么（提取）
```

### 2.1 基本结构

```
步骤1: 打开网址 <URL>
步骤2: 点击 "<按钮/标签文本>"
步骤3: <遍历要求>
步骤4: 提取 <字段描述>

样本数据:
{"字段1": "示例值1", "字段2": "示例值2"}
```

### 2.2 导航指令

告诉系统如何到达目标页面：

| 动作 | 写法示例 |
|------|---------|
| 打开网址 | `打开网址 https://example.com` |
| 点击按钮/标签 | `点击 "Files and versions" 标签页` |
| 等待加载 | `等待页面加载完成` |

可以组合多个步骤：

```
步骤1: 打开网址 https://huggingface.co/meta-llama/Llama-3-8B
步骤2: 点击 "Files and versions" 标签页
```

### 2.3 遍历指令

告诉系统如何获取所有数据（不只是第一页）：

| 遍历模式 | 触发关键词 | 适用场景 |
|---------|-----------|---------|
| 加载更多 | `加载更多`、`Load more`、`全部加载` | 页面有"加载更多"按钮 |
| 子页面遍历 | `遍历子页面`、`进入每个文件夹`、`子文件夹` | 页面有文件夹/分类需要逐个进入 |
| URL 分页 | `翻页`、`所有页`、`每一页` | 页面有页码导航 |
| 下一页按钮 | `下一页`、`next page` | 页面有"下一页"按钮 |

遍历模式可以叠加：

```
步骤3: 点击"加载更多"直到全部加载，然后遍历所有子文件夹
```

**如果不需要遍历**（只提取当前页），不写遍历指令即可。

### 2.4 提取目标

告诉系统要提取哪些字段：

```
提取文件的文件名和下载URL，用json格式
```

### 2.5 样本数据（强烈推荐）

提供 1-2 条样本数据可以**大幅提高提取准确率**，尤其是 URL 字段：

```
样本数据:
{"file_name": ".gitattributes", "download_url": "/repo/blob/main/.gitattributes"}
{"file_name": "README.md", "download_url": "/repo/blob/main/README.md"}
```

样本的作用：
- 让 AutoScraper 可以直接基于样本训练规则，跳过 LLM 采样
- 帮助 Formatter 自动推断 URL 构造规则（如 `download_url = prefix + file_name`）
- 明确字段名称，避免 LLM 理解偏差

## 3. 指令示例

### 3.1 提取 HuggingFace 模型文件列表

```
步骤1: 打开网址 https://huggingface.co/baichuan-inc/Baichuan-M3-235B-FP8
步骤2: 找到并点击 "Files and versions" 标签页
步骤3: 下滑页面到最底部，如果页面有 "Load more files" 请点击，直到按钮消失
步骤4: 遍历所有子文件夹
步骤5: 提取文件的文件名和下载URL，用json格式

样本数据:
{"file_name": ".gitattributes", "download_url": "/baichuan-inc/Baichuan-M3-235B-FP8/blob/main/.gitattributes"}
```

**触发的能力**：导航 → 点击标签 → 加载更多 + 子页面遍历 → JSON 提取

### 3.2 提取商品列表（带翻页）

```
步骤1: 打开网址 https://example-shop.com/products
步骤2: 提取每一页的商品名称、价格和链接

样本数据:
{"name": "无线鼠标", "price": "¥89.00", "link": "/products/wireless-mouse"}
```

**触发的能力**：导航 → 自动检测翻页 → 多页提取

### 3.3 单页简单提取

```
步骤1: 打开网址 https://github.com/trending
步骤2: 提取仓库名称、描述和星标数
```

**触发的能力**：导航 → 单页提取（无遍历）

### 3.4 论坛帖子列表（下一页按钮）

```
步骤1: 打开网址 https://forum.example.com/latest
步骤2: 提取所有页的帖子标题、作者和发布时间，点击下一页直到没有更多

样本数据:
{"title": "如何学习Python", "author": "张三", "date": "2024-01-15"}
```

**触发的能力**：导航 → 下一页按钮遍历 → 多页提取

## 4. 使用方式

### 4.1 Python API

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()

from agent_scraper import AgentScraper
from agent_scraper.extraction.formatter import Formatter

async def main():
    scraper = AgentScraper(headless=False)  # headless=True 隐藏浏览器
    result = await scraper.run("你的指令...")

    # 输出 JSON
    print(Formatter.to_json(result))

    # 输出 CSV
    print(Formatter.to_csv(result))

    # 直接访问数据
    for item in result.data:
        print(item)

asyncio.run(main())
```

### 4.2 Web UI

```bash
# 开发模式（前后端热重载）
python run_server.py

# 生产模式
python run_server.py --prod
```

访问 `http://localhost:5173`（开发模式）或 `http://localhost:8000`（生产模式）。

在 Web UI 中：
1. 在文本框中输入自然语言指令
2. （可选）上传参考截图：点击上传按钮或 Ctrl+V 粘贴截图，用红框标注目标元素
3. 点击"开始"按钮
4. 实时查看执行日志
5. 任务完成后查看/下载结果

### 4.4 图片参考功能

当目标页面的元素不容易用文字描述（如特定位置的按钮、图标等），可以：

1. 截取目标页面的截图
2. 用红色方框标注要操作的目标元素
3. 在 Web UI 中上传截图（支持拖拽、点击上传或 Ctrl+V 粘贴）
4. 系统会将截图传递给 AI Agent，帮助其精准定位元素

截图会被自动转为 base64 编码，通过多模态 LLM 进行视觉理解。

### 4.3 REST API

```bash
# 创建任务
curl -X POST http://localhost:8000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"instruction": "你的指令...", "headless": true, "images": []}'
# 返回: {"task_id": "abc12345"}
# images 可传入 base64 data URL 数组（可选）

# 查询状态
curl http://localhost:8000/api/tasks/abc12345

# 取消任务
curl -X POST http://localhost:8000/api/tasks/abc12345/cancel
```

WebSocket 实时日志：

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/abc12345");
ws.onmessage = (e) => {
    const event = JSON.parse(e.data);
    // event.type: "log" | "progress" | "result" | "error" | "done"
    console.log(event);
};
```

## 5. 提高准确率的技巧

### 5.1 提供样本数据

这是最有效的方式。去目标页面手动复制 1-2 条数据作为样本：

```
样本数据:
{"file_name": "config.json", "download_url": "/repo/blob/main/config.json"}
{"file_name": "model.bin", "download_url": "/repo/blob/main/model.bin"}
```

### 5.2 明确字段描述

模糊：`提取文件信息`

清晰：`提取文件的文件名和下载链接URL`

### 5.3 指定遍历方式

模糊：`获取所有数据`

清晰：`点击"加载更多"直到按钮消失，然后进入每个子文件夹提取`

### 5.4 分步描述导航

模糊：`去 HuggingFace 找 Llama 3 的文件`

清晰：
```
步骤1: 打开网址 https://huggingface.co/meta-llama/Llama-3-8B
步骤2: 点击 "Files and versions" 标签页
```

## 6. 常见问题

### Q: 浏览器没有打开？

确保已安装 Playwright 浏览器：`playwright install chromium`

### Q: LLM API 报错？

检查 `.env` 中的 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 是否正确。可以用以下命令测试：

```bash
curl $OPENAI_BASE_URL/models -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Q: 提取结果为空？

1. 检查页面是否需要登录（当前不支持需要登录的页面）
2. 尝试添加样本数据
3. 设置 `headless=False` 观察浏览器实际行为
4. 查看终端日志中 `[Extractor]` 的提取策略选择

### Q: 遍历没有生效？

检查指令中是否包含遍历关键词（如"加载更多"、"遍历子文件夹"、"翻页"）。系统通过关键词识别遍历意图。

### Q: Token 消耗过高？

- LLM 只在第一页调用（解析指令 + 发现规则 + 首次提取）
- 第 2+ 页使用缓存的 CSS 选择器，零 LLM 调用
- 如需进一步降低消耗，可将 `MODEL_NAME` 换成更便宜的模型（如 `gpt-4o-mini`）
