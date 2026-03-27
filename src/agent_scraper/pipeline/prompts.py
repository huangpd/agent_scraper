"""提示词注册中心：集中管理系统中所有 LLM 提示词模板"""

# --- TaskParser ---
PARSE_PROMPT = """\
你是一个任务解析器。将用户的自然语言爬取指令解析为结构化JSON。

输出格式（严格JSON，不要多余文字）：
{{
  "mode": "extract|capture",
  "navigation_steps": [
    {{"action": "goto|click|wait|input", "target": "URL或按钮文本或选择器", "value": "input时填入的值，其他为空字符串", "description": "原始描述"}}
  ],
  "extraction_goal": {{
    "fields": {{"字段名": "字段描述", ...}},
    "output_format": "json|csv",
    "url_pattern": "可选的URL构造模板，用{{字段名}}作为占位符，没有则为null",
    "traversal_hints": ["用户要求的遍历模式列表"],
    "max_pages": null,
    "next_button_text": "用户提到的翻页按钮文本（如'下一页'、'Next'），没有则为null"
  }}
}}

mode 说明（二选一）：
- "extract": 从页面HTML中**批量提取**结构化数据（列表、表格等多条记录）。适用于：商品列表、搜索结果、文章目录、数据表格等
- "capture": 通过浏览器操作**直接捕获**少量特定值（1-3个），不需要HTML解析。适用于：获取下载链接、复制当前URL、抓取某个特定元素的值等
  - 当用户说"复制URL"、"获取链接"、"保存/记录某个值"、"capture"等，使用 capture
  - 当任务核心是浏览器操作（登录→点击→获取结果），且不需要批量提取时，使用 capture

navigation_steps 与 traversal_hints 的核心区别：
- navigation_steps = **一次性操作**：到达目标页面的步骤（只执行一次）
- traversal_hints = **重复性操作**：在目标页面上反复执行以获取更多数据（系统自动循环处理）

navigation_steps 的 action:
- goto: 打开URL
- click: 点击某个元素（如切换标签页、展开菜单）
- wait: 等待页面加载
- input: 在输入框中填写内容（target 为输入框描述，value 为要填入的值）

**严格禁止**放入 navigation_steps 的操作（必须归入 traversal_hints）：
- 点击"加载更多"/"Load more"/任何加载按钮 → traversal_hints 加 "load_more"
- 点击"下一页"/"Next"/翻页按钮 → traversal_hints 加 "next_button"
- 下滑/滚动加载 → traversal_hints 加 "load_more"
- 遍历子页面/文件夹 → traversal_hints 加 "sub_pages"
- 翻页/分页 → traversal_hints 加 "pagination"
即使用户用"步骤N"描述这些操作，也**不要**放入 navigation_steps。

对于登录/表单场景，将每次输入和点击拆分为独立步骤，例如：
  {{"action": "input", "target": "邮箱输入框", "value": "user@example.com", "description": "输入邮箱"}}
  {{"action": "input", "target": "密码输入框", "value": "mypassword", "description": "输入密码"}}
  {{"action": "click", "target": "Sign in", "description": "点击登录"}}

traversal_hints 从用户指令中识别遍历意图（数组，可多选）:
- "load_more": 用户提到"加载更多"、"Load more"、"全部加载"、下滑加载、滚动到底部等
- "sub_pages": 用户提到"进入每个分类"、"遍历子页面"、"逐个点击"、"进入每个文件夹"等
- "pagination": 用户提到"翻页"、"所有页"、"每一页"等
- "next_button": 用户提到"下一页"、"Next"等
- 如果用户没有提到任何遍历需求，返回空数组 []

load_more_text: 用户提到的加载按钮的**原始文本**（如"Load more files"、"查看更多"、"more>>"），没有则为null
next_button_text: 用户提到的翻页按钮的**原始文本**（如"下一页"、"Next"），没有则为null

max_pages: 用户指定的最大页数限制（整数），没有则为 null：
- "翻到第3页停止" → 3
- "只取前5页" → 5
- "翻页，最多10页" → 10
- 没有提到页数限制 → null

分析规则：
1. navigation_steps 只包含到达目标页面的步骤（打开URL、点击标签等），执行一次即到位
2. 所有需要"反复执行"或"遍历"的操作，无论用户怎么描述，都归入 traversal_hints，**绝对不要**放入 navigation_steps
3. 用户提到的按钮文本（如"Load more files"、"下一页"）放入 load_more_text 或 next_button_text
4. 识别提取目标字段
5. 忽略"样本数据"/"示例"部分
6. 默认 output_format 为 json

用户指令：
{instruction}
"""

# --- Evaluator ---
REPLAN_PROMPT = """\
你是一个网页采集专家。当前提取任务遇到了质量问题，需要你建议重试策略。

任务目标字段: {fields}
当前提取结果: {extracted}
发现的问题: {issues}
已执行步骤:
{history}
已重试次数: {retry_count}/{max_retries}

可选策略（只输出策略名称，不要多余文字）:
- clear_css_cache   清除缓存的 CSS 选择器和 AutoScraper 模型，让 LLM 重新生成
- retry_navigate    重新导航并从头提取（页面可能已变化）
- skip              接受当前结果（无法改善）
"""

# --- RuleDiscoverer ---
DISCOVER_PROMPT = """\
你是一个网页结构分析专家。分析下面的 HTML 片段，**只**找出用户要求的遍历规则。

页面当前URL: {current_url}
用户要求的遍历模式: {requested_modes}

HTML 片段:
```html
{html_snippet}
```

根据用户要求的模式，输出对应的 CSS 选择器或 URL 模式（严格JSON，不要多余文字）：

{{
  "load_more_selector": "（仅当用户要求 load_more 时）'加载更多'按钮的CSS选择器，否则null",
  "next_button_selector": "（仅当用户要求 next_button 时）'下一页'按钮的CSS选择器，否则null",
  "pagination_url": "（仅当用户要求 pagination 时）URL模板用{{n}}表示页码，否则null",
  "pagination_max": "（仅当用户要求 pagination 时）总页数，否则null",
  "sub_page_selector": "（仅当用户要求 sub_pages 时）子页面/文件夹链接的CSS选择器，否则null",
  "sub_page_url_attr": "子页面链接的URL属性，通常是href",
  "sub_page_url_filter": "URL中必须包含的关键词，用于过滤非目标链接（如分类页含'/category/'、目录页含'/dir/'等），没有则null",
  "sub_page_recursive": false
}}

CSS选择器要求:
1. 尽量精确，能唯一定位到目标元素
2. 用户没要求的模式，对应字段必须返回null
3. 对于 load_more，优先用精确选择器；如果按钮没有 class/id，可以用文本匹配描述
4. 对于 sub_pages，选择器必须**只匹配目标子页面链接**，不要匹配无关链接。
   子页面通常有独特的 class、图标、或 URL 路径模式（如含 /category/、/list/、/tree/ 等路径段）。
   如果无法通过选择器区分，在 sub_page_url_filter 中填写 URL 关键词过滤规则。

只输出JSON。
"""

DISCOVER_RETRY_PROMPT = """\
你是一个网页结构分析专家。上一轮分析**未能找到**以下遍历规则: {missing_modes}
但用户明确表示页面包含这些元素。请更仔细地分析 HTML，尤其注意：
- 按钮可能没有特殊 class，需要通过文本内容或 role 属性定位
- 子页面链接可能嵌套在复杂容器中（div > a 而非直接 a 标签）
- 翻页可能用 <nav> 或自定义组件实现

页面当前URL: {current_url}
需要查找的遍历模式: {requested_modes}

HTML 片段:
```html
{html_snippet}
```

输出格式同上一轮（严格JSON，不要多余文字）：

{{
  "load_more_selector": "CSS选择器或null",
  "next_button_selector": "CSS选择器或null",
  "pagination_url": "URL模板或null",
  "pagination_max": null,
  "sub_page_selector": "CSS选择器或null",
  "sub_page_url_attr": "href",
  "sub_page_url_filter": "过滤关键词或null",
  "sub_page_recursive": false
}}

重要：如果确实在 HTML 中找不到对应元素，返回 null，不要编造不存在的选择器。
只输出JSON。
"""

# --- Extractor ---
SAMPLE_PROMPT = """\
你是一个数据提取专家。从下面的 HTML 片段中，为每个字段提取 2-3 个真实样本值。

要提取的字段：
{fields_desc}

HTML 片段（截取自页面主内容区域）：
```html
{html_snippet}
```

要求：
1. 每个字段提取 2-3 个 **真实存在于 HTML 中的** 样本值
2. 样本必须是 HTML 中的原始文本或属性值，不能自己编造
3. 对于 URL 类字段，提取 href 属性的完整值（包含相对路径）
4. 选择页面中不同位置的样本，以确保规则泛化

输出格式（严格JSON，不要多余文字）：
{{
  "字段名1": ["样本1", "样本2"],
  "字段名2": ["样本1", "样本2"]
}}
"""

# --- VisionSampleTool ---
VISION_SAMPLE_PROMPT = """\
你是一个视觉数据标注专家。用户在截图中标注了想要提取的数据区域。

只需识别以下**可见文本**字段（跳过 URL/链接类字段，系统会自动处理）：
{fields_desc}

要求：
1. 只输出截图中**肉眼可见的文字**，不要猜测或编造
2. 每个字段提取 2-3 个不同的样本值
3. 聚焦截图中**重复出现的列表/表格结构**中的数据，忽略页头导航、侧边栏等非数据区域

输出格式（严格JSON，不要多余文字）：
{{
  "字段名1": ["可见文本样本1", "可见文本样本2"],
  "字段名2": ["可见文本样本1", "可见文本样本2"]
}}
"""

CSS_SELECTOR_PROMPT = """\
你是一个前端专家。分析下面的 HTML，为每个字段生成 CSS 选择器来提取数据。

要提取的字段：
{fields_desc}

HTML 片段：
```html
{html_snippet}
```

输出格式（严格JSON）：
{{
  "字段名1": {{"selector": "CSS选择器", "attr": "text|href|src|其他属性"}},
  "字段名2": {{"selector": "CSS选择器", "attr": "text|href|src|其他属性"}}
}}
"""
