"""提示词注册中心：集中管理系统中所有 LLM 提示词模板"""

# --- TaskParser ---
PARSE_PROMPT = """\
你是一个任务解析器。将用户的自然语言爬取指令解析为结构化JSON。

输出格式（严格JSON，不要多余文字）：
{{
  "mode": "extract|capture",
  "navigation_steps": [
    {{"action": "goto|click|wait|input", "target": "URL或按钮文本或选择器", "value": "input时填入的值，其他为空字符串", "description": "原始描述", "extract_point": false}}
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

extract_point（布尔值，默认 false）：
当任务需要**分别访问多个不同页面**提取相同字段时（如"分别搜索 A、B、C 三家公司"），
在每个实体的**最后一步**（通常是 click 进入目标页面）设置 "extract_point": true。
系统会在该步骤完成后缓存当前页面 HTML，最终从所有缓存页面批量提取数据。
- 只有多实体任务才需要设置 extract_point，单目标任务全部保持 false
- 如果用户要"分别"/"依次"/"逐个"访问多个目标并提取相同字段，就是多实体模式
- 示例: 搜索 OpenAI → click 文章(extract_point=true) → 搜索 Apple → click 文章(extract_point=true)

**严格禁止**放入 navigation_steps 的操作（必须归入 traversal_hints）：
- 点击"加载更多"/"Load more"/任何加载按钮 → traversal_hints 加 "load_more"
- 点击"下一页"/"Next"/翻页按钮/翻页/分页 → traversal_hints 加 "next_button"
- 下滑/滚动加载 → traversal_hints 加 "load_more"
- 遍历子页面/文件夹 → traversal_hints 加 "sub_pages"
即使用户用"步骤N"描述这些操作，也**不要**放入 navigation_steps。

对于登录/表单场景，将每次输入和点击拆分为独立步骤，例如：
  {{"action": "input", "target": "邮箱输入框", "value": "user@example.com", "description": "输入邮箱"}}
  {{"action": "input", "target": "密码输入框", "value": "mypassword", "description": "输入密码"}}
  {{"action": "click", "target": "Sign in", "description": "点击登录"}}

traversal_hints 从用户指令中识别遍历意图（数组，可多选）:
- "load_more": 用户提到"加载更多"、"Load more"、"全部加载"、下滑加载、滚动到底部等
- "sub_pages": 用户提到"进入每个分类"、"遍历子页面"、"逐个点击"、"进入每个文件夹"等
- "next_button": 用户提到"下一页"、"Next"、"翻页"、"所有页"、"每一页"、"分页"等
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
  "next_button_selector": "（仅当用户要求 next_button 时）'下一页'/翻页按钮的CSS选择器，否则null",
  "pagination_max": "（仅当用户要求 next_button 时）从HTML中分析出的总页数，否则null",
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
5. **页面无关性原则**: next_button / load_more 的选择器会被系统在每一页反复使用，因此**禁止包含任何会随页面变化的值**（如具体页码、当前URL片段）。选择器必须只依赖元素自身的稳定属性（class、文本内容、aria-label、在容器中的结构位置等）。

只输出JSON。
"""

DISCOVER_RETRY_PROMPT = """\
你是一个网页结构分析专家。上一轮分析**未能找到**以下遍历规则: {missing_modes}

**上一轮失败的选择器（被验证系统拒绝）：**
{failed_feedback}

请分析上述选择器为什么失败，然后生成**完全不同**的选择器。常见失败原因及修正方向：
- "无匹配" → 选择器中使用了HTML中不存在的 class/属性/标签，换用其他锚点（data-*、role、文本内容）
- "嵌套过深" → 简化层级，用更直接的选择器路径
- "匹配过多" → 选择器太宽泛，加更精确的属性约束

额外提示：
- 按钮可能没有特殊 class，需要通过文本内容或 role 属性定位
- 子页面链接可能嵌套在复杂容器中（div > a 而非直接 a 标签）
- 翻页可能用 <nav> 或自定义组件实现
- next_button / load_more 选择器**禁止包含随页面变化的值**（如具体页码），只用元素自身稳定属性定位

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

# --- CSS Engine (shared by Extractor & RuleDiscoverer) ---
CSS_SYSTEM_PROMPT = """\
你是一个专业的网页结构分析专家，专门从 HTML 中提取稳定、可移植的 CSS 选择器。

## CSS 选择器稳定性优先级（从高到低）

1. **[最稳定] data-testid / data-cy / data-test**
   - 示例：`[data-testid="file-row"] a`

2. **[稳定] 其他 data-* 属性**
   - data-id / data-key / data-type / data-name / data-target / data-component
   - 示例：`[data-target="FileList"] li a`

3. **[稳定] aria-label / role**
   - 示例：`nav[aria-label="breadcrumb"] a`

4. **[较稳定] id 属性（非动态生成）**
   - 排除动态 id（含长数字、哈希、:r1: 等）
   - 示例：`#product-list .item a`

5. **[一般] 语义 class（非工具类）**
   - 排除 Tailwind / Bootstrap 工具类（flex/grid/p-4/mt-8/rounded 等）
   - 排除状态类（active/current/selected）和哈希类
   - 示例：`.product-card .product-name`

6. **[最不稳定] 纯结构路径**
   - 仅在完全没有任何有效属性时使用
   - 示例：`main > section:nth-child(2) > ul > li > a`

## 思考步骤
1. 扫描 HTML 找 data-testid/data-cy/data-target/aria-label → 优先锚点
2. 找稳定 id（排除动态生成）
3. 找语义 class（排除工具类）
4. 组合：锚点 + 最短路径到目标节点

## 禁止事项
- 禁止使用 Tailwind 工具类
- 禁止使用动态 id
- 禁止输出 XPath
- 禁止嵌套超过 5 层
- JSON 之外不要输出任何解释文字
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

