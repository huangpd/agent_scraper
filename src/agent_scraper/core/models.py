"""数据模型定义"""

from pydantic import BaseModel


class NavigationStep(BaseModel):
    """单个导航步骤"""
    action: str          # "goto" | "click" | "wait" | "input" | "scroll"
    target: str = ""     # URL / 按钮文本 / CSS选择器（wait 时可为空）
    value: str = ""      # input 动作的输入值（账号、密码等）
    description: str = ""  # 原始自然语言描述


class ExtractionGoal(BaseModel):
    """提取目标"""
    fields: dict[str, str]        # {"filename": "文件名", "url": "下载链接"}
    output_format: str = "json"   # "json" | "csv"
    url_pattern: str | None = None  # 可选URL构造模板
    samples: dict[str, list[str]] | None = None  # 用户提供的样本数据
    # 用户指定的遍历模式，空=只提取当前页
    # 可选值: "load_more" | "pagination" | "sub_pages" | "next_button"
    traversal_hints: list[str] = []


class PageRules(BaseModel):
    """LLM 发现的页面遍历规则 — AI 只产出规则，代码执行"""
    # 四种翻页模式，可叠加
    load_more_selector: str | None = None     # "button:has-text('Load more')"
    next_button_selector: str | None = None   # "a.pagination-next"
    pagination_url: str | None = None         # "https://xxx/page/{n}"
    pagination_max: int | None = None         # 最大页数
    sub_page_selector: str | None = None      # "a.folder-link" 子页面入口
    sub_page_url_attr: str = "href"           # 从哪个属性取URL
    sub_page_url_filter: str | None = None    # URL 过滤关键词，如 "/tree/" 只保留含此的URL
    sub_page_recursive: bool = False          # 是否递归进入子页面的子页面


class ParsedTask(BaseModel):
    """解析后的完整任务"""
    navigation_steps: list[NavigationStep]
    extraction_goal: ExtractionGoal
    raw_instruction: str
    mode: str = "extract"  # "extract": 批量提取 | "capture": 浏览器直接捕获值


class ScrapedResult(BaseModel):
    """最终输出"""
    data: list[dict]
    total_count: int
    source_url: str
