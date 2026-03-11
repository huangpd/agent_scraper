"""核心提取模块：LLM 采样 + AutoScraper 批量提取 + 三级降级"""

import json
import os
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from openai import AsyncOpenAI

from agent_scraper.models import ExtractionGoal
from autoscraper import AutoScraper

# ── LLM 采样 Prompt ──────────────────────────────────────

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

# ── CSS Selector 降级 Prompt ─────────────────────────────

CSS_SELECTOR_PROMPT = """\
你是一个前端专家。分析下面的 HTML，为每个字段生成 CSS 选择器来提取数据。

要提取的字段：
{fields_desc}

HTML 片段：
```html
{html_snippet}
```

要求：
1. 每个选择器应能匹配页面中 **所有** 同类元素（不是只匹配一个）
2. 对于文本内容，选择器指向包含文本的元素
3. 对于 URL，选择器指向包含 href 的 <a> 标签
4. 说明是提取 text 还是某个属性（如 href）

输出格式（严格JSON）：
{{
  "字段名1": {{"selector": "CSS选择器", "attr": "text|href|src|其他属性"}},
  "字段名2": {{"selector": "CSS选择器", "attr": "text|href|src|其他属性"}}
}}
"""

MAX_HTML_SIZE = 50 * 1024  # 50KB


class Extractor:
    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.model = os.getenv("MODEL_NAME", "gpt-4o")
        self._trained_scraper: AutoScraper | None = None  # 复用已训练规则

    async def _llm_call(self, prompt: str) -> str:
        """统一的 LLM 调用"""
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    async def extract(self, html: str, goal: ExtractionGoal) -> dict[str, list]:
        """
        提取策略（要求所有目标字段都必须提取到）：
          1. 复用已训练的 AutoScraper 规则（第 2+ 页）
          2. 用户提供样本 → 训练 AutoScraper
          3. LLM 采样 → 训练 AutoScraper
          4. CSS Selector 降级（LLM 生成选择器）
        """
        expected_fields = set(goal.fields.keys())

        # 优先复用已训练的规则（多页提取时第 2+ 页走这里）
        if self._trained_scraper:
            result = self._apply_trained_scraper(html)
            if result and self._validate_result(result, expected_fields):
                print(f"[Extractor] 复用已训练规则成功: { {k: len(v) for k, v in result.items()} }")
                return result
            print("[Extractor] 复用规则失败，重新提取...")

        # 准备 wanted_dict
        if goal.samples:
            print(f"[Extractor] 使用用户提供的样本: { {k: v[:2] for k, v in goal.samples.items()} }")
            wanted_dict = self._normalize_url_samples(goal.samples)
        else:
            print("[Extractor] 阶段A: LLM 采样生成样本...")
            wanted_dict = await self._llm_sample(html, goal)
            if not wanted_dict or not any(wanted_dict.values()):
                print("[Extractor] LLM 采样失败，直接跳到 CSS Selector 降级")
                return await self._css_selector_fallback(html, goal)
            print(f"[Extractor] LLM 采样结果: { {k: v[:2] for k, v in wanted_dict.items()} }")

        # AutoScraper 批量提取
        print("[Extractor] AutoScraper 批量提取...")
        result = self._autoscraper_extract(html, wanted_dict)

        if result and self._validate_result(result, expected_fields):
            print(f"[Extractor] AutoScraper 提取成功: { {k: len(v) for k, v in result.items()} }")
            return result

        # CSS Selector 降级
        print("[Extractor] AutoScraper 失败（缺少字段），降级到 CSS Selector...")
        return await self._css_selector_fallback(html, goal)

    def _get_main_content_snippet(self, html: str) -> str:
        """提取主内容区域的 HTML 片段，限制大小"""
        soup = BeautifulSoup(html, "lxml")

        # 移除 script/style/nav/header/footer 等噪音
        for tag in soup.find_all(["script", "style", "nav", "header", "footer", "noscript"]):
            tag.decompose()

        # 尝试找到主内容区域
        main = (
            soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup.find("article")
            or soup.find("div", class_=re.compile(r"content|main|container", re.I))
            or soup.body
        )

        if not main:
            main = soup

        content = str(main)
        if len(content) > MAX_HTML_SIZE:
            content = content[:MAX_HTML_SIZE]
        return content

    async def _llm_sample(self, html: str, goal: ExtractionGoal) -> dict[str, list]:
        """LLM 从 HTML 中提取 2-3 个样本作为 AutoScraper 的训练数据"""
        snippet = self._get_main_content_snippet(html)
        fields_desc = "\n".join(f"- {k}: {v}" for k, v in goal.fields.items())

        prompt = SAMPLE_PROMPT.format(
            fields_desc=fields_desc,
            html_snippet=snippet,
        )

        try:
            content = await self._llm_call(prompt)

            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            wanted_dict = json.loads(content)

            # 确保所有值都是列表且非空
            cleaned = {}
            for key, values in wanted_dict.items():
                if isinstance(values, list) and values:
                    cleaned[key] = [str(v) for v in values if v]
            return cleaned

        except Exception as e:
            print(f"[Extractor] LLM 采样出错: {e}")
            return {}

    def _autoscraper_extract(self, html: str, wanted_dict: dict[str, list]) -> dict[str, list]:
        """用 AutoScraper 进行批量提取，训练成功后保存 scraper 供后续复用"""
        scraper = AutoScraper()

        try:
            build_result = scraper.build(
                html=html,
                wanted_dict=wanted_dict,
            )

            if isinstance(build_result, dict) and build_result:
                self._trained_scraper = scraper  # 保存供后续页面复用
                return build_result

            result = scraper.get_result_similar(
                html=html,
                group_by_alias=True,
            )

            if isinstance(result, dict) and result:
                self._trained_scraper = scraper  # 保存供后续页面复用
                return result

        except Exception as e:
            print(f"[Extractor] AutoScraper 出错: {e}")

        return {}

    def _apply_trained_scraper(self, html: str) -> dict[str, list]:
        """用已训练的 scraper 提取数据"""
        try:
            result = self._trained_scraper.get_result_similar(
                html=html,
                group_by_alias=True,
            )
            if isinstance(result, dict):
                return result
        except Exception as e:
            print(f"[Extractor] 复用规则出错: {e}")
        return {}

    async def _css_selector_fallback(self, html: str, goal: ExtractionGoal) -> dict[str, list]:
        """CSS Selector 降级：LLM 生成选择器，直接从 HTML 提取"""
        print("[Extractor] 降级: LLM 生成 CSS Selector...")
        snippet = self._get_main_content_snippet(html)
        fields_desc = "\n".join(f"- {k}: {v}" for k, v in goal.fields.items())

        prompt = CSS_SELECTOR_PROMPT.format(
            fields_desc=fields_desc,
            html_snippet=snippet,
        )

        try:
            content = await self._llm_call(prompt)

            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            selectors = json.loads(content)
        except Exception as e:
            print(f"[Extractor] CSS Selector 降级失败: {e}")
            return {}

        # 用完整 HTML 执行 CSS 选择器提取
        soup = BeautifulSoup(html, "lxml")
        result = {}

        for field_name, sel_info in selectors.items():
            selector = sel_info.get("selector", "")
            attr = sel_info.get("attr", "text")

            elements = soup.select(selector)
            values = []
            for el in elements:
                if attr == "text":
                    val = el.get_text(strip=True)
                else:
                    val = el.get(attr, "")
                if val:
                    values.append(str(val))

            result[field_name] = values

        if result:
            print(f"[Extractor] CSS Selector 提取结果: { {k: len(v) for k, v in result.items()} }")

        return result

    @staticmethod
    def _normalize_url_samples(samples: dict[str, list[str]]) -> dict[str, list[str]]:
        """对 URL 样本同时提供完整URL和路径版本，帮助 AutoScraper 匹配 HTML 中的相对路径"""
        result = {}
        for key, values in samples.items():
            expanded = list(values)
            for v in values:
                if v.startswith("http"):
                    path = urlparse(v).path
                    if path and path not in expanded:
                        expanded.append(path)
            result[key] = expanded
        return result

    @staticmethod
    def _validate_result(result: dict[str, list], expected_fields: set[str] | None = None) -> bool:
        """验证提取结果是否合理：所有期望字段都必须存在且有数据"""
        if not result:
            return False

        # 检查所有期望字段是否都已提取到
        if expected_fields:
            missing = expected_fields - set(result.keys())
            if missing:
                print(f"[Extractor] 验证失败: 缺少字段 {missing}")
                return False

        lengths = [len(v) for v in result.values()]

        # 所有字段都必须有数据
        if any(l == 0 for l in lengths):
            return False

        # 各字段长度应该相等（或至少差距不大）
        if len(lengths) > 1:
            max_len = max(lengths)
            min_len = min(lengths)
            if min_len == 0 or max_len / min_len > 2:
                return False

        return True
