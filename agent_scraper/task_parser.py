"""LLM 解析自然语言指令 → 结构化 ParsedTask"""

import json
import os
import re
from collections import defaultdict

from openai import AsyncOpenAI

from agent_scraper.models import ExtractionGoal, NavigationStep, ParsedTask

PARSE_PROMPT = """\
你是一个任务解析器。将用户的自然语言爬取指令解析为结构化JSON。

输出格式（严格JSON，不要多余文字）：
{{
  "navigation_steps": [
    {{"action": "goto|click|wait", "target": "URL或按钮文本或选择器", "description": "原始描述"}}
  ],
  "extraction_goal": {{
    "fields": {{"字段名": "字段描述", ...}},
    "output_format": "json|csv",
    "url_pattern": "可选的URL构造模板，用{{字段名}}作为占位符，没有则为null",
    "traversal_hints": ["用户要求的遍历模式列表"]
  }}
}}

navigation_steps 的 action 只包含需要AI理解的操作:
- goto: 打开URL
- click: 点击某个元素
- wait: 等待页面加载

traversal_hints 从用户指令中识别遍历意图（数组，可多选）:
- "load_more": 用户提到"加载更多"、"Load more"、"全部加载"等
- "sub_pages": 用户提到"进入每个文件夹"、"遍历子页面"、"逐个点击"等
- "pagination": 用户提到"翻页"、"所有页"、"每一页"等
- "next_button": 用户提到"下一页"等
- 如果用户没有提到任何遍历需求，返回空数组 []

分析规则：
1. navigation_steps 只包含到达目标页面的步骤（打开URL、点击标签等）
2. "加载更多"、"翻页"、"进入文件夹"这些不算导航步骤，归入 traversal_hints
3. 识别提取目标字段
4. 忽略"样本数据"/"示例"部分
5. 默认 output_format 为 json

用户指令：
{instruction}
"""


class TaskParser:
    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.model = os.getenv("MODEL_NAME", "gpt-4o")

    async def parse(self, instruction: str) -> ParsedTask:
        # 先提取用户提供的样本数据
        samples = self._extract_samples(instruction)

        prompt = PARSE_PROMPT.format(instruction=instruction)
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content.strip()

        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        data = json.loads(content)

        steps = [NavigationStep(**s) for s in data["navigation_steps"]]
        goal_data = data["extraction_goal"]
        # LLM 识别遍历模式 + 关键词兜底（防止 LLM 遗漏）
        hints = goal_data.get("traversal_hints", [])
        hints = self._ensure_traversal_hints(hints, instruction)

        goal = ExtractionGoal(
            fields=goal_data["fields"],
            output_format=goal_data.get("output_format", "json"),
            url_pattern=goal_data.get("url_pattern"),
            samples=samples if samples else None,
            traversal_hints=hints,
        )

        return ParsedTask(
            navigation_steps=steps,
            extraction_goal=goal,
            raw_instruction=instruction,
        )

    @staticmethod
    def _ensure_traversal_hints(hints: list[str], instruction: str) -> list[str]:
        """关键词兜底：防止 LLM 遗漏用户明确要求的遍历模式"""
        text = instruction.lower()
        checks = {
            "load_more": ["load more", "加载更多", "全部加载", "加载全部"],
            "sub_pages": ["进入每个文件夹", "遍历子页面", "每个文件夹", "进入文件夹", "子文件夹"],
            "pagination": ["翻页", "所有页", "每一页", "分页"],
            "next_button": ["下一页", "next page"],
        }
        for hint_type, keywords in checks.items():
            if hint_type not in hints and any(kw in text for kw in keywords):
                hints.append(hint_type)
        return hints

    @staticmethod
    def _extract_samples(instruction: str) -> dict[str, list[str]] | None:
        """从指令中提取用户提供的 JSONL 样本数据"""
        json_lines = []
        for line in instruction.strip().splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        json_lines.append(obj)
                except json.JSONDecodeError:
                    continue

        if not json_lines:
            return None

        samples = defaultdict(list)
        for obj in json_lines:
            for key, value in obj.items():
                samples[key].append(str(value))

        return dict(samples)
