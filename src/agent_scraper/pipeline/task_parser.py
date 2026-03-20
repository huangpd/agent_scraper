"""LLM 解析自然语言指令 → 结构化 ParsedTask"""

import json
import re
import logging
from collections import defaultdict

from agent_scraper.core.llm import LLMService
from agent_scraper.core.models import ExtractionGoal, NavigationStep, ParsedTask
from agent_scraper.pipeline.prompts import PARSE_PROMPT

logger = logging.getLogger(__name__)

class TaskParser:
    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service or LLMService()

    async def parse(self, instruction: str) -> ParsedTask:
        # 先提取用户提供的样本数据
        samples = self._extract_samples(instruction)

        prompt = PARSE_PROMPT.format(instruction=instruction)
        
        # 这里的 call 已经自动包含了 Logging 和 Trace ID
        content = await self.llm_service.call(prompt, caller="TaskParser")

        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        data = json.loads(content)
        
        # 结构化结果打印 (使用统一 Trace ID)
        from agent_scraper.core.trace import get_trace_id
        logger.info("[TaskParser][#%s] Parsed JSON: %s", get_trace_id(), json.dumps(data, indent=2, ensure_ascii=False)[:300])

        steps = [NavigationStep(**s) for s in data["navigation_steps"]]
        goal_data = data["extraction_goal"]
        
        hints = self._ensure_traversal_hints(goal_data.get("traversal_hints", []), instruction)
        mode = self._ensure_mode(data.get("mode", "extract"), instruction)
        max_pages = self._ensure_max_pages(goal_data.get("max_pages"), instruction)

        goal = ExtractionGoal(
            fields=goal_data["fields"],
            output_format=goal_data.get("output_format", "json"),
            url_pattern=goal_data.get("url_pattern"),
            samples=samples if samples else None,
            traversal_hints=hints,
            max_pages=max_pages,
        )

        return ParsedTask(
            navigation_steps=steps,
            extraction_goal=goal,
            raw_instruction=instruction,
            mode=mode,
        )

    @staticmethod
    def _ensure_traversal_hints(hints: list[str], instruction: str) -> list[str]:
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
    def _ensure_mode(mode: str, instruction: str) -> str:
        if mode == "capture": return mode
        text = instruction.lower()
        capture_keywords = [
            "复制url", "copy_url", "copy url", "获取链接", "获取下载链接",
            "捕获", "capture", "保存链接", "记录链接", "抓取链接",
            "获取当前url", "获取当前页面url",
        ]
        if any(kw in text for kw in capture_keywords): return "capture"
        return mode

    @staticmethod
    def _ensure_max_pages(max_pages: int | None, instruction: str) -> int | None:
        if max_pages: return max_pages
        patterns = [
            r'第\s*(\d+)\s*页.*?停',
            r'前\s*(\d+)\s*页',
            r'最多\s*(\d+)\s*页',
            r'翻\s*(\d+)\s*页',
            r'(\d+)\s*页.*?(?:为止|即可|就行|够了|停止)',
        ]
        for pattern in patterns:
            m = re.search(pattern, instruction)
            if m: return int(m.group(1))
        return None

    @staticmethod
    def _extract_samples(instruction: str) -> dict[str, list[str]] | None:
        json_objects = []
        for line in instruction.strip().splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict): json_objects.append(obj)
                except json.JSONDecodeError: continue
        if not json_objects:
            for m in re.finditer(r'\{[^{}]+\}', instruction):
                try:
                    obj = json.loads(m.group())
                    if isinstance(obj, dict) and len(obj) >= 2: json_objects.append(obj)
                except json.JSONDecodeError: continue
        if not json_objects: return None
        samples = defaultdict(list)
        for obj in json_objects:
            for key, value in obj.items(): samples[key].append(str(value))
        return dict(samples)
