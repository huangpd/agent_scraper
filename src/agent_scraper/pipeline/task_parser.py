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
        
        raw_steps = [NavigationStep(**s) for s in data["navigation_steps"]]
        goal_data = data["extraction_goal"]

        hints = self._ensure_traversal_hints(goal_data.get("traversal_hints", []), instruction)
        # LLM 有时会把 load_more / 翻页类操作误放进 navigation_steps，
        # 这些应该由 PageIterator 处理，不能让 browser-use 先点掉
        steps, hints, load_more_text, next_button_text = self._strip_traversal_from_steps(raw_steps, hints)
        # 兜底：从用户原始指令的引号中提取按钮文本
        if not load_more_text and "load_more" in hints:
            load_more_text = self._extract_quoted_text(instruction)
        # next_button_text: LLM 解析 > strip 提取 > 指令引号提取
        if not next_button_text:
            next_button_text = goal_data.get("next_button_text")
        if not next_button_text and "next_button" in hints:
            next_button_text = self._extract_next_button_text(instruction)
        mode = self._ensure_mode(data.get("mode", "extract"), instruction)
        max_pages = self._ensure_max_pages(goal_data.get("max_pages"), instruction)

        goal = ExtractionGoal(
            fields=goal_data["fields"],
            output_format=goal_data.get("output_format", "json"),
            url_pattern=goal_data.get("url_pattern"),
            samples=samples if samples else None,
            traversal_hints=hints,
            max_pages=max_pages,
            load_more_text=load_more_text,
            next_button_text=next_button_text,
        )

        return ParsedTask(
            navigation_steps=steps,
            extraction_goal=goal,
            raw_instruction=instruction,
            mode=mode,
        )

    @staticmethod
    def _strip_traversal_from_steps(
        steps: list[NavigationStep], hints: list[str],
    ) -> tuple[list[NavigationStep], list[str], str | None, str | None]:
        """把 LLM 误放进 navigation_steps 的遍历操作剔除，转入 traversal_hints。

        返回 (clean_steps, hints, load_more_text, next_button_text)。
        """
        _LOAD_MORE = re.compile(r"load\s*more|加载更多|show\s*more|全部加载", re.I)
        _NEXT_PAGE = re.compile(r"下一页|下一頁|next\s*page", re.I)
        _SCROLL_LOAD = re.compile(r"下滑.*加载|滚动.*加载|scroll.*load", re.I)

        clean_steps: list[NavigationStep] = []
        load_more_text: str | None = None
        next_button_text: str | None = None
        for step in steps:
            combined = f"{step.target} {step.description}"
            if step.action in ("click", "scroll"):
                target = step.target.strip()
                if _LOAD_MORE.search(combined) or _SCROLL_LOAD.search(combined):
                    if "load_more" not in hints:
                        hints.append("load_more")
                    if target and (_LOAD_MORE.search(target) or _SCROLL_LOAD.search(target)):
                        load_more_text = target
                    logger.info("剔除导航步骤 → traversal_hints[load_more]: %s", step.description)
                    continue
                if _NEXT_PAGE.search(combined):
                    if "next_button" not in hints:
                        hints.append("next_button")
                    if target and _NEXT_PAGE.search(target):
                        next_button_text = target
                    logger.info("剔除导航步骤 → traversal_hints[next_button]: %s", step.description)
                    continue
            clean_steps.append(step)
        return clean_steps, hints, load_more_text, next_button_text

    @staticmethod
    def _extract_quoted_text(instruction: str) -> str | None:
        """从指令中提取引号包裹的文本，优先返回像 load_more 按钮的文本"""
        _LOAD_MORE = re.compile(r"load\s*more|加载更多|show\s*more|全部加载", re.I)
        matches = re.findall(r'["\u201c\u201d\'](.*?)["\u201c\u201d\']', instruction)
        # 优先返回匹配 load_more 模式的
        for m in matches:
            if _LOAD_MORE.search(m):
                return m.strip()
        # 都不匹配则返回第一个
        return matches[0].strip() if matches else None

    @staticmethod
    def _extract_next_button_text(instruction: str) -> str | None:
        """从指令引号中提取翻页按钮文本"""
        _NEXT_PAGE = re.compile(r"下一页|下一頁|next\s*page", re.I)
        matches = re.findall(r'["\u201c\u201d\'](.*?)["\u201c\u201d\']', instruction)
        for m in matches:
            if _NEXT_PAGE.search(m):
                return m.strip()
        return None

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
