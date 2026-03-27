"""Tool 协议 + 注册表 + 6 个 Tool 实现

将 Navigator / Extractor / PageIterator / RuleDiscoverer / Formatter 的能力
统一封装为标准 Tool 接口，Reasoner 通过 ToolRegistry 按名称选取工具。
"""

import logging
from abc import ABC, abstractmethod

from agent_scraper.pipeline.context import AgentContext, ToolResult

logger = logging.getLogger(__name__)


# ── 辅助: 宽容 Pydantic 模型 ─────────────────────────────

def _create_extract_output_model(fields: dict[str, str]):
    """创建 browser-use output_model_schema 用的 Pydantic 模型。

    生成一个扁平模型，只有一个 items 字段（JSON 数组），避免嵌套 $defs/$ref。
    LLM 返回格式: {"items": [{"URL": "...", "star": "..."}, ...]}

    容错设计:
    - model_validator 自动 strip 键名空白、清洗 items 内每条记录
    - extra="ignore" 忽略 LLM 多余字段
    """
    from pydantic import BaseModel, ConfigDict, Field, model_validator

    expected_keys = set(fields.keys())

    class ExtractResult(BaseModel):
        model_config = ConfigDict(extra="ignore")

        items: list[dict] = Field(
            ...,
            description=f"数据行列表，每行含字段: {', '.join(fields.keys())}",
        )

        @model_validator(mode="before")
        @classmethod
        def _clean_items(cls, data):
            if not isinstance(data, dict):
                return data
            raw_items = data.get("items", [])
            if not isinstance(raw_items, list):
                return data
            cleaned = []
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                # strip 键名和值的空白，过滤非预期字段
                row = {
                    k.strip(): v.strip() if isinstance(v, str) else v
                    for k, v in item.items()
                    if k.strip() in expected_keys
                }
                cleaned.append(row)
            data["items"] = cleaned
            return data

    return ExtractResult


# ── 协议 + 注册表 ────────────────────────────────────────


class Tool(ABC):
    """工具基类：每个 Tool 有名称、描述、execute"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    async def execute(self, ctx: AgentContext, **params) -> ToolResult: ...


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]


# ── Tool 实现 ─────────────────────────────────────────────


class NavigateTool(Tool):
    """浏览器导航：Agent 执行步骤到达目标页面，返回 HTML"""

    name = "navigate"
    description = "使用浏览器 Agent 执行导航步骤到达目标页面，返回页面 HTML"

    def __init__(self, navigator):
        self._nav = navigator

    async def execute(self, ctx: AgentContext, **params) -> ToolResult:
        # 清理上一轮浏览器（重试场景）
        if ctx.browser:
            try:
                await ctx.browser.stop()
            except Exception:
                pass
            ctx.browser = None

        try:
            # 仅复杂导航（多步骤 / 含点击等交互）才传截图作参考
            # 简单 goto 不需要截图，避免标注截图干扰 Agent
            steps = ctx.task.navigation_steps
            needs_visual = any(s.action != "goto" for s in steps)
            nav = await self._nav.navigate(
                steps, images=ctx.images if needs_visual else None,
            )
            ctx.browser = nav.browser
            ctx.html = nav.html
            # 从浏览器获取实际 URL（处理重定向/SPA 路由跳转）
            try:
                ctx.source_url = await ctx.browser.get_current_page_url()
            except Exception:
                pass
            # 兜底：浏览器取不到时从 goto 步骤推断
            if not ctx.source_url:
                for step in ctx.task.navigation_steps:
                    if step.action == "goto":
                        ctx.source_url = step.target
                        break
            return ToolResult(
                success=True,
                data={"html_size": len(nav.html)},
                summary=f"导航成功，HTML {len(nav.html) / 1024:.0f}KB",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), summary=f"导航失败: {e}")


class CaptureNavigateTool(Tool):
    """Capture 模式：浏览器导航 + 直接捕获少量值"""

    name = "capture_navigate"
    description = "浏览器导航并直接捕获少量特定值（capture 模式）"

    def __init__(self, navigator):
        self._nav = navigator

    async def execute(self, ctx: AgentContext, **params) -> ToolResult:
        try:
            steps = ctx.task.navigation_steps
            needs_visual = any(s.action != "goto" for s in steps)
            cap = await self._nav.navigate_and_capture(
                steps,
                ctx.task.extraction_goal.fields,
                raw_instruction=ctx.task.raw_instruction,
                images=ctx.images if needs_visual else None,
            )
            ctx.browser = cap.browser
            ctx.captured = cap.captured
            ctx.source_url = cap.page_url or ctx.source_url
            if cap.captured:
                return ToolResult(
                    success=True,
                    data=cap.captured,
                    summary=f"捕获到 {len(cap.captured)} 个字段: {list(cap.captured.keys())}",
                )
            return ToolResult(
                success=False,
                error="未捕获到任何值",
                summary="Capture 未捕获到值",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), summary=f"Capture 失败: {e}")


# ── 辅助: RuleDiscovery 校验 ─────────────────────────────

_HINT_RULE_MAP = {
    "load_more": lambda r: r.load_more_selector,
    "sub_pages": lambda r: r.sub_page_selector,
    "next_button": lambda r: r.next_button_selector,
    "pagination": lambda r: r.pagination_url,
}

def _check_missing(hints: list[str], rules) -> list[str]:
    """返回用户要求但规则中未找到的遍历模式"""
    return [h for h in hints if h in _HINT_RULE_MAP and not _HINT_RULE_MAP[h](rules)]

def _merge_rules(base, extra):
    """将 extra 中非空字段合并到 base"""
    data = base.model_dump()
    for k, v in extra.model_dump().items():
        if v is not None and v != "" and v is not False:
            data[k] = v
    from agent_scraper.core.models import PageRules
    return PageRules(**data)


class DiscoverRulesTool(Tool):
    """AI 分析页面结构，发现遍历规则（含幻觉校验 + 缺失重试）"""

    name = "discover_rules"
    description = "AI 分析页面结构，发现翻页 / 加载更多 / 子页面遍历规则"

    def __init__(self, rule_discoverer):
        self._discoverer = rule_discoverer

    async def execute(self, ctx: AgentContext, **params) -> ToolResult:
        from agent_scraper.core.models import PageRules
        from agent_scraper.extraction.rule_discoverer import RuleDiscoverer

        hints = ctx.task.extraction_goal.traversal_hints or []

        try:
            # 1. 首次发现
            rules = await self._discoverer.discover(
                ctx.html, ctx.source_url, hints,
            )

            # 2. CSS selector 真实性校验（防幻觉）
            if ctx.html:
                invalid = RuleDiscoverer.validate_selectors(ctx.html, rules)
                if invalid:
                    logger.warning("LLM 幻觉 selector: %s 在页面中无匹配，已清除", invalid)
                    rules = rules.model_copy(update={f: None for f in invalid})

            # 3. 用户意图校验：哪些 hint 没找到规则？
            missing = _check_missing(hints, rules)

            # 4. 缺失时重试一次（用更聚焦的提示词）
            if missing:
                logger.info("用户要求 %s 未发现规则，重试...", missing)
                retry_rules = await self._discoverer.discover_retry(
                    ctx.html, ctx.source_url, missing, missing_modes=missing,
                )
                # 重试结果也要校验
                if ctx.html:
                    invalid2 = RuleDiscoverer.validate_selectors(ctx.html, retry_rules)
                    if invalid2:
                        logger.warning("重试仍幻觉: %s，已清除", invalid2)
                        retry_rules = retry_rules.model_copy(update={f: None for f in invalid2})
                rules = _merge_rules(rules, retry_rules)
                missing = _check_missing(hints, rules)

            ctx.page_rules = rules

            # 5. 构建摘要
            info = []
            if rules.load_more_selector:
                info.append(f"load_more='{rules.load_more_selector}'")
            if rules.sub_page_selector:
                info.append(f"sub_page='{rules.sub_page_selector}'")
            if rules.next_button_selector:
                info.append(f"next_button='{rules.next_button_selector}'")
            if rules.pagination_url:
                info.append(f"pagination='{rules.pagination_url}'")

            summary = ", ".join(info) if info else "无遍历规则（单页模式）"

            # 6. 缺失反馈（有文本兜底的不算真正缺失）
            if missing:
                load_more_text = ctx.task.extraction_goal.load_more_text
                next_button_text = ctx.task.extraction_goal.next_button_text
                missing_names = []
                for m in missing:
                    if m == "load_more" and load_more_text:
                        info.append(f"load_more 将通过按钮文本 '{load_more_text}' 查找")
                    elif m == "next_button" and next_button_text:
                        info.append(f"next_button 将通过按钮文本 '{next_button_text}' 查找")
                    elif m == "next_button":
                        info.append("next_button 将通过常见文本（下一页/Next）查找")
                    else:
                        missing_names.append(m)
                if missing_names:
                    summary += f"; ⚠️ 未找到: {', '.join(missing_names)}"

            return ToolResult(success=True, data=rules, summary=f"规则: {summary}")
        except Exception as e:
            # 规则发现失败不致命，降级为单页
            ctx.page_rules = PageRules()
            return ToolResult(
                success=True,
                summary=f"规则发现异常，降级为单页: {e}",
            )


class IteratePagesTool(Tool):
    """根据规则遍历页面并逐页提取数据（流式，不缓存 HTML）"""

    name = "iterate_pages"
    description = "根据规则遍历页面并逐页提取数据（流式，不缓存 HTML）"

    def __init__(self, extractor=None):
        self._extractor = extractor

    async def execute(self, ctx: AgentContext, **params) -> ToolResult:
        from agent_scraper.browser.page_iterator import PageIterator
        from agent_scraper.core.models import PageRules

        has_samples = bool(ctx.task.extraction_goal.samples)
        logger.info("[IteratePages] 提取模式: %s",
                    "AutoScraper (有样本)" if has_samples else "LLM 兜底 (无样本)")

        if not ctx.browser:
            # 无浏览器兜底：直接对 ctx.html 提取，不缓存
            if self._extractor and has_samples and ctx.html:
                page_data = await self._extractor.extract(ctx.html, ctx.task.extraction_goal)
                ctx.extracted_data = page_data
            return ToolResult(
                success=bool(ctx.html),
                data={"page_count": 1 if ctx.html else 0},
                summary="无浏览器，使用已有 HTML",
            )

        rules = ctx.page_rules or PageRules()
        overrides = {}
        # 用户指定的 max_pages 优先于规则发现的 pagination_max
        user_max = ctx.task.extraction_goal.max_pages
        if user_max:
            overrides["pagination_max"] = user_max
        if overrides:
            rules = rules.model_copy(update=overrides)
        # 按钮文本传给 PageIterator 做兜底点击
        load_more_text = ctx.task.extraction_goal.load_more_text
        next_button_text = ctx.task.extraction_goal.next_button_text
        try:
            iterator = PageIterator(ctx.browser)
            all_data: dict[str, list] = {}
            page_count = 0
            skipped_pages = 0

            async for html in iterator.iterate(
                ctx.html, rules, ctx.source_url,
                load_more_text=load_more_text,
                next_button_text=next_button_text,
            ):
                page_count += 1
                logger.info("  提取页面 [%d] (%.0fKB)...", page_count, len(html) / 1024)
                ctx.on_event("progress", {"current": page_count, "total": 0})

                if self._extractor and has_samples:
                    page_data = await self._extractor.extract(html, ctx.task.extraction_goal)
                    # 跳过字段数量不一致的页面，避免污染全局对齐
                    counts = [len(v) for v in page_data.values()]
                    if page_data and len(set(counts)) > 1:
                        logger.warning("  页面 [%d] 字段数量不一致 %s，跳过",
                                       page_count, {k: len(v) for k, v in page_data.items()})
                        skipped_pages += 1
                    else:
                        for key, values in page_data.items():
                            all_data.setdefault(key, []).extend(values)
                        # 增量保存：每页提取后立即写入 ctx，防止中途崩溃丢数据
                        ctx.extracted_data = all_data

            if all_data:
                ctx.extracted_data = all_data

            # 构建带预期对比的 summary
            summary = f"遍历完成，共 {page_count} 个页面"
            expected = rules.pagination_max or user_max
            if expected and page_count < expected:
                summary += f" (预期 {expected} 页，实际 {page_count} 页，可能页面不足或提前终止)"
            if skipped_pages:
                summary += f"; ⚠️ {skipped_pages} 个页面因字段不一致被跳过"

            return ToolResult(
                success=True,
                data={"page_count": page_count},
                summary=summary,
            )
        except Exception as e:
            err_msg = str(e)
            is_browser_dead = any(k in err_msg for k in ["无法恢复", "-32000", "Failed to open"])
            collected = sum(len(v) for v in ctx.extracted_data.values()) if ctx.extracted_data else 0

            if is_browser_dead:
                # 浏览器彻底死亡：安全关闭 + 返回 success=False → Reasoner 自动重试整个计划
                logger.error("浏览器崩溃 (已收集 %d 条)，触发重试", collected)
                if ctx.browser:
                    try:
                        await ctx.browser.stop()
                    except Exception:
                        pass
                    ctx.browser = None  # NavigateTool 下轮创建新浏览器
                return ToolResult(
                    success=False,
                    error=err_msg,
                    data={"page_count": page_count, "browser_dead": True, "collected": collected},
                    summary=f"浏览器崩溃 (已收集 {collected} 条)，需要重新导航",
                )

            # 非浏览器死亡：优先保留已收集的数据，否则降级为单页提取
            if not ctx.extracted_data and self._extractor and has_samples and ctx.html:
                page_data = await self._extractor.extract(ctx.html, ctx.task.extraction_goal)
                ctx.extracted_data = page_data
            collected = sum(len(v) for v in ctx.extracted_data.values()) if ctx.extracted_data else 0
            if collected:
                return ToolResult(
                    success=True,
                    error=err_msg,
                    summary=f"遍历中断，已收集 {collected} 条数据: {e}",
                )
            return ToolResult(
                success=bool(ctx.html),
                error=err_msg,
                summary=f"遍历异常，降级为单页: {e}",
            )

class ExtractTool(Tool):
    """从 HTML 页面提取结构化数据

    有样本 → AutoScraper XPath 路径
    无样本（自由模式）→ browser-use Agent + output_model
    """

    name = "extract"
    description = "从 HTML 页面中提取结构化数据，有样本走 AutoScraper，无样本走 browser-use Agent"

    def __init__(self, extractor):
        self._extractor = extractor

    async def execute(self, ctx: AgentContext, **params) -> ToolResult:
        has_samples = bool(ctx.task.extraction_goal.samples)

        # 自由模式: 无样本 → browser-use Agent + output_model
        if not has_samples:
            logger.info("[ExtractTool] 策略: browser-use Agent (自由模式，无样本)")
            if ctx.browser:
                return await self._browser_agent_extract(ctx)
            return ToolResult(
                success=False,
                error="自由模式需要浏览器实例",
                summary="自由模式: 无浏览器",
            )

        # 有样本: IteratePagesTool 已经完成了流式提取
        if ctx.extracted_data:
            logger.info("[ExtractTool] 使用 IteratePages 已提取的数据 (AutoScraper/CSS)")
            field_counts = {k: len(v) for k, v in ctx.extracted_data.items()}
            total = sum(field_counts.values())
            summary = self._build_extract_summary(field_counts, ctx.task.extraction_goal.fields)
            return ToolResult(
                success=total > 0,
                data=field_counts,
                summary=summary,
                error="提取到 0 条数据" if total == 0 else None,
            )

        # 兜底: 有样本但无数据 + 有 HTML → 单页提取
        if ctx.html:
            try:
                page_data = await self._extractor.extract(ctx.html, ctx.task.extraction_goal)
                ctx.extracted_data = page_data
                field_counts = {k: len(v) for k, v in page_data.items()}
                total = sum(field_counts.values())
                summary = self._build_extract_summary(field_counts, ctx.task.extraction_goal.fields)
                return ToolResult(
                    success=total > 0,
                    data=field_counts,
                    summary=summary,
                    error="提取到 0 条数据" if total == 0 else None,
                )
            except Exception as e:
                return ToolResult(success=False, error=str(e), summary=f"提取失败: {e}")

        return ToolResult(success=False, error="无 HTML 页面", summary="没有可提取的页面")

    # ── 提取摘要构建 ──────────────────────────────────────

    @staticmethod
    def _build_extract_summary(field_counts: dict[str, int], expected_fields: dict[str, str]) -> str:
        """构建包含质量反馈的提取摘要"""
        total = sum(field_counts.values())
        if total == 0:
            return "提取到 0 条数据"

        summary = f"提取: {field_counts}"
        warnings = []

        # 字段缺失检查
        missing = set(expected_fields.keys()) - set(field_counts.keys())
        if missing:
            warnings.append(f"缺少字段: {', '.join(missing)}")

        # 字段长度不一致检查
        counts = list(field_counts.values())
        if len(counts) > 1 and min(counts) != max(counts):
            short = {k: v for k, v in field_counts.items() if v == min(counts)}
            warnings.append(f"字段数量不一致，最少 {min(counts)} 条: {list(short.keys())}")

        if warnings:
            summary += "; ⚠️ " + "; ".join(warnings)

        return summary

    # ── 自由模式: browser-use Agent 提取 ─────────────────

    async def _browser_agent_extract(self, ctx: AgentContext) -> ToolResult:
        """无样本时，用 browser-use Agent + 动态 Pydantic output_model_schema 提取"""
        import json
        import os

        from browser_use import Agent
        from browser_use.llm import ChatOpenAI

        from agent_scraper.core.llm import get_model_name

        goal = ctx.task.extraction_goal
        fields = goal.fields

        # 1. 动态创建扁平 Pydantic 模型（避免嵌套 $defs/$ref，browser-use 不支持）
        ExtractResult = _create_extract_output_model(fields)

        # 2. 构建提取任务
        fields_desc = "\n".join(f"- {k}: {v}" for k, v in fields.items())
        task_text = (
            f"从当前页面提取以下所有数据行:\n{fields_desc}\n\n"
            f"提取页面上所有匹配的数据行，每行包含上述所有字段。\n"
            f"不要导航到其他页面，仅从当前页面提取。"
        )

        logger.info("[ExtractTool] 自由模式: browser-use Agent + output_model_schema")
        logger.info("  字段: %s", list(fields.keys()))

        # 3. 创建 LLM + Agent
        llm = ChatOpenAI(
            model=get_model_name(),
            temperature=0,
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        try:
            agent = Agent(
                task=task_text,
                llm=llm,
                browser=ctx.browser,
                output_model_schema=ExtractResult,
            )
            history = await agent.run(max_steps=10)

            # 4. 解析结果 — 优先用 structured_output（自动 model_validate_json）
            records = []
            structured = history.structured_output
            if structured and hasattr(structured, "items") and structured.items:
                records = list(structured.items)  # list[dict]，已由 _clean_items 清洗
                logger.info("[ExtractTool] structured_output 解析成功: %d 条", len(records))

            # 降级: 手动解析 final_result JSON
            if not records:
                raw = history.final_result()
                records = self._parse_agent_result(raw, fields)

            if not records:
                logger.warning("[ExtractTool] browser-use Agent 未提取到数据")
                return ToolResult(
                    success=False,
                    error="Agent 未提取到数据",
                    summary="自由模式: 0 条数据",
                )

            # 5. 转换为 {field: [values]} 格式（空值用 "" 占位，保持各字段长度对齐）
            all_data: dict[str, list] = {f: [] for f in fields}
            for record in records:
                for field_name in fields:
                    val = record.get(field_name, "")
                    all_data[field_name].append(str(val) if val else "")

            ctx.extracted_data = all_data
            field_counts = {k: len(v) for k, v in all_data.items()}
            total = sum(field_counts.values())
            logger.info("[ExtractTool] 自由模式提取完成: %s", field_counts)

            return ToolResult(
                success=total > 0,
                data=field_counts,
                summary=f"自由模式提取: {field_counts}",
                error=None if total > 0 else "提取到 0 条数据",
            )
        except Exception as e:
            logger.error("[ExtractTool] browser-use Agent 提取异常: %s", e)
            return ToolResult(success=False, error=str(e), summary=f"自由模式提取失败: {e}")

    @staticmethod
    def _parse_agent_result(raw, fields: dict[str, str]) -> list[dict]:
        """解析 browser-use Agent 的 output_model 结果为记录列表"""
        import json

        if not raw:
            return []

        # 已经是列表（Pydantic 模型直接返回）
        if isinstance(raw, list):
            result = []
            for item in raw:
                if isinstance(item, dict):
                    result.append(item)
                elif hasattr(item, "model_dump"):
                    result.append(item.model_dump())
                elif hasattr(item, "__dict__"):
                    result.append(vars(item))
            return result

        # 字符串 → 尝试解析 JSON
        if isinstance(raw, str):
            text = raw.strip()
            if "```" in text:
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else text
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [r for r in parsed if isinstance(r, dict)]
                if isinstance(parsed, dict):
                    return [parsed]
            except json.JSONDecodeError:
                pass

        return []

    def clear_cache(self):
        """清除 CSS 选择器缓存和 AutoScraper 训练模型（供重试使用）"""
        self._extractor._css_rule_cache.clear()
        self._extractor._trained_scraper = None


class VisionSampleTool(Tool):
    """从截图标注区域识别文字 → 生成 AutoScraper 样本

    职责分离：
    - VLM 只负责识别可见文本字段（title、日期、价格等）
    - URL 类字段不需要样本，由 Extractor 从文本字段 XPath 推导 /@href
    """

    name = "vision_sample"
    description = "用 VLM 从截图识别标注区域文字，生成 AutoScraper 所需的样本数据"

    def __init__(self, llm_service):
        self._llm = llm_service

    async def execute(self, ctx: AgentContext, **params) -> ToolResult:
        import json
        from agent_scraper.pipeline.prompts import VISION_SAMPLE_PROMPT

        if not ctx.images:
            return ToolResult(success=False, error="无截图", summary="VisionSample: 无截图，跳过")

        fields = ctx.task.extraction_goal.fields

        # 1. 分离：文本字段交给 VLM，URL 字段由 Extractor XPath 推导
        text_fields = {k: v for k, v in fields.items() if not self._is_url_field(k, v)}

        if not text_fields:
            # 全是 URL 字段，没有文本字段可供 VLM 识别 → 跳过
            return ToolResult(
                success=True, error="所有字段均为 URL 类型，无法从截图生成样本",
                summary="VisionSample: 无文本字段，跳过",
            )

        fields_desc = "\n".join(f"- {k}: {v}" for k, v in text_fields.items())
        prompt = VISION_SAMPLE_PROMPT.format(fields_desc=fields_desc)

        try:
            # 2. VLM 只识别文本字段
            raw = await self._llm.call_with_images(prompt, ctx.images, caller="VisionSample")
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            visible_samples: dict[str, list[str]] = json.loads(raw)
            logger.info("[VisionSample] VLM 识别结果: %s", visible_samples)

            # 3. 写入 samples（URL 字段不需要样本，由 Extractor XPath 推导）
            ctx.task.extraction_goal.samples = visible_samples
            field_info = {k: len(v) for k, v in visible_samples.items()}
            return ToolResult(
                success=True,
                data=visible_samples,
                summary=f"VisionSample 生成样本: {field_info}",
            )
        except Exception as e:
            logger.warning("[VisionSample] 失败，降级为无样本模式: %s", e)
            return ToolResult(
                success=True,  # 不阻断流程，降级为无样本
                error=str(e),
                summary=f"VisionSample 失败，将走 Agent 兜底: {e}",
            )

    # ── 字段分类 ──────────────────────────────────────────

    @staticmethod
    def _is_url_field(field_name: str, field_desc: str) -> bool:
        """判断字段是否为 URL 类型"""
        indicators = ["url", "链接", "link", "href"]
        text = (field_name + " " + field_desc).lower()
        return any(kw in text for kw in indicators)



class FormatTool(Tool):
    """将提取的原始数据格式化为最终 ScrapedResult"""

    name = "format"
    description = "将提取的原始数据格式化为结构化结果（对齐字段、去重、补全 URL）"

    def __init__(self, formatter):
        self._formatter = formatter

    async def execute(self, ctx: AgentContext, **params) -> ToolResult:
        from agent_scraper.core.models import ScrapedResult

        # Capture 模式
        if ctx.captured:
            ctx.result = ScrapedResult(
                data=[ctx.captured],
                total_count=1,
                source_url=ctx.source_url,
            )
            return ToolResult(success=True, data={"total": 1}, summary="格式化: 1 条捕获记录")

        # Extract 模式
        if not ctx.extracted_data:
            ctx.result = ScrapedResult(data=[], total_count=0, source_url=ctx.source_url)
            return ToolResult(success=True, data={"total": 0}, summary="无数据需要格式化")

        try:
            result = await self._formatter.format(
                ctx.extracted_data, ctx.task.extraction_goal, ctx.source_url,
            )
            ctx.result = result
            return ToolResult(
                success=True,
                data={"total": result.total_count},
                summary=f"格式化完成: {result.total_count} 条记录",
            )
        except Exception as e:
            ctx.result = ScrapedResult(data=[], total_count=0, source_url=ctx.source_url)
            return ToolResult(success=False, error=str(e), summary=f"格式化失败: {e}")
