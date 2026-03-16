"""质量评估器：FieldCheck + QualityScore + Replanner

插入在 ExtractTool 之后，将"一次性提取"变成"提取→评估→重试"的自愈闭环。
- FieldCheck:  字段完整性（程序化）
- QualityScore: 数量一致性 + 样本对比（程序化）
- Replanner:   分数不够时 LLM 生成重试策略
"""

import json
import logging
import unicodedata
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from agent_scraper.core.models import ExtractionGoal
from agent_scraper.pipeline.context import AgentContext

logger = logging.getLogger(__name__)


def _normalize_text(s: str) -> str:
    """归一化文本用于宽松比较：全角→半角、大小写、去空白"""
    return unicodedata.normalize("NFKC", s).strip().lower()


@dataclass
class EvalResult:
    """评估结果"""
    passed: bool
    field_check: bool
    quality_score: float          # 0.0 – 1.0
    issues: list[str] = field(default_factory=list)
    retry_strategy: str | None = None  # Replanner 建议


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


class Evaluator:
    """质量评估 + LLM Replanner"""

    # 通过阈值，质量评分 ≥ 此值视为通过
    PASS_THRESHOLD = 0.6

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def evaluate(self, ctx: AgentContext) -> EvalResult:
        """FieldCheck → QualityScore → (失败时) Replanner"""
        issues: list[str] = []

        field_ok = self._field_check(ctx.extracted_data, ctx.task.extraction_goal, issues)
        quality = self._quality_score(ctx.extracted_data, ctx.task.extraction_goal, issues)
        passed = field_ok and quality >= self.PASS_THRESHOLD

        retry_strategy = None
        if not passed and ctx.retry_count < ctx.max_retries - 1:
            retry_strategy = await self._replan(ctx, issues)

        result = EvalResult(
            passed=passed,
            field_check=field_ok,
            quality_score=quality,
            issues=issues,
            retry_strategy=retry_strategy,
        )
        logger.info(
            "评估: passed=%s, fields=%s, quality=%.2f, issues=%s",
            passed, field_ok, quality, issues or "无",
        )
        return result

    # ── FieldCheck ────────────────────────────────────────

    @staticmethod
    def _field_check(
        data: dict[str, list], goal: ExtractionGoal, issues: list[str],
    ) -> bool:
        """所有目标字段是否都已提取到"""
        expected = set(goal.fields.keys())
        got = {k for k, v in data.items() if v}
        missing = expected - got
        if missing:
            issues.append(f"缺少字段: {missing}")
            return False
        return True

    # ── QualityScore ──────────────────────────────────────

    @staticmethod
    def _quality_score(
        data: dict[str, list], goal: ExtractionGoal, issues: list[str],
    ) -> float:
        """综合评分：非空率 + 字段长度一致性 + 数据量 + 样本对比"""
        if not data:
            issues.append("提取数据为空")
            return 0.0

        scores: list[float] = []

        # 1) 非空率
        lengths = [len(v) for v in data.values()]
        if any(l == 0 for l in lengths):
            issues.append("存在空字段")
            scores.append(0.0)
        else:
            scores.append(1.0)

        # 2) 字段长度一致性（各字段记录数应该相近）
        if len(lengths) > 1 and max(lengths) > 0:
            ratio = min(lengths) / max(lengths)
            if ratio < 0.5:
                issues.append(f"字段长度不一致: {dict(zip(data.keys(), lengths))}")
            scores.append(ratio)
        else:
            scores.append(1.0)

        # 3) 数据量
        total = sum(lengths)
        if total == 0:
            scores.append(0.0)
        elif total < 3:
            issues.append(f"数据量过少: {total} 条")
            scores.append(0.5)
        else:
            scores.append(1.0)

        # 4) 样本对比（如果有用户样本）
        if goal.samples:
            sample_match = 0
            sample_total = 0
            for field_name, sample_values in goal.samples.items():
                if field_name in data:
                    for sv in sample_values:
                        sample_total += 1
                        sv_n = _normalize_text(sv)
                        if any(sv_n in _normalize_text(str(v)) for v in data[field_name]):
                            sample_match += 1
            if sample_total > 0:
                match_rate = sample_match / sample_total
                if match_rate < 0.5:
                    issues.append(f"样本匹配率低: {match_rate:.0%}")
                scores.append(match_rate)

        return sum(scores) / len(scores) if scores else 0.0

    # ── Replanner ─────────────────────────────────────────

    async def _replan(self, ctx: AgentContext, issues: list[str]) -> str:
        """LLM 根据失败原因建议重试策略"""
        history_lines = "\n".join(
            f"  {i + 1}. {s.tool_name}: {'✓' if s.result.success else '✗'} {s.result.summary}"
            for i, s in enumerate(ctx.steps)
        )
        prompt = REPLAN_PROMPT.format(
            fields=list(ctx.task.extraction_goal.fields.keys()),
            extracted=json.dumps(
                {k: len(v) for k, v in ctx.extracted_data.items()}, ensure_ascii=False,
            ),
            issues=issues,
            history=history_lines,
            retry_count=ctx.retry_count,
            max_retries=ctx.max_retries,
        )
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            strategy = resp.choices[0].message.content.strip().lower()
            # 清洗：只保留已知策略名
            known = {"clear_css_cache", "retry_navigate", "skip"}
            strategy = strategy if strategy in known else "clear_css_cache"
            logger.info("Replanner 策略: %s", strategy)
            return strategy
        except Exception as e:
            logger.error("Replanner LLM 调用失败: %s，默认 clear_css_cache", e)
            return "clear_css_cache"
