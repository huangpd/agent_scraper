"""质量评估器：FieldCheck + QualityScore + RetryEscalator

插入在 ExtractTool 之后，将"一次性提取"变成"提取→评估→重试"的自愈闭环。
- FieldCheck:      字段完整性（程序化）
- QualityScore:    数量一致性 + 样本对比（程序化）
- RetryEscalator:  分数不够时程序化分级升级重试策略（替代 LLM Replanner）
"""

import logging
import unicodedata
from dataclasses import dataclass, field

from agent_scraper.core.llm import LLMService
from agent_scraper.core.models import ExtractionGoal
from agent_scraper.pipeline.context import AgentContext
from agent_scraper.pipeline.retry_escalator import RetryEscalator

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
    retry_strategy: str | None = None  # RetryEscalator 决策

class Evaluator:
    """质量评估 + 程序化重试策略"""

    # 通过阈值，质量评分 ≥ 此值视为通过
    PASS_THRESHOLD = 0.6

    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service  # 保留兼容性，不再用于 replan
        self._escalator = RetryEscalator()

    async def evaluate(self, ctx: AgentContext) -> EvalResult:
        """FieldCheck → QualityScore → (失败时) RetryEscalator"""
        issues: list[str] = []

        field_ok = self._field_check(ctx.extracted_data, ctx.task.extraction_goal, issues)
        quality = self._quality_score(ctx.extracted_data, ctx.task.extraction_goal, issues)
        passed = field_ok and quality >= self.PASS_THRESHOLD

        retry_strategy = None
        if not passed and ctx.retry_count < ctx.max_retries - 1:
            self._record_failures(ctx)
            retry_strategy = self._escalator.decide(
                ctx.retry_count, issues, ctx.failure_memory,
            )

        result = EvalResult(
            passed=passed,
            field_check=field_ok,
            quality_score=quality,
            issues=issues,
            retry_strategy=retry_strategy,
        )
        logger.info(
            "评估: passed=%s, fields=%s, quality=%.2f, issues=%s, strategy=%s",
            passed, field_ok, quality, issues or "无", retry_strategy,
        )
        return result

    @staticmethod
    def _record_failures(ctx: AgentContext):
        """将失败的字段记录到 FailureMemory"""
        expected = set(ctx.task.extraction_goal.fields.keys())
        got = {k for k, v in ctx.extracted_data.items() if v}
        for field_name in expected - got:
            ctx.failure_memory.record_selector_failure(field_name, "<empty>")

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
                        if any(
                            sv_n in _normalize_text(str(v)) or _normalize_text(str(v)) in sv_n
                            for v in data[field_name]
                        ):
                            sample_match += 1
            if sample_total > 0:
                match_rate = sample_match / sample_total
                if match_rate < 0.5:
                    issues.append(f"样本匹配率低: {match_rate:.0%}")
                scores.append(match_rate)

        return sum(scores) / len(scores) if scores else 0.0

