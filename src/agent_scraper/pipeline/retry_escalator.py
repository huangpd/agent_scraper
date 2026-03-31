"""分级重试策略：程序化决策替代 LLM Replanner

把"LLM 的不稳定性"变成"工程上的可控流程"——
每次失败升级一个级别，每个级别改变不同的条件。

    L0: refine_selectors       只清 CSS 缓存，保留 AutoScraper 模型
    L1: clear_and_regenerate   全部清除，重头生成
    L2: switch_strategy        放弃 CSS/AutoScraper，切换到 browser-use Agent
    L3+: skip                  接受当前结果
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FailureMemory:
    """跨重试的失败记忆：记录哪些选择器失败过、尝试过哪些策略。"""

    failed_selectors: dict[str, list[str]] = field(default_factory=dict)
    attempted_strategies: list[str] = field(default_factory=list)

    def record_selector_failure(self, field_name: str, selector: str):
        self.failed_selectors.setdefault(field_name, []).append(selector)

    def record_strategy(self, strategy: str):
        self.attempted_strategies.append(strategy)


class RetryEscalator:
    """程序化重试策略决策器。

    根据失败次数自动升级策略——不重复同一级别，不调用 LLM。
    """

    LEVELS = ["refine_selectors", "clear_and_regenerate", "switch_strategy", "skip"]

    def decide(self, attempt: int, issues: list[str], memory: FailureMemory) -> str:
        """返回当前失败次数对应的重试策略。

        Args:
            attempt: 刚失败的尝试编号（0-indexed）
            issues: Evaluator 发现的问题列表
            memory: 累积的失败记忆
        """
        level = min(attempt, len(self.LEVELS) - 1)
        strategy = self.LEVELS[level]
        memory.record_strategy(strategy)

        logger.info("[RetryEscalator] attempt=%d → L%d %s (issues: %s, history: %s)",
                    attempt, level, strategy, issues, memory.attempted_strategies)
        return strategy
