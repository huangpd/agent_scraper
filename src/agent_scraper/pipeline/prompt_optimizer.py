"""PromptOptimizer: 策略编排器，在用户输入和 TaskParser 之间插入一层推理。

将用户模糊的自然语言指令转化为框架最优指令，消除 TaskParser 的猜测空间。
纯推理，不访问目标网页，基于用户描述 + 框架知识做决策。
"""

import json
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from agent_scraper.core.llm import create_openai_client, get_model_name

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是 Agent Scraper 框架的策略编排专家。

## 第零原则：保留用户原文，最小补充

用户写的步骤式指令就是框架最优格式。你的工作不是重写指令，而是：
1. **原样保留**用户的步骤、字段、样本、措辞
2. **只做最小补充**：补缺失的 URL 字段（仅在进详情页时）

如果用户指令已经清晰完整，直接返回 skippable=true，optimized 原样返回。
大多数情况下用户指令都是清晰的，你应该倾向于返回 skippable=true。

你**绝对不能**做以下事情：
- 不能改写用户的用词和措辞（如 "Load more files"、"遍历子文件夹" 等都是正确的表达）
- 不能把步骤式指令改写成"目标页面:/提取字段:/遍历方式:"等摘要格式
- 不能添加用户没要求的遍历策略
- 不能删减用户列出的字段
- 不能修改用户提供的样本数据
- **不能要求或建议用户提供样本数据**

## 框架的两种提取模式

1. **样本模式**（有样本）：用户提供 JSONL 样本 → AutoScraper 训练 XPath → 秒级提取
2. **自由模式**（无样本）：无需样本 → browser-use Agent + LLM 智能提取 → 自动沉淀规则缓存

两种模式都是完整支持的。**没有样本数据不是缺陷，不需要补充样本。**

## 什么时候 skippable=true

以下条件**任一**满足时，返回 skippable=true：
- 指令步骤清晰完整（有导航步骤 + 提取目标）
- 指令已经包含足够信息让 TaskParser 正确解析
- 没有样本数据也算清晰完整（自由模式会处理）

## 补充规则（仅在信息缺失时才补充）

1. **URL 字段**：用户要求进详情页但样本中没有 URL 字段 → 在样本中补充 "URL":""
2. **capture 模式**：用户说"复制URL"/"获取链接"/"capture" 等少量值场景 → 提醒用 capture 模式

## 输出格式

严格输出 JSON，不要 ```json``` 代码块：

{
  "reasoning": "① ... ② ...",
  "optimized": "保留用户原文或仅做最小修改后的指令",
  "changes": ["改动1"],
  "skippable": false
}

optimized 必须是自然语言步骤式文本，不能是 XML/JSON 结构。
changes 为空数组时 skippable 应为 true。
"""

USER_PROMPT = """\
请优化以下用户指令：

{instruction}
"""


@dataclass
class OptimizeResult:
    """优化结果"""
    optimized: str        # 标准化指令文本
    reasoning: str        # 推理过程
    changes: list[str]    # 改动摘要
    skippable: bool       # 指令已规范时 True


class PromptOptimizer:
    """策略编排器：将用户模糊指令转化为框架最优指令"""

    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or create_openai_client()
        self.model = get_model_name()

    async def optimize(self, instruction: str) -> OptimizeResult:
        """分析用户指令，输出优化后的标准化指令 + 推理过程"""
        logger.info("[PromptOptimizer] 开始优化指令...")

        prompt = USER_PROMPT.format(instruction=instruction)

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            content = resp.choices[0].message.content.strip()
            return self._parse_response(content, instruction)

        except Exception as e:
            logger.error("[PromptOptimizer] LLM 调用失败: %s，返回原始指令", e)
            return OptimizeResult(
                optimized=instruction,
                reasoning=f"优化失败: {e}",
                changes=[],
                skippable=True,
            )

    @staticmethod
    def _parse_response(content: str, original: str) -> OptimizeResult:
        """解析 LLM 的 JSON 响应"""
        # 去掉可能的 ```json ``` 包裹
        text = content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("[PromptOptimizer] JSON 解析失败，返回原始指令")
            return OptimizeResult(
                optimized=original,
                reasoning="",
                changes=[],
                skippable=False,
            )

        reasoning = data.get("reasoning", "")
        optimized = data.get("optimized", original)
        changes = data.get("changes", [])
        skippable = data.get("skippable", False)

        if not isinstance(changes, list):
            changes = []

        logger.info(
            "[PromptOptimizer] 优化完成: %d 项改动, skippable=%s",
            len(changes), skippable,
        )
        return OptimizeResult(
            optimized=optimized.strip() if isinstance(optimized, str) else original,
            reasoning=reasoning.strip() if isinstance(reasoning, str) else "",
            changes=changes,
            skippable=bool(skippable),
        )
