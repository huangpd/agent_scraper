"""测试 PromptOptimizer 策略推理"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_scraper.pipeline.prompt_optimizer import PromptOptimizer, OptimizeResult


# ── _parse_response 单元测试 ──


class TestParseResponse:
    def test_full_response(self):
        content = json.dumps({
            "reasoning": "① 批量提取 → extract\n② 翻页: next_button + sub_pages",
            "optimized": "目标页面: https://example.com\n提取字段: title, time, URL\n遍历方式: sub_pages + next_button",
            "changes": ["添加 URL 字段", "设置 max_pages=20"],
            "skippable": False,
        }, ensure_ascii=False)

        result = PromptOptimizer._parse_response(content, "原始指令")
        assert "extract" in result.reasoning
        assert "https://example.com" in result.optimized
        assert len(result.changes) == 2
        assert result.changes[0] == "添加 URL 字段"
        assert result.skippable is False

    def test_skippable_true(self):
        content = json.dumps({
            "reasoning": "指令已经规范",
            "optimized": "原始指令",
            "changes": [],
            "skippable": True,
        })

        result = PromptOptimizer._parse_response(content, "原始指令")
        assert result.skippable is True
        assert result.changes == []

    def test_json_in_code_block(self):
        """LLM 返回 ```json ... ``` 包裹时也能解析"""
        inner = json.dumps({
            "reasoning": "ok",
            "optimized": "优化后",
            "changes": [],
            "skippable": False,
        })
        content = f"```json\n{inner}\n```"

        result = PromptOptimizer._parse_response(content, "原始")
        assert result.optimized == "优化后"

    def test_invalid_json_fallback(self):
        """JSON 解析失败时应降级使用原始指令"""
        result = PromptOptimizer._parse_response("不是JSON", "原始指令")
        assert result.optimized == "原始指令"
        assert result.reasoning == ""
        assert result.skippable is False

    def test_missing_fields_fallback(self):
        """JSON 中缺少字段时使用默认值"""
        content = json.dumps({"reasoning": "只有推理"})
        result = PromptOptimizer._parse_response(content, "原始指令")
        assert result.optimized == "原始指令"
        assert result.changes == []
        assert result.skippable is False


# ── optimize() 集成测试 ──


class TestOptimize:
    @pytest.mark.asyncio
    async def test_optimize_calls_llm(self):
        """验证 optimize 正确调用 LLM 并解析 JSON 响应"""
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps({
            "reasoning": "① 批量提取 → extract",
            "optimized": "优化后的指令",
            "changes": ["改动1"],
            "skippable": False,
        }, ensure_ascii=False)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        optimizer = PromptOptimizer(client=mock_client)
        result = await optimizer.optimize("测试指令")

        assert result.optimized == "优化后的指令"
        assert len(result.changes) == 1
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_optimize_llm_failure_returns_original(self):
        """LLM 调用失败时应返回原始指令"""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("API down")
        )

        optimizer = PromptOptimizer(client=mock_client)
        result = await optimizer.optimize("原始指令")

        assert result.optimized == "原始指令"
        assert result.skippable is True
        assert "失败" in result.reasoning


# ── 典型场景测试 ──


class TestScenarios:
    def test_news_with_sub_pages(self):
        """新闻列表 + 下一页 → 应保留所有字段和 sub_pages"""
        content = json.dumps({
            "reasoning": "① 批量提取新闻 → extract\n② 用户要求进子页 → sub_pages\n③ 翻页 → next_button\n④ 补充 URL 字段\n⑤ max_pages=20",
            "optimized": "目标页面: https://news.example.com/list\n提取字段: title(标题), time(发布时间), URL(详情页链接)\n遍历方式: sub_pages + next_button\n页数限制: 20\n样本数据:\n{\"title\":\"新闻标题\",\"time\":\"2026-01-01\",\"URL\":\"\"}",
            "changes": ["添加 URL 字段", "明确 sub_pages + next_button", "设置页数限制 20"],
            "skippable": False,
        }, ensure_ascii=False)

        result = PromptOptimizer._parse_response(content, "")
        assert "URL" in result.optimized
        assert "sub_pages" in result.optimized
        assert len(result.changes) == 3

    def test_capture_mode(self):
        """复制 URL → capture 模式，skippable"""
        content = json.dumps({
            "reasoning": "① 获取单个链接 → capture",
            "optimized": "复制当前页面的下载链接",
            "changes": [],
            "skippable": True,
        })

        result = PromptOptimizer._parse_response(content, "")
        assert result.skippable is True

    def test_single_page_no_traversal(self):
        """单页表格 → 无遍历"""
        content = json.dumps({
            "reasoning": "① 单页数据 → 无需遍历",
            "optimized": "目标页面: https://example.com/table\n提取字段: name(名称), price(价格)",
            "changes": [],
            "skippable": True,
        })

        result = PromptOptimizer._parse_response(content, "")
        assert result.skippable is True
        assert "遍历" not in result.optimized
