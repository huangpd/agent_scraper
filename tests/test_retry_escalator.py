"""测试 RetryEscalator: 分级重试策略 + FailureMemory"""

from agent_scraper.pipeline.retry_escalator import FailureMemory, RetryEscalator


class TestFailureMemory:
    def test_record_selector_failure(self):
        mem = FailureMemory()
        mem.record_selector_failure("name", ".item a")
        mem.record_selector_failure("name", ".product-title")
        assert mem.failed_selectors == {"name": [".item a", ".product-title"]}

    def test_record_strategy(self):
        mem = FailureMemory()
        mem.record_strategy("refine_selectors")
        mem.record_strategy("clear_and_regenerate")
        assert mem.attempted_strategies == ["refine_selectors", "clear_and_regenerate"]

    def test_empty_initial_state(self):
        mem = FailureMemory()
        assert mem.failed_selectors == {}
        assert mem.attempted_strategies == []


class TestRetryEscalator:
    def setup_method(self):
        self.escalator = RetryEscalator()
        self.memory = FailureMemory()

    def test_l0_refine_selectors(self):
        strategy = self.escalator.decide(0, ["缺少字段"], self.memory)
        assert strategy == "refine_selectors"
        assert self.memory.attempted_strategies == ["refine_selectors"]

    def test_l1_clear_and_regenerate(self):
        strategy = self.escalator.decide(1, ["数据量过少"], self.memory)
        assert strategy == "clear_and_regenerate"

    def test_l2_switch_strategy(self):
        strategy = self.escalator.decide(2, ["字段长度不一致"], self.memory)
        assert strategy == "switch_strategy"

    def test_l3_skip(self):
        strategy = self.escalator.decide(3, ["样本匹配率低"], self.memory)
        assert strategy == "skip"

    def test_beyond_l3_still_skip(self):
        """超出级别数时保持 skip"""
        strategy = self.escalator.decide(10, ["问题"], self.memory)
        assert strategy == "skip"

    def test_progressive_escalation(self):
        """模拟完整的升级序列"""
        strategies = []
        for attempt in range(4):
            s = self.escalator.decide(attempt, ["问题"], self.memory)
            strategies.append(s)
        assert strategies == [
            "refine_selectors",
            "clear_and_regenerate",
            "switch_strategy",
            "skip",
        ]
        assert len(self.memory.attempted_strategies) == 4

    def test_memory_accumulates_across_decisions(self):
        """FailureMemory 在多次 decide 调用间累积"""
        self.escalator.decide(0, ["a"], self.memory)
        self.escalator.decide(1, ["b"], self.memory)
        assert self.memory.attempted_strategies == ["refine_selectors", "clear_and_regenerate"]
