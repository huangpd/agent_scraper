"""OutputGuard: LLM 输出护栏 — 语法 + 结构 + 语义 + 边界

每个 LLM 输出点都必须经过对应的 guard 函数。
纯函数，无状态，无 LLM 调用。返回清洗后的结果（而非抛异常）。
"""

import logging
import re

from bs4 import BeautifulSoup

from agent_scraper.extraction.css_engine import validate_css

logger = logging.getLogger(__name__)

# CSS 选择器匹配数量的合理上界——超过此值说明选择器太宽泛
_MAX_CSS_MATCHES = 500
# CSS 选择器嵌套深度上限（> 的数量）
_MAX_CSS_DEPTH = 5


def validate_task(task) -> list[str]:
    """校验 ParsedTask 结构合理性。返回问题列表（空=通过）。"""
    issues = []

    # 至少一个 goto 步骤
    has_goto = any(s.action == "goto" for s in task.navigation_steps)
    if not has_goto and task.mode != "capture":
        issues.append("navigation_steps 缺少 goto 步骤")

    # fields 非空
    if not task.extraction_goal.fields:
        issues.append("extraction_goal.fields 为空")

    # mode 合法
    if task.mode not in ("extract", "capture"):
        issues.append(f"mode 不合法: {task.mode}")

    if issues:
        logger.warning("[OutputGuard] TaskParser 输出问题: %s", issues)

    return issues


def validate_css_selectors(selectors: dict, html: str) -> dict:
    """校验并过滤 CSS 选择器。返回有效的 selectors 子集。

    检查:
    1. CSS 语法合法（BeautifulSoup 可解析）
    2. 嵌套深度 ≤ _MAX_CSS_DEPTH
    3. 匹配数量在 [1, _MAX_CSS_MATCHES]
    """
    valid = {}
    for field, info in selectors.items():
        css = info if isinstance(info, str) else info.get("selector", info.get("css", ""))
        attr = "text" if isinstance(info, str) else info.get("attr", "text")

        if not css:
            logger.warning("[OutputGuard] 字段 '%s' 选择器为空，跳过", field)
            continue

        # 嵌套深度
        depth = css.count(">")
        if depth > _MAX_CSS_DEPTH:
            logger.warning("[OutputGuard] 字段 '%s' 选择器嵌套过深 (%d 层): %s", field, depth, css)
            continue

        # 语法 + 匹配
        result = validate_css(html, css, attr)
        if not result["valid"]:
            logger.warning("[OutputGuard] 字段 '%s' 选择器无匹配: %s", field, css)
            continue

        if result["count"] > _MAX_CSS_MATCHES:
            logger.warning("[OutputGuard] 字段 '%s' 选择器匹配过多 (%d 条，上限 %d): %s",
                           field, result["count"], _MAX_CSS_MATCHES, css)
            continue

        valid[field] = info

    filtered = len(selectors) - len(valid)
    if filtered:
        logger.info("[OutputGuard] CSS 选择器校验: %d/%d 有效，过滤 %d 个",
                    len(valid), len(selectors), filtered)
    return valid


def validate_samples(samples: dict, html: str) -> dict:
    """校验样本值确实存在于 HTML 文本中。返回过滤后的 samples。

    对每个字段的每个样本值，检查它是否出现在 HTML 纯文本或属性值中。
    过滤不存在的样本；整个字段无有效样本则删除该字段。
    """
    if not samples or not html:
        return {}

    # 提取 HTML 纯文本 + 所有 href/src 属性值
    soup = BeautifulSoup(html, "lxml")
    text_content = soup.get_text()
    attr_values = set()
    for tag in soup.find_all(True):
        for attr in ("href", "src", "data-href"):
            val = tag.get(attr)
            if val:
                attr_values.add(val)
    search_text = text_content + " " + " ".join(attr_values)

    valid = {}
    for field, values in samples.items():
        matched = []
        for v in values:
            v_str = str(v).strip()
            if not v_str:
                continue
            if v_str in search_text:
                matched.append(v)
            else:
                logger.warning("[OutputGuard] 样本 '%s'.'%s' 不存在于 HTML 中，丢弃",
                               field, v_str[:60])
        if matched:
            valid[field] = matched
        else:
            logger.warning("[OutputGuard] 字段 '%s' 全部样本无匹配，丢弃", field)

    if valid != samples:
        logger.info("[OutputGuard] 样本校验: %d/%d 个字段有效",
                    len(valid), len(samples))
    return valid


def validate_page_rules(rules, html: str):
    """校验 PageRules 中的 CSS 选择器。返回清洗后的 PageRules + 失败详情。

    统一的校验入口，替代 RuleDiscoverer.validate_selectors + DiscoverRulesTool 双重校验。

    Returns:
        (cleaned_rules, failures) — failures 是 list[dict]，每项含:
            field: 字段名 (如 "sub_page_selector")
            selector: 被拒绝的选择器
            reason: 拒绝原因 (如 "无匹配"、"嵌套过深(6层)"、"匹配过多(523条)")
    """
    failures: list[dict] = []

    for field_name, selector in [
        ("load_more_selector", rules.load_more_selector),
        ("next_button_selector", rules.next_button_selector),
        ("sub_page_selector", rules.sub_page_selector),
    ]:
        if not selector:
            continue

        # 嵌套深度
        depth = selector.count(">")
        if depth > _MAX_CSS_DEPTH:
            reason = f"嵌套过深({depth}层，上限{_MAX_CSS_DEPTH})"
            logger.warning("[OutputGuard] %s %s: %s", field_name, reason, selector)
            failures.append({"field": field_name, "selector": selector, "reason": reason})
            continue

        result = validate_css(html, selector)
        if not result["valid"]:
            reason = "无匹配（选择器在当前HTML中匹配0个元素）"
            logger.warning("[OutputGuard] %s %s: %s", field_name, reason, selector)
            failures.append({"field": field_name, "selector": selector, "reason": reason})
            continue

        # sub_page_selector 匹配过多可能是误选
        if field_name == "sub_page_selector" and result["count"] > _MAX_CSS_MATCHES:
            reason = f"匹配过多({result['count']}条，上限{_MAX_CSS_MATCHES})"
            logger.warning("[OutputGuard] %s %s: %s", field_name, reason, selector)
            failures.append({"field": field_name, "selector": selector, "reason": reason})

    if failures:
        invalid_fields = [f["field"] for f in failures]
        rules = rules.model_copy(update={f: None for f in invalid_fields})
        logger.info("[OutputGuard] PageRules 校验: 清除 %s", invalid_fields)

    return rules, failures
