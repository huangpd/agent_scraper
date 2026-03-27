"""端到端测试：用户提示词 → TaskParser → 导航 → RuleDiscoverer → PageRules JSON

用法：
    python test_prompt_to_rules.py                     # 交互模式，粘贴提示词
    python test_prompt_to_rules.py --instruction "..."  # 命令行传入
    python test_prompt_to_rules.py --skip-nav --html page.html  # 跳过导航，直接用本地 HTML
"""

import asyncio
import argparse
import json
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-5s [%(name)s] %(message)s",
)
# 静默第三方库
for name in ("httpx", "httpcore", "openai", "browser_use", "urllib3"):
    logging.getLogger(name).setLevel(logging.WARNING)


async def run(instruction: str, skip_nav: bool = False, html_file: str | None = None, headless: bool = True):
    from agent_scraper.core.llm import LLMService
    from agent_scraper.pipeline.task_parser import TaskParser
    from agent_scraper.extraction.rule_discoverer import RuleDiscoverer

    llm = LLMService()

    # ── 1. 解析提示词 ──
    print("\n" + "=" * 60)
    print("1. TaskParser 解析")
    print("=" * 60)
    parser = TaskParser(llm)
    task = await parser.parse(instruction)

    print(f"  mode:            {task.mode}")
    print(f"  navigation:      {[s.model_dump(exclude_defaults=True) for s in task.navigation_steps]}")
    print(f"  fields:          {task.extraction_goal.fields}")
    print(f"  traversal_hints: {task.extraction_goal.traversal_hints}")
    print(f"  max_pages:       {task.extraction_goal.max_pages}")
    print(f"  load_more_text:  {task.extraction_goal.load_more_text}")
    print(f"  next_button_text:{task.extraction_goal.next_button_text}")

    # ── 2. 获取 HTML ──
    html = ""
    source_url = ""
    browser = None

    if html_file:
        print(f"\n  使用本地 HTML: {html_file}")
        with open(html_file) as f:
            html = f.read()
        for step in task.navigation_steps:
            if step.action == "goto":
                source_url = step.target
                break
    elif not skip_nav:
        print("\n" + "=" * 60)
        print("2. Navigator 导航")
        print("=" * 60)
        from agent_scraper.browser.navigator import Navigator
        nav = Navigator(headless=headless)
        try:
            result = await nav.navigate(task.navigation_steps)
            html = result.html
            browser = result.browser
            try:
                source_url = await browser.get_current_page_url()
            except Exception:
                pass
            if not source_url:
                for step in task.navigation_steps:
                    if step.action == "goto":
                        source_url = step.target
                        break
            print(f"  source_url: {source_url}")
            print(f"  html_size:  {len(html) / 1024:.0f} KB")
        except Exception as e:
            print(f"  导航失败: {e}")
            return
    else:
        print("\n  跳过导航（无 HTML，仅测试解析）")

    # ── 3. RuleDiscoverer ──
    if html and task.extraction_goal.traversal_hints:
        print("\n" + "=" * 60)
        print("3. RuleDiscoverer 规则发现")
        print("=" * 60)
        discoverer = RuleDiscoverer(llm)
        rules = await discoverer.discover(html, source_url, task.extraction_goal.traversal_hints)

        # 校验 selector 真实性
        invalid = RuleDiscoverer.validate_selectors(html, rules)
        if invalid:
            print(f"\n  ⚠️ 幻觉 selector (页面无匹配): {invalid}")
            rules = rules.model_copy(update={f: None for f in invalid})

        # 检查缺失
        from agent_scraper.pipeline.tools import _check_missing
        missing = _check_missing(task.extraction_goal.traversal_hints, rules)

        # 缺失时重试
        if missing:
            print(f"\n  缺失 {missing}，用 DISCOVER_RETRY_PROMPT 重试...")
            retry_rules = await discoverer.discover_retry(
                html, source_url, missing, missing_modes=missing,
            )
            invalid2 = RuleDiscoverer.validate_selectors(html, retry_rules)
            if invalid2:
                print(f"  重试仍幻觉: {invalid2}")
                retry_rules = retry_rules.model_copy(update={f: None for f in invalid2})
            from agent_scraper.pipeline.tools import _merge_rules
            rules = _merge_rules(rules, retry_rules)
            missing = _check_missing(task.extraction_goal.traversal_hints, rules)

        # 输出结果
        rules_json = rules.model_dump(exclude_defaults=False)
        print("\n" + "=" * 60)
        print("PageRules 结果:")
        print("=" * 60)
        print(json.dumps(rules_json, indent=2, ensure_ascii=False))

        if missing:
            print(f"\n  ⚠️ 最终仍缺失: {missing}")
        else:
            print("\n  ✓ 所有 traversal_hints 均已找到规则")
    elif not task.extraction_goal.traversal_hints:
        print("\n  无 traversal_hints，单页模式，无需发现规则")
    else:
        print("\n  无 HTML，跳过规则发现")

    # 清理
    if browser:
        try:
            await browser.stop()
        except Exception:
            pass


def read_instruction():
    """从 stdin 读取多行提示词，空行结束"""
    print("请输入提示词（输入空行结束）:")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="测试：提示词 → 解析 → 导航 → 规则发现")
    ap.add_argument("--instruction", "-i", type=str, help="提示词（不传则交互输入）")
    ap.add_argument("--skip-nav", action="store_true", help="跳过浏览器导航，仅测试 TaskParser")
    ap.add_argument("--html", type=str, help="本地 HTML 文件路径，跳过导航直接用")
    ap.add_argument("--no-headless", action="store_true", help="显示浏览器窗口")
    args = ap.parse_args()

    instruction = args.instruction or read_instruction()
    if not instruction.strip():
        print("未输入提示词，退出")
        sys.exit(1)

    print(f"\n提示词:\n{instruction}")

    asyncio.run(run(
        instruction=instruction,
        skip_nav=args.skip_nav,
        html_file=args.html,
        headless=not args.no_headless,
    ))


if __name__ == "__main__":
    main()
