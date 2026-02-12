"""
生成中文深度研报（类 Deep Research）：根据 decisions + analyst_signals 调用 LLM 输出 Markdown。
"""
from __future__ import annotations

import json
from pydantic import BaseModel, Field

from src.prompts import get_prompt_template
from src.utils.llm import call_llm


class FinalReportOutput(BaseModel):
    """LLM 返回的研报正文（可为 Markdown 或纯文本）。"""
    report: str = Field(description="完整研报正文，Markdown 格式")


def _format_context(decisions: dict, analyst_signals: dict, current_prices: dict | None = None) -> str:
    """将决策与分析师信号格式化为供 LLM 使用的上下文字符串。"""
    lines = ["## 组合经理最终决策", "```json", json.dumps(decisions, ensure_ascii=False, indent=2), "```"]
    lines.append("\n## 各分析师信号（按标的）")
    tickers = set()
    for _agent, signals in analyst_signals.items():
        tickers.update(signals.keys())
    for ticker in sorted(tickers):
        lines.append(f"\n### {ticker}")
        if current_prices and ticker in current_prices:
            lines.append(f"当前价格: {current_prices[ticker]}")
        for agent, signals in analyst_signals.items():
            if ticker not in signals:
                continue
            s = signals[ticker]
            lines.append(f"- **{agent}**: {s.get('signal', '')} (置信度: {s.get('confidence', 0)})")
            if s.get("reasoning"):
                lines.append(f"  {s['reasoning'][:200]}{'...' if len(s.get('reasoning', '')) > 200 else ''}")
    return "\n".join(lines)


def generate_final_report(
    decisions: dict,
    analyst_signals: dict,
    current_prices: dict | None = None,
    state: dict | None = None,
) -> str:
    """
    根据最终决策与分析师信号生成中文深度研报（Markdown）。
    若 state 提供 metadata（model_name, model_provider, request），则使用同配置的 LLM。
    """
    context = _format_context(decisions, analyst_signals or {}, current_prices)
    template = get_prompt_template("hedge-fund/final_report")
    prompt = template.invoke({"context": context})
    agent_name = "final_report"
    out = call_llm(
        prompt=prompt,
        pydantic_model=FinalReportOutput,
        agent_name=agent_name,
        state=state,
        default_factory=lambda: FinalReportOutput(report="研报生成失败，请查看各分析师信号与决策。"),
    )
    return out.report if out else ""
