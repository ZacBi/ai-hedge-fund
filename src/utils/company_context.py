"""Helpers for company context (name, sector, industry) passed from stock input to agents."""

import logging

from src.tools.api import get_company_facts

logger = logging.getLogger(__name__)


def build_company_context(tickers: list[str], api_key: str | None = None) -> dict:
    """
    Fetch company facts for each ticker. Returns dict ticker -> {name, sector, industry, ...}.
    Used to pass company details from the graph start into analyst nodes.
    """
    company_context = {}
    for ticker in tickers:
        facts = get_company_facts(ticker, api_key=api_key)
        if facts is not None:
            logger.debug("get_company_facts %s: name=%s sector=%s", ticker, getattr(facts, "name", None), getattr(facts, "sector", None))
            company_context[ticker] = {
                "name": facts.name,
                "sector": facts.sector,
                "industry": facts.industry,
                "category": facts.category,
                "exchange": facts.exchange,
                "location": facts.location,
            }
        else:
            logger.warning("get_company_facts %s: no data (API returned none)", ticker)
            company_context[ticker] = {}
    return company_context


def format_company_context_for_prompt(ticker: str, state_data: dict) -> str:
    """
    From state["data"]["company_context"], format a short line for the given ticker
    for use in agent prompts (e.g. "Company: Apple Inc., Sector: Technology, Industry: Consumer Electronics").
    """
    ctx = (state_data.get("company_context") or {}).get(ticker) or {}
    if not ctx:
        return ""
    parts = []
    if ctx.get("name"):
        parts.append(f"Company: {ctx['name']}")
    if ctx.get("sector"):
        parts.append(f"Sector: {ctx['sector']}")
    if ctx.get("industry"):
        parts.append(f"Industry: {ctx['industry']}")
    if ctx.get("exchange"):
        parts.append(f"Exchange: {ctx['exchange']}")
    if not parts:
        return ""
    return " | ".join(parts)
