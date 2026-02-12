import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Style, init
import questionary
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.graph.state import AgentState
from src.utils.display import print_trading_output
from src.utils.analysts import ANALYST_ORDER, get_analyst_nodes
from src.utils.progress import progress
from src.utils.visualize import save_graph_as_png
from src.utils.langfuse_callback import get_langfuse_callbacks
from src.utils.langsmith_tracing import langsmith_flush
from src.utils.report import generate_final_report
from src.utils.company_context import build_company_context
from src.cli.input import (
    parse_cli_inputs,
)

import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json

# Load environment variables from .env file
load_dotenv()

init(autoreset=True)


def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        return None
    except TypeError as e:
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        return None


def parse_portfolio_manager_content(content: str) -> tuple[dict, str]:
    """Parse portfolio manager output. Returns (decisions, report)."""
    parsed = parse_hedge_fund_response(content)
    if not parsed or not isinstance(parsed, dict):
        return {}, ""
    if "decisions" in parsed:
        return parsed.get("decisions", {}), (parsed.get("report") or "")
    return parsed, ""


##### Run the Hedge Fund #####
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4.1",
    model_provider: str = "OpenAI",
):
    # Start progress tracking
    progress.start()

    try:
        # Build workflow (default to all analysts when none provided)
        workflow = create_workflow(selected_analysts if selected_analysts else None)
        agent = workflow.compile()

        callbacks = get_langfuse_callbacks(tags=["hedge-fund", "cli"])
        config = {"callbacks": callbacks} if callbacks else {}
        company_context = build_company_context(tickers, api_key=None)
        company_lines = []
        for t in tickers:
            ctx = company_context.get(t, {})
            name = ctx.get("name") or t
            sector = ctx.get("sector") or "—"
            industry = ctx.get("industry") or "—"
            company_lines.append(f"  • {t}: {name} (Sector: {sector}, Industry: {industry})")
        companies_summary = "\n".join(company_lines) if company_lines else "  (no company details)"
        initial_content = (
            "Make trading decisions based on the provided data.\n\n"
            "Companies under analysis:\n" + companies_summary
        )
        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(content=initial_content),
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                    "company_context": company_context,
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            },
            config=config,
        )
        langsmith_flush()
        last_content = final_state["messages"][-1].content
        decisions, report = parse_portfolio_manager_content(last_content)
        decisions = decisions or {}
        analyst_signals = final_state["data"]["analyst_signals"]
        current_prices = final_state["data"].get("current_prices", {})
        if not (report and report.strip()):
            report = generate_final_report(
                decisions=decisions,
                analyst_signals=analyst_signals,
                current_prices=current_prices,
                state=final_state,
            )
        return {
            "decisions": decisions,
            "analyst_signals": analyst_signals,
            "report": report,
        }
    finally:
        # Stop progress tracking
        progress.stop()


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts=None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Get analyst nodes from the configuration
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none selected
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())
    # Add selected analyst nodes
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    # Connect selected analysts to risk management
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    workflow.set_entry_point("start_node")
    return workflow


if __name__ == "__main__":
    inputs = parse_cli_inputs(
        description="Run the hedge fund trading system",
        require_tickers=True,
        default_months_back=None,
        include_graph_flag=True,
        include_reasoning_flag=True,
    )

    tickers = inputs.tickers
    selected_analysts = inputs.selected_analysts

    # Construct portfolio here
    portfolio = {
        "cash": inputs.initial_cash,
        "margin_requirement": inputs.margin_requirement,
        "margin_used": 0.0,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
            for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,
                "short": 0.0,
            }
            for ticker in tickers
        },
    }

    result = run_hedge_fund(
        tickers=tickers,
        start_date=inputs.start_date,
        end_date=inputs.end_date,
        portfolio=portfolio,
        show_reasoning=inputs.show_reasoning,
        selected_analysts=inputs.selected_analysts,
        model_name=inputs.model_name,
        model_provider=inputs.model_provider,
    )
    print_trading_output(result)
