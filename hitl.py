# Rich library imports for beautiful terminal output
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich import box
from rich.markup import escape

# LangChain message types
from langchain_core.messages import AIMessage, ToolMessage

# Console object for all terminal output
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTION 1 — Display the pending tool call to human
# ─────────────────────────────────────────────────────────────────────────────

def _render_tool_call(tool_name: str, tool_input: str) -> None:
    """Pretty-print the pending tool call for human review."""

    # Create a table with two columns: Field and Value
    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column("Field", style="bold cyan", width=16)
    table.add_column("Value", style="white")

    # Add tool name and input rows to the table
    table.add_row("Tool", f"[bold yellow]{tool_name}[/bold yellow]")
    table.add_row("Input / Query", f"[green]{tool_input}[/green]")

    # Print the table inside a red panel
    console.print()
    console.print(
        Panel(
            table,
            title="[bold red]⛔  INTERRUPT — Pending Tool Call[/bold red]",
            border_style="red",
            expand=False,
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTION 2 — Display the 3 options menu
# ─────────────────────────────────────────────────────────────────────────────

def _render_menu() -> None:
    """Print the three-option HITL control menu."""
    console.print()
    console.print("[bold white]  Choose an action:[/bold white]")
    console.print("  [bold green][A][/bold green] Approve  — run as-is")
    console.print("  [bold yellow][E][/bold yellow] Edit     — modify the tool input")
    console.print("  [bold red][R][/bold red] Reject   — skip this tool call")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HITL NODE — This is called by LangGraph at every tool call
# ─────────────────────────────────────────────────────────────────────────────

def hitl_node(state: dict) -> dict:
    """
    LangGraph node that pauses execution and prompts the human.
    Returns updated state based on human decision.
    """

    messages = state["messages"]

    # ── Step 1: Find the last AI message that has a tool call ────────────────
    ai_msg: AIMessage | None = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            ai_msg = msg
            break

    # If no tool call found, just approve and continue
    if ai_msg is None:
        return {
            "hitl_decision": "approved",
            "rejection_reason": None,
            "edited_tool_input": None,
        }

    # ── Step 2: Extract tool name and input from the AI message ──────────────
    tool_call = ai_msg.tool_calls[0]
    tool_name: str = tool_call["name"]
    tool_args: dict = tool_call.get("args", {})

    # Normalize input — different tools store query in different keys
    if "query" in tool_args:
        tool_input: str = tool_args["query"]
    elif "__arg1" in tool_args:
        tool_input = tool_args["__arg1"]
    else:
        tool_input = json.dumps(tool_args)

    # ── Step 3: Show the tool call and menu to human ─────────────────────────
    _render_tool_call(tool_name, tool_input)
    _render_menu()

    # ── Step 4: Wait for human decision in a loop ────────────────────────────
    while True:
        choice = Prompt.ask(
            "[bold white]Your decision[/bold white]",
            choices=["a", "e", "r", "A", "E", "R"],
            show_choices=False,
        ).lower()

        # ── APPROVE: run tool as-is ───────────────────────────────────────────
        if choice == "a":
            console.print("[bold green]✓ Approved.[/bold green] Tool will execute.\n")
            return {
                "hitl_decision": "approved",
                "rejection_reason": None,
                "edited_tool_input": None,
            }

        # ── EDIT: human provides new query ────────────────────────────────────
        elif choice == "e":
            console.print(f"\n[yellow]Current input:[/yellow] {tool_input}")
            new_input = Prompt.ask("[bold yellow]New input[/bold yellow]")

            # Make sure new input is not empty
            if new_input.strip():
                console.print(
                    f"[bold yellow]✎ Edited.[/bold yellow] "
                    f"Tool will run with: [green]{new_input}[/green]\n"
                )
                return {
                    "hitl_decision": "edited",
                    "rejection_reason": None,
                    "edited_tool_input": new_input.strip(),
                }
            else:
                console.print("[red]Empty input — please try again.[red]")

        # ── REJECT: tool is skipped, reason sent back to agent ───────────────
        elif choice == "r":
            reason = Prompt.ask(
                "[bold red]Rejection reason[/bold red] (shown to agent)"
            )
            reason = reason.strip() or "Rejected by human reviewer."
            console.print(f"[bold red]✗ Rejected.[/bold red] Reason: {reason}\n")

            # Inject a fake ToolMessage so graph stays structurally valid
            # Without this, LangGraph would crash expecting a tool result
            rejection_msg = ToolMessage(
                content=(
                    f"[HITL REJECTION] This tool call was rejected by the human.\n"
                    f"Reason: {reason}\n"
                    f"Do NOT retry this exact tool call. "
                    f"Either reformulate or answer from existing context."
                ),
                tool_call_id=tool_call["id"],
            )

            return {
                "hitl_decision": "rejected",
                "rejection_reason": reason,
                "edited_tool_input": None,
                "messages": [rejection_msg],
            }