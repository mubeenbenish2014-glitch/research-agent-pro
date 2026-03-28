
import uuid
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.markup import escape
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from graph import build_graph

console = Console()


def run(graph, user_input: str, thread_id: str):
    """Stream graph events and print each step."""

    config = {"configurable": {"thread_id": thread_id}}

    # Initial state passed into the graph
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "hitl_decision": None,
        "rejection_reason": None,
        "edited_tool_input": None,
    }

    console.print(Rule("[bold blue]Agent Started[/bold blue]"))

    # Stream graph events one by one
    for event in graph.stream(initial_state, config=config, stream_mode="values"):
        messages = event.get("messages", [])
        if not messages:
            continue

        last_msg = messages[-1]

        # Agent proposed a tool call
        if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
            tc = last_msg.tool_calls[0]
            args = tc.get("args", {})
            query = args.get("query") or args.get("__arg1") or str(args)
            console.print(f"\n[cyan]🤖 Agent wants to use:[/cyan] [yellow]{tc['name']}[/yellow]")
            console.print(f"[cyan]   Query:[/cyan] [green]{escape(query)}[/green]")

        # Tool result received
        elif isinstance(last_msg, ToolMessage):
            content = str(last_msg.content)
            if "[HITL REJECTION]" in content:
                console.print("\n[red]⛔ Tool was rejected by human[/red]")
            else:
                preview = content[:300] + ("..." if len(content) > 300 else "")
                console.print(Panel(escape(preview), title="[green]🔧 Tool Result[/green]", expand=False))

        # Final answer from agent
        elif isinstance(last_msg, AIMessage) and last_msg.content:
            console.print(Panel(escape(str(last_msg.content)), title="[bold green]✅ Final Answer[/bold green]", border_style="green"))

    console.print(Rule("[dim]Done[/dim]"))


def main():
    """Main REPL loop."""
    console.print("[bold cyan]Research Agent Pro[/bold cyan] — HITL Powered\n")

    # One thread per session
    thread_id = str(uuid.uuid4())
    graph = build_graph("checkpoints.db")

    console.print("[green]Ready![/green] Type your question. (type 'exit' to quit)\n")

    while True:
        user_input = console.input("[bold white]You ▶ [/bold white]").strip()

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye![/dim]")
            break

        run(graph, user_input, thread_id)


if __name__ == "__main__":
    main()