"""
Main Application Entry Point.
Provides an interactive, conversational CLI for exploring the codebase
using the LangGraph ReAct Agent with live reasoning traces and memory.
"""

import sys
import uuid
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

from src.agent import get_agent_executor, stream_agent_reasoning

console = Console()


def print_banner():
    """Displays a welcoming header in the CLI."""
    banner_text = """
[bold cyan]╔════════════════════════════════════════════════════════════════╗[/bold cyan]
[bold cyan]║[/bold cyan]            [bold yellow]🤖 Agentic Codebase Intelligence Q&A System[/bold yellow]          [bold cyan]║[/bold cyan]
[bold cyan]║[/bold cyan]      [green]Target Repository: psf/requests[/green] | [magenta]Engine: LangGraph + Qdrant[/magenta]   [bold cyan]║[/bold cyan]
[bold cyan]╚════════════════════════════════════════════════════════════════╝[/bold cyan]
Type your questions about architecture, function behavior, or usage.
Type [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to stop.
"""
    console.print(banner_text)


def run_interactive_session():
    """Runs the interactive multi-turn conversation loop."""
    print_banner()

    with console.status("[bold green]Initializing Agent..."):
        try:
            agent = get_agent_executor()
        except Exception as e:
            console.print(f"[bold red]Initialization Error:[/bold red] {e}")
            sys.exit(1)

    # Unique thread ID for in-memory session persistence
    session_thread_id = str(uuid.uuid4())
    console.print(f"[dim]Session initialized with ID: {session_thread_id[:8]}...[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]Ask a question[/bold cyan]")
            if not user_input.strip():
                continue

            if user_input.lower().strip() in ("exit", "quit", "q"):
                console.print("\n[yellow]Goodbye![/yellow]")
                break

            console.print("\n[dim]Agent is thinking and exploring...[/dim]")

            final_answer = ""
            for event in stream_agent_reasoning(agent, user_input, thread_id=session_thread_id):
                ev_type = event.get("type")

                if ev_type == "tool_call":
                    tool_name = event.get("tool")
                    tool_args = event.get("args")
                    console.print(
                        f"  [bold yellow]⚙ Action:[/bold yellow] Agent invoked [cyan]`{tool_name}`[/cyan] with args: [dim]{tool_args}[/dim]"
                    )

                elif ev_type == "tool_result":
                    tool_name = event.get("tool")
                    preview = event.get("content", "").replace("\n", " ")
                    console.print(
                        f"  [bold blue]👁 Observation:[/bold blue] [dim]{tool_name} returned -> {preview[:120]}...[/dim]"
                    )

                elif ev_type == "final_answer":
                    final_answer = event.get("content", "")

            # Render markdown response
            if final_answer:
                console.print("\n")
                console.print(
                    Panel(
                        Markdown(final_answer),
                        title="[bold green]Agent Answer[/bold green]",
                        border_style="green",
                        expand=False,
                    )
                )
            else:
                console.print("[red]No final response produced by agent.[/red]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted. Exiting...[/yellow]")
            break
        except Exception as err:
            console.print(f"[bold red]Unexpected error during query:[/bold red] {err}")


if __name__ == "__main__":
    run_interactive_session()