#!/usr/bin/env python3
import sys
import argparse
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich import print as rprint
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import InMemoryHistory
import memory
import agent
from config import MINIMAX_API_KEY, MODEL
from tools import TOOL_SCHEMAS

console = Console()


def print_welcome():
    console.print(Panel.fit(
        f"[bold cyan]Personal Agent[/bold cyan]  [dim]model: {MODEL}[/dim]\n"
        "[dim]Commands: /exit  /new  /tools[/dim]",
        border_style="cyan"
    ))


def print_tool_call(name: str, args: dict):
    args_str = "  ".join(f"[dim]{k}=[/dim][yellow]{repr(v)[:60]}[/yellow]" for k, v in args.items())
    console.print(f"  [bold magenta]⚙ {name}[/bold magenta]  {args_str}")


def print_tool_result(name: str, result: str):
    lines = result.splitlines()
    preview = "\n".join(lines[:6])
    if len(lines) > 6:
        preview += f"\n[dim]...({len(lines) - 6} more lines)[/dim]"
    console.print(f"  [dim]└─ {preview}[/dim]\n")


def run_agent_turn(messages: list[dict]) -> list[dict]:
    text_buffer = []
    live_text = Text()

    def on_text(chunk: str):
        text_buffer.append(chunk)

    def on_tool_start(name, args):
        # Print buffered text so far if any
        if text_buffer:
            full = "".join(text_buffer)
            console.print(Markdown(full))
            text_buffer.clear()
        print_tool_call(name, args)

    def on_tool_result(name, result):
        print_tool_result(name, result)

    with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
        # We use a two-phase approach: collect during status, print after
        try:
            updated = agent.run_turn(
                messages,
                on_text=on_text,
                on_tool_start=on_tool_start,
                on_tool_result=on_tool_result,
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return messages

    # Print any remaining text
    if text_buffer:
        console.print(Markdown("".join(text_buffer)))

    return updated


def main():
    parser = argparse.ArgumentParser(description="Personal AI agent")
    parser.add_argument("--new", action="store_true", help="Start a fresh session")
    args = parser.parse_args()

    if not MINIMAX_API_KEY:
        console.print("[red]Error: MINIMAX_API_KEY not set. Add it to .env[/red]")
        sys.exit(1)

    if args.new:
        memory.clear()
        console.print("[dim]Session cleared.[/dim]")

    print_welcome()

    messages = memory.load()
    if messages:
        console.print(f"[dim]Resuming session ({len([m for m in messages if m['role'] == 'user'])} prior exchanges)[/dim]\n")

    input_history = InMemoryHistory()

    while True:
        try:
            user_input = pt_prompt("You: ", history=input_history).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            console.print("[dim]Goodbye.[/dim]")
            break
        elif user_input == "/new":
            memory.clear()
            messages = []
            console.print("[dim]Session cleared.[/dim]\n")
            continue
        elif user_input == "/tools":
            names = [s["function"]["name"] for s in TOOL_SCHEMAS]
            console.print("[bold]Available tools:[/bold] " + ", ".join(names))
            continue

        messages.append({"role": "user", "content": user_input})
        console.print()

        messages = run_agent_turn(messages)
        memory.save(messages)
        console.print()


if __name__ == "__main__":
    main()
