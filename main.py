#!/usr/bin/env python3
import sys
import argparse
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import InMemoryHistory
import memory
import agent
from config import ANTHROPIC_API_KEY, MINIMAX_API_KEY, MODELS, DEFAULT_MODEL
from tools import TOOL_SCHEMAS

console = Console()


def _model_label(key: str) -> str:
    return MODELS[key]["label"]


def print_welcome(model_key: str):
    console.print(Panel.fit(
        f"[bold cyan]Olympus[/bold cyan]  [dim]model: {_model_label(model_key)}[/dim]\n"
        "[dim]Commands: /exit  /new  /tools  /model[/dim]",
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


def run_agent_turn(messages: list, model_key: str) -> list:
    text_buffer = []

    def on_text(chunk: str):
        text_buffer.append(chunk)

    def on_tool_start(name, args):
        if text_buffer:
            console.print(Markdown("".join(text_buffer)))
            text_buffer.clear()
        print_tool_call(name, args)

    def on_tool_result(name, result):
        print_tool_result(name, result)

    with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
        try:
            updated = agent.run_turn(
                messages,
                model_key=model_key,
                on_text=on_text,
                on_tool_start=on_tool_start,
                on_tool_result=on_tool_result,
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return messages

    if text_buffer:
        console.print(Markdown("".join(text_buffer)))

    return updated


def handle_model_command(current_key: str) -> str:
    console.print("\n[bold]Available models:[/bold]")
    keys = list(MODELS.keys())
    for i, key in enumerate(keys, 1):
        cfg = MODELS[key]
        active = " [green]← active[/green]" if key == current_key else ""
        key_needed = "ANTHROPIC_API_KEY" if cfg["provider"] == "anthropic" else "MINIMAX_API_KEY"
        has_key = bool(ANTHROPIC_API_KEY if cfg["provider"] == "anthropic" else MINIMAX_API_KEY)
        key_status = "[green]✓[/green]" if has_key else "[red]✗ no key[/red]"
        console.print(f"  [cyan]{i}[/cyan]. {cfg['label']}  {key_status}{active}")

    console.print()
    try:
        choice = pt_prompt("Switch to (number or name, Enter to cancel): ").strip()
    except (EOFError, KeyboardInterrupt):
        return current_key

    if not choice:
        return current_key

    # Accept number or shorthand name
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(keys):
            new_key = keys[idx]
        else:
            console.print("[red]Invalid choice.[/red]")
            return current_key
    elif choice in MODELS:
        new_key = choice
    else:
        console.print("[red]Unknown model.[/red]")
        return current_key

    cfg = MODELS[new_key]
    has_key = bool(ANTHROPIC_API_KEY if cfg["provider"] == "anthropic" else MINIMAX_API_KEY)
    if not has_key:
        console.print(f"[red]No API key set for {cfg['label']}. Add it to .env[/red]")
        return current_key

    console.print(f"[green]Switched to {cfg['label']}[/green]\n")
    return new_key


def main():
    parser = argparse.ArgumentParser(description="Personal AI agent")
    parser.add_argument("--new", action="store_true", help="Start a fresh session")
    parser.add_argument("--model", choices=list(MODELS.keys()), default=DEFAULT_MODEL,
                        help="Model to use (default: sonnet)")
    args = parser.parse_args()

    model_key = args.model

    # Validate API key for chosen provider
    cfg = MODELS[model_key]
    has_key = bool(ANTHROPIC_API_KEY if cfg["provider"] == "anthropic" else MINIMAX_API_KEY)
    if not has_key:
        console.print(f"[red]Error: no API key for {cfg['label']}. Add it to .env[/red]")
        sys.exit(1)

    if args.new:
        memory.clear()
        console.print("[dim]Session cleared.[/dim]")

    print_welcome(model_key)

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
        elif user_input == "/model":
            model_key = handle_model_command(model_key)
            continue

        messages.append({"role": "user", "content": user_input})
        console.print()

        messages = run_agent_turn(messages, model_key)
        memory.save(messages)
        console.print()


if __name__ == "__main__":
    main()
