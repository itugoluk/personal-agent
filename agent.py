import json
import httpx
from typing import Generator, Union
from config import MINIMAX_API_KEY, MINIMAX_API_BASE, MODEL
from tools import TOOL_SCHEMAS, dispatch

SYSTEM_PROMPT = """You are a personal AI agent running on a MacBook. You are a skilled assistant for coding, file management, and general task automation.

You have access to tools that let you run shell commands, read/write files, search the web, execute Python, and list directories. Use them proactively when needed — don't just describe what you'd do, actually do it.

When writing or editing code, create actual files. When the user asks about their system, check it. Be concise but thorough."""


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }


def stream_response(messages: list) -> Generator[Union[str, dict], None, None]:
    """
    Streams the assistant response. Yields:
    - str chunks for text tokens
    - dict {"tool_call": {...}} when a tool call is encountered
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "tools": TOOL_SCHEMAS,
        "stream": True,
    }

    tool_calls_acc: dict[int, dict] = {}

    with httpx.Client(timeout=60) as client:
        with client.stream(
            "POST",
            f"{MINIMAX_API_BASE}/text/chatcompletion_v2",
            headers=_headers(),
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                # Text token
                if delta.get("content"):
                    yield delta["content"]

                # Tool call deltas
                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.get("id"):
                        tool_calls_acc[idx]["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        tool_calls_acc[idx]["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        tool_calls_acc[idx]["function"]["arguments"] += fn["arguments"]

                # Emit completed tool calls on finish
                finish = choices[0].get("finish_reason")
                if finish == "tool_calls":
                    for tc in tool_calls_acc.values():
                        yield {"tool_call": tc}
                    tool_calls_acc = {}


def run_turn(messages: list, on_text=None, on_tool_start=None, on_tool_result=None) -> list:
    """
    Runs one full agentic turn (may involve multiple tool calls).
    Callbacks:
      on_text(chunk: str)
      on_tool_start(name: str, args: dict)
      on_tool_result(name: str, result: str)
    Returns the updated messages list.
    """
    messages = list(messages)

    while True:
        text_parts = []
        tool_calls_in_turn = []

        for item in stream_response(messages):
            if isinstance(item, str):
                text_parts.append(item)
                if on_text:
                    on_text(item)
            elif isinstance(item, dict) and "tool_call" in item:
                tool_calls_in_turn.append(item["tool_call"])

        assistant_msg: dict = {"role": "assistant"}
        full_text = "".join(text_parts)

        if tool_calls_in_turn:
            assistant_msg["tool_calls"] = tool_calls_in_turn
            if full_text:
                assistant_msg["content"] = full_text
            messages.append(assistant_msg)

            for tc in tool_calls_in_turn:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                if on_tool_start:
                    on_tool_start(name, args)
                result = dispatch(name, args)
                if on_tool_result:
                    on_tool_result(name, result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
        else:
            assistant_msg["content"] = full_text
            messages.append(assistant_msg)
            break

    return messages
