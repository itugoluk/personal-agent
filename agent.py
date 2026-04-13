import json
import httpx
from typing import Generator, Union
from config import ANTHROPIC_API_KEY, MINIMAX_API_KEY, MINIMAX_API_BASE, MODELS
from tools import TOOL_SCHEMAS, dispatch

SYSTEM_PROMPT = """You are a personal AI agent running on a MacBook. You are a skilled assistant for coding, file management, and general task automation.

You have access to tools that let you run shell commands, read/write files, search the web, execute Python, and list directories. Use them proactively when needed — don't just describe what you'd do, actually do it.

When writing or editing code, create actual files. When the user asks about their system, check it. Be concise but thorough."""

# Anthropic tool schemas (different format from OpenAI-style)
ANTHROPIC_TOOL_SCHEMAS = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in TOOL_SCHEMAS
]


# ---------------------------------------------------------------------------
# Message format conversion
# Internal storage uses OpenAI-style format. Convert to Anthropic on the fly.
# ---------------------------------------------------------------------------

def _to_anthropic_messages(messages: list) -> list:
    """Convert OpenAI-style message list to Anthropic format."""
    result = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg["role"]

        if role == "user":
            result.append({"role": "user", "content": msg["content"]})
            i += 1

        elif role == "assistant":
            content = []
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})
            for tc in msg.get("tool_calls", []):
                try:
                    inp = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    inp = {}
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": inp,
                })
            result.append({"role": "assistant", "content": content})
            i += 1

        elif role == "tool":
            # Collect consecutive tool results into one user message
            tool_results = []
            while i < len(messages) and messages[i]["role"] == "tool":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": messages[i]["tool_call_id"],
                    "content": messages[i]["content"],
                })
                i += 1
            result.append({"role": "user", "content": tool_results})

        else:
            i += 1

    return result


# ---------------------------------------------------------------------------
# Anthropic streaming
# ---------------------------------------------------------------------------

def _stream_anthropic(messages: list, model_id: str) -> Generator[Union[str, dict], None, None]:
    payload = {
        "model": model_id,
        "max_tokens": 8096,
        "system": SYSTEM_PROMPT,
        "messages": _to_anthropic_messages(messages),
        "tools": ANTHROPIC_TOOL_SCHEMAS,
        "stream": True,
    }
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    tool_calls_acc = {}  # id -> {id, name, input_str}

    with httpx.Client(timeout=120) as client:
        with client.stream("POST", "https://api.anthropic.com/v1/messages",
                           headers=headers, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")

                if etype == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "tool_use":
                        tool_calls_acc[block["id"]] = {
                            "id": block["id"],
                            "name": block["name"],
                            "input_str": "",
                        }

                elif etype == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")
                    elif delta.get("type") == "input_json_delta":
                        # Find the current tool call (last one added)
                        if tool_calls_acc:
                            last_id = list(tool_calls_acc)[-1]
                            tool_calls_acc[last_id]["input_str"] += delta.get("partial_json", "")

                elif etype == "message_delta":
                    stop = event.get("delta", {}).get("stop_reason")
                    if stop == "tool_use":
                        for tc in tool_calls_acc.values():
                            try:
                                inp = json.loads(tc["input_str"]) if tc["input_str"] else {}
                            except json.JSONDecodeError:
                                inp = {}
                            yield {"tool_call": {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(inp),
                                },
                            }}
                        tool_calls_acc = {}


# ---------------------------------------------------------------------------
# MiniMax streaming (OpenAI-compatible)
# ---------------------------------------------------------------------------

def _stream_minimax(messages: list, model_id: str) -> Generator[Union[str, dict], None, None]:
    payload = {
        "model": model_id,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "tools": TOOL_SCHEMAS,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }
    tool_calls_acc = {}

    with httpx.Client(timeout=120) as client:
        with client.stream("POST", f"{MINIMAX_API_BASE}/text/chatcompletion_v2",
                           headers=headers, json=payload) as response:
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

                if delta.get("content"):
                    yield delta["content"]

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

                if choices[0].get("finish_reason") == "tool_calls":
                    for tc in tool_calls_acc.values():
                        yield {"tool_call": tc}
                    tool_calls_acc = {}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def stream_response(messages: list, model_key: str) -> Generator[Union[str, dict], None, None]:
    cfg = MODELS[model_key]
    if cfg["provider"] == "anthropic":
        yield from _stream_anthropic(messages, cfg["model_id"])
    else:
        yield from _stream_minimax(messages, cfg["model_id"])


def run_turn(messages: list, model_key: str, on_text=None, on_tool_start=None, on_tool_result=None) -> list:
    messages = list(messages)

    while True:
        text_parts = []
        tool_calls_in_turn = []

        for item in stream_response(messages, model_key):
            if isinstance(item, str):
                text_parts.append(item)
                if on_text:
                    on_text(item)
            elif isinstance(item, dict) and "tool_call" in item:
                tool_calls_in_turn.append(item["tool_call"])

        full_text = "".join(text_parts)
        assistant_msg = {"role": "assistant"}

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
