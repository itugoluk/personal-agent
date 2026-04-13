import os
import subprocess
import json
from config import MAX_OUTPUT_CHARS

try:
    from duckduckgo_search import DDGS
    _ddgs_available = True
except ImportError:
    _ddgs_available = False


def _cap(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n[...truncated at {MAX_OUTPUT_CHARS} chars]"
    return text


def shell_exec(cmd: str) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        out = result.stdout + (("\nSTDERR:\n" + result.stderr) if result.stderr else "")
        return _cap(out.strip()) or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"


def read_file(path: str) -> str:
    path = os.path.expanduser(path)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return _cap(f.read())
    except Exception as e:
        return f"Error: {e}"


def write_file(path: str, content: str) -> str:
    path = os.path.expanduser(path)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"Error: {e}"


def list_dir(path: str = ".") -> str:
    path = os.path.expanduser(path)
    try:
        entries = os.listdir(path)
        lines = []
        for e in sorted(entries):
            full = os.path.join(path, e)
            suffix = "/" if os.path.isdir(full) else ""
            lines.append(e + suffix)
        return "\n".join(lines) or "(empty)"
    except Exception as e:
        return f"Error: {e}"


def web_search(query: str) -> str:
    if not _ddgs_available:
        return "Error: duckduckgo-search not installed"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found"
        lines = []
        for r in results:
            lines.append(f"**{r.get('title', '')}**\n{r.get('href', '')}\n{r.get('body', '')}\n")
        return _cap("\n".join(lines))
    except Exception as e:
        return f"Error: {e}"


def python_exec(code: str) -> str:
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=30
        )
        out = result.stdout + (("\nSTDERR:\n" + result.stderr) if result.stderr else "")
        return _cap(out.strip()) or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"


TOOL_FUNCTIONS = {
    "shell_exec": shell_exec,
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "web_search": web_search,
    "python_exec": python_exec,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": "Run a shell command on the Mac and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "The shell command to run"}
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or ~ path to the file"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it if it doesn't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or ~ path to the file"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: current directory)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo and return the top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": "Execute Python 3 code and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    },
]


def dispatch(name: str, args: dict) -> str:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    return fn(**args)
