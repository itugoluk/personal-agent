import os
import re
import subprocess
from config import MAX_OUTPUT_CHARS, SANDBOX_DIR

try:
    from ddgs import DDGS
    _ddgs_available = True
except ImportError:
    _ddgs_available = False

# Ensure sandbox exists
os.makedirs(SANDBOX_DIR, exist_ok=True)


# ── Sandbox enforcement ───────────────────────────────────────────────────────

def _safe_path(path: str) -> str:
    """
    Resolve the path and verify it sits inside SANDBOX_DIR.
    Raises PermissionError if it does not.
    """
    resolved = os.path.realpath(os.path.expanduser(path))
    if resolved != SANDBOX_DIR and not resolved.startswith(SANDBOX_DIR + os.sep):
        raise PermissionError(
            f"Access denied: '{path}' is outside the sandbox. "
            f"All file operations are restricted to {SANDBOX_DIR}"
        )
    return resolved


def _shell_cmd_safe(cmd: str) -> None:
    """
    Reject shell commands that reference absolute paths outside the sandbox.
    Only flags tokens that both look like paths AND exist on the filesystem,
    avoiding false positives on format strings, sed patterns, etc.
    """
    candidates = re.findall(r'(?:~\/|\/)[^\s\'\";|&><$(){}\\]*', cmd)
    for c in candidates:
        expanded = os.path.realpath(os.path.expanduser(c))
        # Only block if the path (or its parent) actually exists outside sandbox
        check = expanded
        if not os.path.exists(check):
            check = os.path.dirname(expanded)
        if not os.path.exists(check):
            continue  # doesn't exist — not a real path, skip
        if check != SANDBOX_DIR and not check.startswith(SANDBOX_DIR + os.sep):
            raise PermissionError(
                f"Access denied: shell command references a path outside the sandbox ('{c}'). "
                f"All operations are restricted to {SANDBOX_DIR}"
            )


def _cap(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n[...truncated at {MAX_OUTPUT_CHARS} chars]"
    return text


# ── Tools ─────────────────────────────────────────────────────────────────────

_SENSITIVE_ENV_KEYS = {"ANTHROPIC_API_KEY", "GROQ_API_KEY", "MINIMAX_API_KEY", "OPENAI_API_KEY"}

def shell_exec(cmd: str) -> str:
    try:
        _shell_cmd_safe(cmd)
    except PermissionError as e:
        return f"SANDBOX ERROR: {e}"
    try:
        env = {k: v for k, v in os.environ.items() if k not in _SENSITIVE_ENV_KEYS}
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30,
            cwd=SANDBOX_DIR, env=env
        )
        out = result.stdout + (("\nSTDERR:\n" + result.stderr) if result.stderr else "")
        return _cap(out.strip()) or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {e}"


def read_file(path: str) -> str:
    try:
        safe = _safe_path(path)
    except PermissionError as e:
        return f"SANDBOX ERROR: {e}"
    try:
        with open(safe, "r", encoding="utf-8", errors="replace") as f:
            return _cap(f.read())
    except Exception as e:
        return f"Error: {e}"


def write_file(path: str, content: str) -> str:
    try:
        safe = _safe_path(path)
    except PermissionError as e:
        return f"SANDBOX ERROR: {e}"
    try:
        os.makedirs(os.path.dirname(safe), exist_ok=True)
        with open(safe, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} chars to {safe}"
    except Exception as e:
        return f"Error: {e}"


def list_dir(path: str = ".") -> str:
    # Default to sandbox if relative or unspecified
    if not os.path.isabs(os.path.expanduser(path)) and path in (".", ""):
        path = SANDBOX_DIR
    try:
        safe = _safe_path(path)
    except PermissionError as e:
        return f"SANDBOX ERROR: {e}"
    try:
        entries = os.listdir(safe)
        lines = []
        for e in sorted(entries):
            full = os.path.join(safe, e)
            lines.append(e + ("/" if os.path.isdir(full) else ""))
        return "\n".join(lines) or "(empty sandbox)"
    except Exception as e:
        return f"Error: {e}"


def web_search(query: str) -> str:
    if not _ddgs_available:
        return "SEARCH FAILED. Do not retry or search again. Tell the user the search failed and ask if they would like to try something different. (Error: duckduckgo-search not installed)"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "SEARCH FAILED. Do not retry or search again. Tell the user the search returned no results and ask if they would like to try something different."
        lines = []
        for r in results:
            lines.append(f"**{r.get('title', '')}**\n{r.get('href', '')}\n{r.get('body', '')}\n")
        return _cap("\n".join(lines))
    except Exception as e:
        return f"SEARCH FAILED. Do not retry or search again. Tell the user the search failed and ask if they would like to try something different. (Error: {e})"


# ── Registry ──────────────────────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "shell_exec": shell_exec,
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "web_search": web_search,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": f"Run a shell command. Runs inside the sandbox ({SANDBOX_DIR}). Cannot access any paths outside the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to run"}
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": f"Read a file. Only files inside the sandbox ({SANDBOX_DIR}) may be read.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": f"Path inside the sandbox, e.g. {SANDBOX_DIR}/file.py"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": f"Write a file. Only files inside the sandbox ({SANDBOX_DIR}) may be written.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": f"Path inside the sandbox"},
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
            "description": f"List a directory. Only the sandbox ({SANDBOX_DIR}) and subdirectories within it may be listed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (defaults to sandbox root)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
]


def dispatch(name: str, args: dict) -> str:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    return fn(**args)
