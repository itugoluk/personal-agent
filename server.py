import json
import asyncio
import queue
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import memory
import agent
from config import MODELS, DEFAULT_MODEL, ANTHROPIC_API_KEY, MINIMAX_API_KEY, GROQ_API_KEY

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)

state = {
    "model_key": DEFAULT_MODEL,
    "messages": memory.load(),
}


def _display_history(messages: list) -> list:
    """Convert internal message format to a flat display list for the frontend."""
    tool_results = {}
    for msg in messages:
        if msg["role"] == "tool":
            tool_results[msg["tool_call_id"]] = msg["content"]

    result = []
    for msg in messages:
        role = msg["role"]
        if role == "user":
            result.append({"type": "user", "content": msg["content"]})
        elif role == "assistant":
            for tc in msg.get("tool_calls", []):
                try:
                    args = json.loads(tc["function"]["arguments"])
                except Exception:
                    args = {}
                result.append({
                    "type": "tool",
                    "name": tc["function"]["name"],
                    "args": args,
                    "result": tool_results.get(tc["id"], ""),
                })
            if msg.get("content"):
                result.append({"type": "assistant", "content": msg["content"]})
    return result


class MessageRequest(BaseModel):
    content: str


class ModelRequest(BaseModel):
    model_key: str


@app.get("/")
async def index():
    with open("frontend/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/models")
async def get_models():
    return {
        "current": state["model_key"],
        "models": {
            k: {
                "label": v["label"],
                "has_key": bool(
                    ANTHROPIC_API_KEY if v["provider"] == "anthropic"
                    else GROQ_API_KEY if v["provider"] == "groq"
                    else MINIMAX_API_KEY
                ),
            }
            for k, v in MODELS.items()
        },
    }


@app.get("/api/history")
async def get_history():
    return {"messages": _display_history(state["messages"])}


@app.post("/api/new")
async def new_chat():
    state["messages"] = []
    memory.clear()
    return {"ok": True}


@app.post("/api/model")
async def set_model(req: ModelRequest):
    if req.model_key not in MODELS:
        return {"error": "Unknown model"}
    cfg = MODELS[req.model_key]
    has_key = bool(
        ANTHROPIC_API_KEY if cfg["provider"] == "anthropic"
        else GROQ_API_KEY if cfg["provider"] == "groq"
        else MINIMAX_API_KEY
    )
    if not has_key:
        return {"error": f"No API key for {cfg['label']}"}
    state["model_key"] = req.model_key
    return {"ok": True}


@app.post("/api/chat")
async def chat(req: MessageRequest):
    state["messages"].append({"role": "user", "content": req.content})
    event_queue: queue.Queue = queue.Queue()

    def run():
        def on_text(chunk):
            event_queue.put(("text", chunk))

        def on_tool_start(name, args):
            event_queue.put(("tool_start", {"name": name, "args": args}))

        def on_tool_result(name, result):
            event_queue.put(("tool_result", {"name": name, "result": result}))

        try:
            updated = agent.run_turn(
                state["messages"],
                model_key=state["model_key"],
                on_text=on_text,
                on_tool_start=on_tool_start,
                on_tool_result=on_tool_result,
            )
            state["messages"] = updated
            memory.save(updated)
            event_queue.put(("done", None))
        except Exception as e:
            event_queue.put(("error", str(e)))
        finally:
            event_queue.put(None)  # sentinel

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, run)

    async def generate():
        while True:
            try:
                item = event_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue
            if item is None:
                break
            event_type, data = item
            yield f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
