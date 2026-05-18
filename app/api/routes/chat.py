# app/api/routes/chat.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.graph.graph import build_graph

router = APIRouter()
_graph = build_graph()  # singleton na poziomie modułu

class Message(BaseModel):
    text: str

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/generate")
async def generate(message: Message):
    result = _graph.invoke({"question": message.text, "context": "", "answer": "", "feedback": "", "iterations": 0})
    return {"answer": result["answer"]}

@router.post("/generate-stream")
async def generate_stream(message: Message):
    async def stream():
        async for event in _graph.astream_events(
            {"question": message.text, "context": "", "answer": "", "feedback": "", "iterations": 0},
            version="v2"
        ):
            if event["event"] == "on_chain_stream" and event.get("name") == "generate":
                chunk = event["data"].get("chunk", {})
                if token := chunk.get("answer", ""):
                    yield token

    return StreamingResponse(stream(), media_type="text/plain")