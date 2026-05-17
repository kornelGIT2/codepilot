import argparse

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.genAI.huggingface_pipeline import TextGenerator

app = FastAPI(title="Model Service")
text_generator = TextGenerator()


class GenerationRequest(BaseModel):
    prompt: str
    context: str = ""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-stream")
async def generate_stream(message: GenerationRequest):
    def stream():
        for token in text_generator.generate_stream(message.prompt, message.context):
            yield token

    return StreamingResponse(stream(), media_type="text/plain")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dedicated model service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    uvicorn.run("app.model_service:app", host=args.host, port=args.port)
