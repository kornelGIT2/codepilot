import argparse
import json
import os
from urllib import error, request

from app.services.genAI.rag.FAISS.load import RAGManager
from app.services.logger.trace_logger import TraceLogger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_manager = RAGManager()
trace_logger = TraceLogger()  
model_service_url = os.getenv("MODEL_SERVICE_URL", "http://127.0.0.1:8001").rstrip("/")
model_stream_endpoint = f"{model_service_url}/generate-stream"

class Message(BaseModel):
    text: str


def prepare_generation_data(query: str):
    data = rag_manager.get_context(query, k=5)
    context_text = "\n\n".join([chunk["content"] for chunk in data])
    stream_generator = stream_from_model_service(query, context_text)
    return data, stream_generator


def stream_from_model_service(prompt: str, context: str):
    payload = json.dumps({"prompt": prompt, "context": context}).encode("utf-8")
    req = request.Request(
        model_stream_endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=300) as response:
            while True:
                chunk = response.read(1024)
                if not chunk:
                    break
                yield chunk.decode("utf-8", errors="ignore")
    except error.URLError as exc:
        raise RuntimeError(
            "Model service is unavailable. Run: python -m app.model_service"
        ) from exc


def log_trace(query: str, data, response: str) -> None:
    trace_logger.log(
        {
            "query": query,
            "retrival": [
                {
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "score": chunk["score"],
                }
                for chunk in data
            ],
            "response": response,
        }
    )

@app.post("/generate")
async def generate(message: Message):
    if not message.text:
        raise HTTPException(status_code=400, detail="Message text cannot be empty")

    try:
        data, stream_generator = prepare_generation_data(message.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # logger based on relevant_chunks and metadata

    async def stream():
        full_response = []

        for token in stream_generator:
            full_response.append(token)
            yield token

        response_text = "".join(full_response)
        log_trace(message.text, data, response_text)
        print(response_text, flush=True)

    return StreamingResponse(stream(), media_type="text/plain")


def run_cli() -> None:
    print("Tryb konsolowy uruchomiony. Wpisz 'exit', aby zakonczyc.")
    print(f"Model service: {model_service_url}")
    while True:
        user_query = input("Ty: ").strip()

        if user_query.lower() in {"exit", "quit", "q"}:
            print("Koniec.")
            break

        if not user_query:
            print("Wiadomosc nie moze byc pusta.")
            continue

        try:
            data, stream_generator = prepare_generation_data(user_query)
        except RuntimeError as exc:
            print(f"Blad: {exc}")
            continue

        full_response = []

        print("AI: ", end="", flush=True)
        for token in stream_generator:
            full_response.append(token)
            print(token, end="", flush=True)
        print()

        log_trace(user_query, data, "".join(full_response))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backend API i tryb konsolowy")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Uruchamia tryb konsolowy bez frontendu.",
    )
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        print("Uruchom API przez uvicorn lub podaj --cli, aby pracowac bez frontendu.")
