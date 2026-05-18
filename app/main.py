import argparse
import json
import os
from urllib import error, request

from app.rag.FAISS.load import RAGManager
from logger.trace_logger import TraceLogger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lekkie serwisy — ładują się szybko przy każdym --reload
rag_manager = RAGManager()
trace_logger = TraceLogger()

model_service_url = os.getenv("MODEL_SERVICE_URL", "http://127.0.0.1:8001").rstrip("/")
model_stream_endpoint = f"{model_service_url}/generate-stream"


class Message(BaseModel):
    text: str


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
            "Model service niedostępny. Uruchom: python -m app.endpoints.model_service"
        ) from exc


@app.post("/generate")
async def generate(message: Message):
    if not message.text:
        raise HTTPException(status_code=400, detail="Wiadomość nie może być pusta")

    data = rag_manager.get_context(message.text, k=5)
    context_text = "\n\n".join([chunk["content"] for chunk in data])

    try:
        stream_generator = stream_from_model_service(message.text, context_text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def stream():
        full_response = []
        for token in stream_generator:
            full_response.append(token)
            yield token

        trace_logger.log({
            "query": message.text,
            "retrival": [
                {
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "score": chunk["score"],
                }
                for chunk in data
            ],
            "response": "".join(full_response),
        })

    return StreamingResponse(stream(), media_type="text/plain")


def run_cli() -> None:
    print("Tryb konsolowy. Wpisz 'exit' aby zakończyć.")
    print(f"Model service: {model_service_url}")

    while True:
        user_query = input("Ty: ").strip()

        if user_query.lower() in {"exit", "quit", "q"}:
            print("Koniec.")
            break

        if not user_query:
            print("Wiadomość nie może być pusta.")
            continue

        data = rag_manager.get_context(user_query, k=5)
        context_text = "\n\n".join([chunk["content"] for chunk in data])

        try:
            stream_generator = stream_from_model_service(user_query, context_text)
        except RuntimeError as exc:
            print(f"Błąd: {exc}")
            continue

        full_response = []
        print("AI: ", end="", flush=True)
        for token in stream_generator:
            full_response.append(token)
            print(token, end="", flush=True)
        print()

        trace_logger.log({
            "query": user_query,
            "retrival": [
                {"content": c["content"], "metadata": c["metadata"], "score": c["score"]}
                for c in data
            ],
            "response": "".join(full_response),
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backend API i tryb konsolowy")
    parser.add_argument("--cli", action="store_true", help="Tryb konsolowy bez frontendu.")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        print("Uruchom API przez uvicorn lub podaj --cli")
