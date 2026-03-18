from app.services.genAI.rag.FAISS.load import RAGManager
from app.services.logger.trace_logger import TraceLogger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.services.genAI.huggingface_pipeline import TextGenerator
from fastapi.responses import StreamingResponse
from app.utils import truncate_words

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

text_generator = TextGenerator()  # Przykładowy prompt, można go dostosować do różnych scenariuszy
rag_manager = RAGManager()
trace_logger = TraceLogger()  

class Message(BaseModel):
    text: str

@app.post("/generate")
async def generate(message: Message):
    if not message.text:
        raise HTTPException(status_code=400, detail="Message text cannot be empty")
    
    data = rag_manager.get_context(message.text, k=5)

    context_text = "\n\n".join([chunk['content'] for chunk in data])  # Łączenie treści z kontekstu

    stream_generator = text_generator.generate_stream(message.text, context_text)  # Przekaż kontekst do generatora tekstu

    # logger based on relevant_chunks and metadata

    async def stream():
        full_response = []

        for token in stream_generator:
            full_response.append(token) 
            yield token

        trace_logger.log({
            "query": message.text,
            "retrival": [
                {
                    "content": chunk['content'],
                    "metadata": chunk['metadata'],
                    "score": chunk['score']
                }
                for chunk in data
            ],
            "response": "".join(full_response)
        })

        

    return StreamingResponse(stream(), media_type="text/plain")
