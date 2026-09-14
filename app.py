import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pipeline import SupportChatbotPipeline

app = FastAPI(title="E-Commerce Support RAG Bot")

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

pipeline = None

@app.on_event("startup")
def startup_event():
    global pipeline
    pipeline = SupportChatbotPipeline(models_dir="models")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def serve_ui():
    return FileResponse("templates/index.html")

@app.get("/api/health")
async def health_check():
    global pipeline
    ready = pipeline is not None
    return {
        "status": "online" if ready else "initializing",
        "models_ready": ready
    }

@app.post("/api/chat")
async def handle_chat(payload: ChatRequest):
    global pipeline
    if not pipeline:
        pipeline = SupportChatbotPipeline(models_dir="models")
        
    result = pipeline.process_query(payload.message)
    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
