"""FastAPI entrypoint. Run with: uvicorn src.app:app --reload"""

from fastapi import FastAPI

from src.whatsapp.webhook import router as whatsapp_router

app = FastAPI(title="Business Intake Agent")
app.include_router(whatsapp_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
