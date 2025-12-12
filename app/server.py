from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    # 1. Read CSV
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    if "text" not in df.columns:
        return {"error": "CSV must contain a 'text' column"}

    # TEMP: return dummy response (to test fetch)
    preview = [
        {
            "text": t,
            "class": "none",
            "score": 0.0,
            "policy": "ALLOW",
            "entities": []
        }
        for t in df["text"].head(5)
    ]

    return {
        "preview": preview,
        "summary": {"ALLOW": len(preview)}
    }
