from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from .dlp_core import score_text, score_dataframe

app = FastAPI(title="DLP XGBoost Server")

class TextIn(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/analyze-text")
def analyze_text(payload: TextIn):
    return score_text(payload.text)

@app.post("/upload-csv")
def upload_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    if "text" not in df.columns:
        return {"error":"CSV must contain a 'text' column"}
    results = score_dataframe(df)
    df_out = df.copy()
    df_out["class"]  = [r["class"] for r in results]
    df_out["score"]  = [r["score"] for r in results]
    df_out["policy"] = [r["policy"] for r in results]
    df_out["phones"] = [", ".join(r["phones"]) for r in results]
    return {
        "rows": len(df_out),
        "preview": df_out.head(25).to_dict(orient="records"),
        "summary": df_out["policy"].value_counts().to_dict()
    }
