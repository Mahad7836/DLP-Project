from fastapi import FastAPI, UploadFile, File
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import pandas as pd
from .dlp_core import score_text, score_dataframe

app = FastAPI(title="DLP XGBoost Server")


@app.get("/")
def root():
    return RedirectResponse(url="/docs")

class TextIn(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze-text")
def analyze_text(payload: TextIn):
    return score_text(payload.text)

@app.post("/upload-csv")
def upload_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    if "text" not in df.columns:
        return {"error": "CSV must contain a 'text' column"}
    results = score_dataframe(df)
    out = []
    for i, (row, r) in enumerate(zip(df.itertuples(index=False), results), start=1):
        out.append({
            "row": i,
            "text": str(getattr(row, "text", ""))[:140],
            "class": r["class"],
            "score": r["score"],
            "policy": r["policy"],
            "phones": ", ".join(r.get("phones", []))
        })
    summary = {}
    for o in out:
        summary[o["policy"]] = summary.get(o["policy"], 0) + 1
    return {"rows": len(out), "summary": summary, "preview": out[:25]}
