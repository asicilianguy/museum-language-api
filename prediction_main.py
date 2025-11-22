from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
import pickle
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r"[^\w\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class LanguageOutput(BaseModel):
    language_code: str
    confidence: float

logger.info("Caricamento pipeline...")
try:
    with open('api/language_pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    logger.info("✅ Pipeline caricato")
except Exception as e:
    logger.error(f"❌ Errore: {e}")
    pipeline = None

app = FastAPI(title="MuseumLangID API", version="1.0.0")

@app.get("/")
def root():
    return {
        "message": "MuseumLangID API",
        "version": "1.0.0",
        "pipeline_loaded": pipeline is not None
    }

@app.get("/identify-language")
def identify_language(text: str = Query(..., min_length=1, max_length=10000)) -> LanguageOutput:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline non disponibile")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Testo vuoto")
    
    try:
        text_clean = preprocess_text(text)
        predicted = pipeline.predict([text_clean])
        language_code = predicted[0].upper()
        proba = pipeline.predict_proba([text_clean])[0]
        confidence = float(max(proba))
        
        return LanguageOutput(
            language_code=language_code,
            confidence=round(confidence, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("prediction_main:app", host="0.0.0.0", port=8000)
