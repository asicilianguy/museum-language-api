from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
import pickle
import logging
import re
from datetime import datetime
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r"[^\w\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class LanguageOutput(BaseModel):
    language_code: str = Field(..., description='Codice ISO della lingua')
    confidence: float = Field(..., description='Confidenza della predizione')


logger.info("Caricamento modello...")


possible_paths = [
    'api/language_model.pkl',
    'language_model.pkl',
    '/var/task/api/language_model.pkl'
]

model = None
vectorizer = None

for base_path in possible_paths:
    model_path = base_path
    vectorizer_path = base_path.replace('language_model', 'vectorizer')
    
    try:
        logger.info(f"Tentativo path: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        logger.info(f"✅ Modello caricato da: {model_path}")
        break
    except FileNotFoundError:
        continue
    except Exception as e:
        logger.error(f"Errore caricamento da {model_path}: {e}")
        continue

if model is None or vectorizer is None:
    logger.error("❌ Impossibile caricare i modelli da nessun path")


app = FastAPI(
    title="MuseumLangID API",
    description="API per identificazione automatica della lingua",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "MuseumLangID API",
        "version": "1.0.0",
        "endpoint": "GET /identify-language?text=...",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }

@app.get("/identify-language")
def identify_language(text: str = Query(..., min_length=1, max_length=10000)) -> LanguageOutput:
    """Identifica la lingua di un testo (IT, EN, DE)"""
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    
    logger.info(f"[{request_id}] RICHIESTA - Testo: '{text[:50]}...'")
    
    
    if model is None or vectorizer is None:
        logger.error("Modello non disponibile")
        raise HTTPException(status_code=503, detail="Modello non disponibile")
    
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Il testo non può essere vuoto")
    
    try:
        
        text_clean = preprocess_text(text)
        text_vector = vectorizer.transform([text_clean])
        predicted_language = model.predict(text_vector)
        language_code = predicted_language[0].upper()
        
        
        predicted_proba = model.predict_proba(text_vector)[0]
        confidence = float(max(predicted_proba))
        
        result = LanguageOutput(
            language_code=language_code,
            confidence=round(confidence, 2)
        )
        
        logger.info(f"[{request_id}] RISPOSTA - Lingua: {result.language_code}, Confidenza: {result.confidence}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Errore: {str(e)}"
        logger.error(f"[{request_id}] ERRORE - {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("prediction_main:app", host="0.0.0.0", port=8000)
