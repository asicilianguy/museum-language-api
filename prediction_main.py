"""
API per Language Detection - MuseumLangID

API REST che carica il modello pre-addestrato e serve un endpoint
per identificare la lingua di testi museali.
"""

from typing_extensions import Annotated
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import pickle
import logging
import re
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('museum_lang_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Funzione di preprocessing (identica a train.py)
def preprocess_text(text):
    """Preprocessing per language detection"""
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r"[^\w\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Schema input
class TextInput(BaseModel):
    model_config = {"extra": "forbid"}

    text: str = Field(..., min_length=1, max_length=10000, description='Testo da analizzare')


# Schema output
class LanguageOutput(BaseModel):
    language_code: str = Field(..., description='Codice ISO della lingua')
    confidence: float = Field(..., description='Confidenza della predizione')


# Caricamento modello e vectorizer
logger.info("Caricamento modello...")
with open('language_model.pkl', 'rb') as f:
    model = pickle.load(f)
logger.info("Modello caricato")

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
logger.info("Vectorizer caricato")

app = FastAPI(
    title="MuseumLangID API",
    description="API per identificazione automatica della lingua di testi museali",
    version="1.0.0"
)


@app.get("/identify-language")
def identify_language(param_query: Annotated[TextInput, Query()]) -> LanguageOutput:
    """
    Identifica la lingua di un testo (IT, EN, DE)
    """
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

    logger.info(f"[{request_id}] RICHIESTA - Testo: '{param_query.text[:50]}...' (lunghezza: {len(param_query.text)})")

    # Validazione testo non vuoto
    if not param_query.text.strip():
        error_msg = "Il testo non può essere vuoto o contenere solo spazi"
        logger.warning(f"[{request_id}] ERRORE - {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)

    try:
        # Preprocessing
        text_clean = preprocess_text(param_query.text)

        # Transform con vectorizer
        text_vector = vectorizer.transform([text_clean])

        # Predizione
        predicted_language = model.predict(text_vector)
        language_code = predicted_language[0].upper()

        # Confidenza
        predicted_proba = model.predict_proba(text_vector)[0]
        confidence = float(max(predicted_proba))

        # Risposta
        result = LanguageOutput(
            language_code=language_code,
            confidence=round(confidence, 2)
        )

        logger.info(f"[{request_id}] RISPOSTA - Lingua: {result.language_code}, Confidenza: {result.confidence}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Errore durante l'identificazione: {str(e)}"
        logger.error(f"[{request_id}] ERRORE - {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


if __name__ == '__main__':
    uvicorn.run("prediction_main:app", host="0.0.0.0", port=8000)