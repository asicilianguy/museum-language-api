"""
Training del modello con Pipeline completo
"""
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r"[^\w\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Caricamento dataset
print("Caricamento dataset...")
url = "https://raw.githubusercontent.com/Profession-AI/progetti-ml/refs/heads/main/Modello%20per%20l'identificazione%20della%20lingua%20dei%20testi%20di%20un%20museo/museo_descrizioni.csv"
df = pd.read_csv(url)

# Preprocessing
print("Preprocessing...")
df['Testo_Pulito'] = df['Testo'].apply(preprocess_text)

# Crea PIPELINE COMPLETO
print("Creazione pipeline...")
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 3),
        lowercase=False,
        min_df=1,
        max_df=1.0
    )),
    ('classifier', MultinomialNB())
])

# Split
X = df['Testo_Pulito']
y = df['Codice Lingua']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

# Training
print("Training pipeline...")
pipeline.fit(X_train, y_train)

# Valutazione
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuratezza Test: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Salva UN SOLO FILE
print("\nSalvataggio pipeline...")
with open('api/language_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("✅ Pipeline salvato: api/language_pipeline.pkl")
print(f"Dimensione: {os.path.getsize('api/language_pipeline.pkl') / 1024:.1f} KB")
