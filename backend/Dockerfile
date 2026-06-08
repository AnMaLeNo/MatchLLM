FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optionnel mais recommandé : Précharger le modèle HuggingFace pour accélérer le démarrage du conteneur API
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY . .

# Exposer le port de l'API
EXPOSE 8000

# Commande de lancement
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
