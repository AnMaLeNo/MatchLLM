FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
ARG TORCH_VERSION=2.12.0+cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --extra-index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Exposer le port de l'API
EXPOSE 8000

# Commande de lancement
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
