FROM python:3.9-slim

WORKDIR /app

# Instalacja zależności systemowych
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Kopiowanie plików projektu
COPY requirements.txt .
COPY app/ ./app/

# Instalacja zależności Pythona
RUN pip install --no-cache-dir -r requirements.txt

# Tworzenie katalogów na dane i modele
RUN mkdir -p /app/data /app/models

# Ustawienie zmiennych środowiskowych
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Komenda startowa
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"] 