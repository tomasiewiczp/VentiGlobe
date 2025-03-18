# VentiGlobe

Aplikacja do przewidywania pogody i rekomendacji miejsc podróży wykorzystująca uczenie maszynowe.

## Funkcjonalności

- Przewidywanie pogody dla wybranego miasta i daty
- Rekomendacje miejsc podróży na podstawie preferencji pogodowych
- Interaktywna mapa z wyszukiwarką miejsc

## Technologie

- Backend: Python (FastAPI)
- Frontend: React.js
- Baza danych: PostgreSQL
- ML: scikit-learn
- Konteneryzacja: Docker & Docker Compose

## Instalacja

1. Sklonuj repozytorium:
```bash
git clone https://github.com/tomasiewiczp/VentiGlobe.git
cd VentiGlobe
```

2. Utwórz i aktywuj virtualenv:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# lub
.\venv\Scripts\activate  # Windows
```

3. Zainstaluj zależności:
```bash
pip install -r backend/requirements.txt
```

4. Uruchom aplikację:
```bash
docker-compose up
```

## Rozwój

- Backend dostępny na: http://localhost:8000
- Frontend dostępny na: http://localhost:3000
- API dokumentacja: http://localhost:8000/docs

## Licencja

MIT 