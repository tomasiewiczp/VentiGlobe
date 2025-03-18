# VentiGlobe

A weather prediction and travel destination recommendation application using machine learning.

## Features

- Weather prediction for selected city and date
- Travel destination recommendations based on weather preferences
- Interactive map with location search

## Technologies

- Backend: Python (FastAPI)
- Frontend: React.js
- Database: PostgreSQL
- ML: scikit-learn
- Containerization: Docker & Docker Compose

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tomasiewiczp/VentiGlobe.git
cd VentiGlobe
```

2. Create and activate virtualenv:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r backend/requirements.txt
```

4. Run the application:
```bash
docker-compose up
```

## Development

- Backend available at: http://localhost:8000
- Frontend available at: http://localhost:3000
- API documentation: http://localhost:8000/docs

## License

MIT 