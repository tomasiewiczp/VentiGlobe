from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import weather

app = FastAPI(title="VentiGlobe Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(weather.router, prefix="/api/weather", tags=["weather"])

@app.get("/")
async def root():
    return {"message": "VentiGlobe Backend is running"} 