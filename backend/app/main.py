from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import weather

app = FastAPI(title="VentiGlobe API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(weather.router, prefix="/api/weather", tags=["weather"])

@app.get("/")
async def root():
    return {"message": "Welcome to VentiGlobe API"} 