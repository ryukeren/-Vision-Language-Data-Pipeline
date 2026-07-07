"""
Vision-Language Data Engineering Pipeline
Main FastAPI application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Vision-Language Pipeline API",
    description="Enterprise-grade Vision-Language Data Engineering Pipeline",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Vision-Language Pipeline API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
