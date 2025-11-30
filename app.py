# app.py
# Thin adapter so Vercel can find your FastAPI app

from grader_backend.main import app  # re-export the existing FastAPI app
