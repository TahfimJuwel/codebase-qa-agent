"""
Configuration settings for the Codebase Q&A Agent.
Centralizes repository settings, model definitions, and storage paths.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPOS_DIR = DATA_DIR / "repos"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# Ensure runtime directories exist
DATA_DIR.mkdir(exist_ok=True)
REPOS_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

# Target Repository Details (Chosen: psf/requests)
REPO_URL = "https://github.com/psf/requests.git"
REPO_NAME = "requests"
REPO_PATH = REPOS_DIR / REPO_NAME

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Using gemini-2.0-flash (fast, structured tool calling, generous free-tier limits)
LLM_MODEL = "gemini-2.0-flash"

# Embedding Model Configuration
# all-MiniLM-L6-v2 produces 384-dimensional vectors; fast on local CPU
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"