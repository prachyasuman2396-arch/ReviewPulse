import os
from pathlib import Path

# Look for .env in project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"


def _load_env_file(filepath: Path):
    """Fallback .env parser if python-dotenv is not installed."""
    if not filepath.exists():
        return
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


# Try python-dotenv first, fallback to manual parser
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ENV_PATH)
except ImportError:
    _load_env_file(ENV_PATH)

# Configuration settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq/compound-mini").strip()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000").strip()


def get_groq_api_key() -> str:
    """Returns the Groq API key from environment."""
    return os.getenv("GROQ_API_KEY", "").strip()


def get_groq_model() -> str:
    """Returns the configured Groq model ID."""
    return os.getenv("GROQ_MODEL", "groq/compound-mini").strip()


def is_groq_configured() -> bool:
    """Checks if a non-placeholder Groq API key is set."""
    key = get_groq_api_key()
    return bool(key and key != "your_groq_api_key_here")
