# Production Container for ReviewPulse Sentiment & GenAI Review Insights
FROM python:3.12-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and production models
COPY pyproject.toml .
COPY src/ src/
COPY models/ models/
COPY .env.example .env

# Install package in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["python", "src/reviewpulse/predict.py"]
