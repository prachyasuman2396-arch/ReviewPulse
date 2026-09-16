import os
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Ensure package root is in sys.path
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from reviewpulse.predict import predict, load_production_model
from reviewpulse.insights import generate_amazon_review_insights
from reviewpulse.config import is_groq_configured, get_groq_model

app = FastAPI(
    title="ReviewPulse API",
    description="Production Sentiment Classification & Amazon-Style GenAI Review Highlights API",
    version="1.0.0",
)


# --- Request & Response Schemas ---
class PredictRequest(BaseModel):
    reviews: List[str] = Field(
        ...,
        min_items=1,
        example=[
            "This product is absolutely amazing! Exceeded all my expectations.",
            "Terrible purchase. It broke within 2 days and customer service was unhelpful."
        ]
    )
    from_registry: bool = Field(False, description="Load model dynamically from MLflow registry")


class PredictionItem(BaseModel):
    text: str
    label: int
    sentiment: str
    confidence: Optional[float] = None
    positive_probability: Optional[float] = None


class PredictResponse(BaseModel):
    total: int
    predictions: List[PredictionItem]


class InsightsRequest(BaseModel):
    reviews: List[str] = Field(..., min_items=1, example=[
        "Great CD: Beautiful vocals and music.",
        "Batteries died within a year ... would not hold a charge.",
        "Works fine, but competitor charger is faster."
    ])
    product_id: Optional[str] = Field(None, example="PROD-B001")
    max_reviews: int = Field(40, description="Max reviews to sample for LLM context")


class HealthResponse(BaseModel):
    status: str
    production_model_available: bool
    groq_configured: bool
    groq_model: str


# --- Endpoints ---
@app.get("/", tags=["General"])
def root():
    return {
        "service": "ReviewPulse",
        "description": "Production Sentiment Classification & Amazon-Style GenAI Review Highlights",
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    model_ok = False
    try:
        load_production_model()
        model_ok = True
    except Exception:
        pass

    return {
        "status": "healthy" if model_ok else "degraded",
        "production_model_available": model_ok,
        "groq_configured": is_groq_configured(),
        "groq_model": get_groq_model(),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def score_reviews(payload: PredictRequest):
    """
    Classifies review sentiment into Positive (1) or Negative (0)
    along with calibrated confidence scores.
    """
    try:
        results = predict(payload.reviews, from_registry=payload.from_registry)
        return PredictResponse(total=len(results), predictions=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insights", tags=["GenAI Insights"])
def generate_insights(payload: InsightsRequest):
    """
    Synthesizes customer reviews into an Amazon-style Customer Review Highlights card:
    - Executive summary ('Customers say...')
    - Estimated star rating
    - What customers liked (with quotes)
    - What customers disliked (with quotes)
    - Aspect-level sentiment breakdown
    """
    try:
        insights = generate_amazon_review_insights(
            reviews=payload.reviews,
            product_id=payload.product_id,
            max_reviews_to_sample=payload.max_reviews,
        )
        return insights
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
