# 🌟 ReviewPulse

> **Production Sentiment Analysis & Amazon-Style GenAI Review Insights Platform**

ReviewPulse is an end-to-end MLOps and Generative AI system that classifies customer review sentiment, tracks model experimentation in **MLflow**, automatically registers champion models, and synthesizes reviews into **Amazon-style Customer Review Highlights** ("Customers Say", Pros/Cons with direct quotes, and aspect sentiment breakdown) powered by **Groq LLaMA 3.3**.

---

## 🚀 Key Features

- **Multi-Model MLflow Leaderboard**: Evaluates and compares **Logistic Regression**, **Calibrated Linear Support Vector Machines (LinearSVC)**, and **Multinomial Naive Bayes** across hyperparameter and TF-IDF feature grids.
- **Automated Model Registry Promotion**: Automatically transitions the champion model to the `Production` stage with aliases (`@champion`, `@production`) in MLflow.
- **Amazon-Style GenAI Review Insights**: Uses **Groq API** (`llama-3.3-70b-versatile` / `groq/compound-mini`) to synthesize customer sentiment into executive summaries, key liked/disliked attributes with direct customer quotes, and aspect breakdown cards.
- **High-Throughput Batch Scoring**: Memory-efficient chunked batch scoring pipeline capable of processing hundreds of thousands of reviews with product-level aggregation dashboards.
- **Zero-Latency Offline Serving**: Standalone production model artifacts saved in `models/` for offline inference with confidence scoring.
- **CI/CD Pipeline**: GitHub Actions workflow covering linting, test suites, model verification, batch pipeline validation, and Docker container build.

---

## 📁 Project Structure

```text
ReviewPulse/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # Automated CI/CD pipeline (Lint, Test, Docker)
├── data/                      # Dataset directory (.gitignore applied)
├── models/                    # Production model artifacts & metadata
│   ├── model.joblib
│   └── model_metadata.json
├── src/
│   └── reviewpulse/
│       ├── __init__.py        # Package initialization & exports
│       ├── config.py          # Environment configuration loader (.env)
│       ├── preprocessing.py   # Text cleaning & FastText/Star dataset parser
│       ├── train.py           # Multi-model training, MLflow tracking & registry
│       ├── predict.py         # Production inference (Local & Registry)
│       ├── batch_score.py     # Batch scoring & dashboard aggregation
│       └── insights.py        # Amazon GenAI review highlights engine (Groq)
├── tests/                     # Test suite (Preprocessing, Predict, GenAI)
│   ├── test_preprocessing.py
│   ├── test_predict.py
│   └── test_insights.py
├── .env.example               # Template environment variables
├── Dockerfile                 # Production container definition
├── pyproject.toml             # Package configuration
└── requirements.txt           # Project dependencies
```

---

## ⚡ Quickstart Guide

### 1. Installation

```bash
# Clone repository
git clone https://github.com/prachyasuman2396-arch/ReviewPulse.git
cd ReviewPulse

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and set your Groq API key (free at [console.groq.com/keys](https://console.groq.com/keys)):

```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=groq/compound-mini
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

---

## 🎯 Usage

### 1. Training & MLflow Leaderboard
Train candidate models, evaluate metrics (Accuracy, F1, Precision, Recall, ROC-AUC), rank them on the leaderboard, and promote the winning model to Production in MLflow:

```bash
python src/reviewpulse/train.py --sample-size 15000
```

### 2. Real-Time Production Sentiment Inference
Predict sentiment and confidence score using either the local model or the remote MLflow registry:

```bash
# Using local production artifact
python src/reviewpulse/predict.py "Best headphones I have ever bought, sound quality is superb!"

# Directly from MLflow Registry (@champion)
python src/reviewpulse/predict.py --from-registry "Battery stopped working after 3 days."
```

### 3. Amazon-Style GenAI Review Highlights
Generate an Amazon-style review synthesis card with customer consensus, pros/cons, and quotes:

```bash
python src/reviewpulse/insights.py
```

### 4. High-Throughput Batch Scoring & Dashboard
Score review datasets in memory-efficient chunks, aggregate product performance, and synthesize GenAI insights:

```bash
python src/reviewpulse/batch_score.py \
  --input data/raw/test.csv \
  --sample-rows 1000 \
  --output data/product_dashboard.csv \
  --ai-summary
```

### 5. Run Automated Tests
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 🐳 Docker Deployment

```bash
# Build production Docker image
docker build -t reviewpulse:latest .

# Run containerized inference
docker run --rm reviewpulse:latest
```

---

## 📜 License
MIT License.
