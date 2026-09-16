import argparse
import os
import sys
import numpy as np
import pandas as pd

# Ensure package root is in sys.path so TextCleaner unpickles cleanly
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from reviewpulse.predict import load_production_model
except ModuleNotFoundError:
    from predict import load_production_model


def resolve_input_file(provided_path: str | None = None) -> str:
    """Finds an existing input CSV or TXT file."""
    if provided_path and os.path.exists(provided_path):
        return provided_path

    candidates = [
        "data/Reviews.csv",
        "data/raw/test.csv",
        "data/raw/test.ft.txt",
        "data/processed/test.csv",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Input file not found. Tried '{provided_path}' and defaults: {candidates}"
    )


def detect_columns(df: pd.DataFrame, text_col: str | None = None, id_col: str | None = None):
    """
    Intelligently detects text and product identifier columns (case-insensitive).
    """
    col_map = {col.lower(): col for col in df.columns}

    # 1. Detect Text column
    resolved_text_col = None
    if text_col and text_col in df.columns:
        resolved_text_col = text_col
    else:
        text_candidates = ["text", "review", "review_text", "body", "summary", "content"]
        for cand in text_candidates:
            if cand in col_map:
                resolved_text_col = col_map[cand]
                break

    if not resolved_text_col:
        raise ValueError(
            f"Could not find a text column in CSV. Available columns: {list(df.columns)}. "
            f"Please specify --text-col."
        )

    # 2. Detect Product ID column (optional)
    resolved_id_col = None
    if id_col and id_col in df.columns:
        resolved_id_col = id_col
    else:
        id_candidates = ["productid", "product_id", "asin", "item_id", "item", "sku", "id"]
        for cand in id_candidates:
            if cand in col_map:
                resolved_id_col = col_map[cand]
                break

    return resolved_text_col, resolved_id_col


def run_batch_pipeline(
    input_csv: str | None = None,
    output_dashboard_csv: str = "data/product_dashboard.csv",
    output_scored_csv: str | None = None,
    chunk_size: int = 25000,
    from_registry: bool = False,
    models_dir: str = "models",
    text_col: str | None = None,
    id_col: str | None = None,
    sample_rows: int = 0,
    generate_ai_summary: bool = False,
    output_ai_json: str = "data/product_ai_insights.json",
):
    """
    Loads raw review data, executes high-throughput batch scoring,
    and aggregates insights for product dashboards.
    """
    resolved_input = resolve_input_file(input_csv)
    print(f"\n[Batch Pipeline] 📂 Loading input data from: {resolved_input}")

    nrows = sample_rows if sample_rows > 0 else None
    df = pd.read_csv(resolved_input, nrows=nrows)
    print(f"[Batch Pipeline] Loaded {len(df):,} total rows.")

    # Detect text and product columns
    resolved_text_col, resolved_id_col = detect_columns(df, text_col=text_col, id_col=id_col)
    print(f"[Batch Pipeline] Using review text column: '{resolved_text_col}'")
    if resolved_id_col:
        print(f"[Batch Pipeline] Using product ID column: '{resolved_id_col}'")
    else:
        print("[Batch Pipeline] No product ID column detected. Generating dataset-level sentiment analysis.")

    # Drop missing reviews
    initial_count = len(df)
    df = df.dropna(subset=[resolved_text_col]).copy()
    if len(df) < initial_count:
        print(f"[Batch Pipeline] Dropped {initial_count - len(df):,} rows with missing text.")

    # Load production model
    source_name = "MLflow Model Registry (@champion)" if from_registry else f"local '{models_dir}/model.joblib'"
    print(f"[Batch Pipeline] 🧠 Loading champion model from {source_name}...")
    pipeline = load_production_model(models_dir=models_dir, from_registry=from_registry)

    # Batch prediction in chunks to keep memory usage low and predictable
    total_reviews = len(df)
    print(f"[Batch Pipeline] 🚀 Scoring {total_reviews:,} reviews in chunks of {chunk_size:,}...")

    predictions = []
    probabilities = []
    has_proba = hasattr(pipeline, "predict_proba")

    num_chunks = int(np.ceil(total_reviews / chunk_size))
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_reviews)
        chunk_text = df[resolved_text_col].iloc[start_idx:end_idx].astype(str)

        chunk_preds = pipeline.predict(chunk_text)
        predictions.extend(chunk_preds)

        if has_proba:
            chunk_probs = pipeline.predict_proba(chunk_text)[:, 1]
            probabilities.extend(chunk_probs)

        print(f"   -> Processed [{end_idx:,}/{total_reviews:,}] reviews ({(end_idx/total_reviews)*100:.1f}%)")

    # Attach predictions
    df["predicted_label"] = predictions
    df["predicted_sentiment"] = np.where(df["predicted_label"] == 1, "Positive", "Negative")

    if has_proba:
        df["confidence"] = np.where(
            df["predicted_label"] == 1,
            np.array(probabilities),
            1.0 - np.array(probabilities),
        ).round(4)
        df["positive_probability"] = np.array(probabilities).round(4)

    # Overall dataset statistics
    pos_count = int((df["predicted_label"] == 1).sum())
    neg_count = total_reviews - pos_count
    pos_pct = round((pos_count / total_reviews) * 100, 2) if total_reviews > 0 else 0.0
    neg_pct = round(100.0 - pos_pct, 2)

    print("\n" + "=" * 65)
    print("                BATCH INFERENCE SUMMARY")
    print("=" * 65)
    print(f" Total Reviews Scored : {total_reviews:,}")
    print(f" 🟢 Positive Reviews  : {pos_count:,} ({pos_pct}%)")
    print(f" 🔴 Negative Reviews  : {neg_count:,} ({neg_pct}%)")
    if has_proba and "confidence" in df:
        print(f" Average Confidence   : {df['confidence'].mean()*100:.2f}%")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # Optional: Save detailed scored reviews
    # -------------------------------------------------------------------------
    if output_scored_csv:
        os.makedirs(os.path.dirname(output_scored_csv) or ".", exist_ok=True)
        df.to_csv(output_scored_csv, index=False)
        print(f"\n[Batch Pipeline] 💾 Saved scored reviews to: {output_scored_csv}")

    # -------------------------------------------------------------------------
    # Product aggregation dashboard
    # -------------------------------------------------------------------------
    if resolved_id_col:
        print(f"\n[Batch Pipeline] 📊 Aggregating sentiment metrics by '{resolved_id_col}'...")
        agg_kwargs = {
            "total_reviews": ("predicted_label", "count"),
            "positive_reviews": ("predicted_label", lambda s: (s == 1).sum()),
            "negative_reviews": ("predicted_label", lambda s: (s == 0).sum()),
        }
        if has_proba and "confidence" in df:
            agg_kwargs["avg_confidence"] = ("confidence", "mean")

        product_stats = df.groupby(resolved_id_col).agg(**agg_kwargs).reset_index()

        product_stats["positive_percentage"] = (
            (product_stats["positive_reviews"] / product_stats["total_reviews"]) * 100
        ).round(2)
        product_stats["negative_percentage"] = (
            (product_stats["negative_reviews"] / product_stats["total_reviews"]) * 100
        ).round(2)
        if "avg_confidence" in product_stats:
            product_stats["avg_confidence"] = (product_stats["avg_confidence"] * 100).round(2)

        # Sort by most reviewed products
        product_stats = product_stats.sort_values(by="total_reviews", ascending=False).reset_index(drop=True)

        os.makedirs(os.path.dirname(output_dashboard_csv) or ".", exist_ok=True)
        product_stats.to_csv(output_dashboard_csv, index=False)
        print(f"[Batch Pipeline] 💾 Saved product dashboard to: {output_dashboard_csv}")

        # Display Top Products Leaderboard
        print("\n=================================================================")
        print("         REVIEWPULSE PRODUCT DASHBOARD (Top 5 Products)          ")
        print("=================================================================")
        for _, row in product_stats.head(5).iterrows():
            print(f"Product: {row[resolved_id_col]} | Reviews: {row['total_reviews']:,}")
            print(f"   🟢 Positive: {row['positive_percentage']}% | 🔴 Negative: {row['negative_percentage']}%")
            if "avg_confidence" in row:
                print(f"   🎯 Avg Confidence: {row['avg_confidence']}%")
            print("-" * 65)
    else:
        # If no Product ID column exists, write the dataset-level stats to the output CSV
        summary_df = pd.DataFrame([
            {
                "total_reviews": total_reviews,
                "positive_reviews": pos_count,
                "negative_reviews": neg_count,
                "positive_percentage": pos_pct,
                "negative_percentage": neg_pct,
            }
        ])
        os.makedirs(os.path.dirname(output_dashboard_csv) or ".", exist_ok=True)
        summary_df.to_csv(output_dashboard_csv, index=False)
        print(f"[Batch Pipeline] 💾 Saved sentiment summary to: {output_dashboard_csv}")

    # -------------------------------------------------------------------------
    # Optional: GenAI Amazon-Style Review Insights Synthesis
    # -------------------------------------------------------------------------
    if generate_ai_summary:
        try:
            from reviewpulse.insights import generate_amazon_review_insights, print_amazon_insights_card
        except ModuleNotFoundError:
            from insights import generate_amazon_review_insights, print_amazon_insights_card

        import json

        print("\n[Batch Pipeline] ✨ Generating Amazon-Style GenAI Review Insights...")
        ai_results = {}

        if resolved_id_col:
            top_products = product_stats.head(3)[resolved_id_col].tolist()
            for pid in top_products:
                prod_reviews = df[df[resolved_id_col] == pid][resolved_text_col].head(30).tolist()
                if prod_reviews:
                    insights = generate_amazon_review_insights(prod_reviews, product_id=str(pid))
                    print_amazon_insights_card(insights)
                    ai_results[str(pid)] = insights
        else:
            all_reviews = df[resolved_text_col].head(40).tolist()
            insights = generate_amazon_review_insights(all_reviews, product_id="Dataset_Summary")
            print_amazon_insights_card(insights)
            ai_results["Dataset_Summary"] = insights

        if output_ai_json:
            os.makedirs(os.path.dirname(output_ai_json) or ".", exist_ok=True)
            with open(output_ai_json, "w", encoding="utf-8") as f:
                json.dump(ai_results, f, indent=2)
            print(f"[Batch Pipeline] 💾 Saved Amazon GenAI insights to: {output_ai_json}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReviewPulse High-Throughput Batch Scoring Pipeline")
    parser.add_argument("--input", type=str, default=None, help="Path to input reviews CSV or TXT file")
    parser.add_argument("--output", type=str, default="data/product_dashboard.csv", help="Path to output dashboard CSV")
    parser.add_argument("--output-scored", type=str, default=None, help="Optional path to save all individual scored reviews")
    parser.add_argument("--chunk-size", type=int, default=25000, help="Batch size for processing reviews")
    parser.add_argument("--sample-rows", type=int, default=0, help="Optional number of rows to sample (0 = all)")
    parser.add_argument("--from-registry", action="store_true", help="Load champion model from MLflow registry")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory containing local model.joblib")
    parser.add_argument("--text-col", type=str, default=None, help="Name of the review text column")
    parser.add_argument("--id-col", type=str, default=None, help="Name of the product ID column")
    parser.add_argument("--ai-summary", action="store_true", help="Generate Amazon-style GenAI review summary")
    parser.add_argument("--output-ai-json", type=str, default="data/product_ai_insights.json", help="Path to save GenAI insights JSON")

    args = parser.parse_args()
    run_batch_pipeline(
        input_csv=args.input,
        output_dashboard_csv=args.output,
        output_scored_csv=args.output_scored,
        chunk_size=args.chunk_size,
        from_registry=args.from_registry,
        models_dir=args.models_dir,
        text_col=args.text_col,
        id_col=args.id_col,
        sample_rows=args.sample_rows,
        generate_ai_summary=args.ai_summary,
        output_ai_json=args.output_ai_json,
    )
