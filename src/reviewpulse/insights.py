import json
import os
import re
import sys
import urllib.request
import urllib.error
from typing import Any

# Ensure package path is configured
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from reviewpulse.config import get_groq_api_key, get_groq_model, is_groq_configured
except ModuleNotFoundError:
    from config import get_groq_api_key, get_groq_model, is_groq_configured


AMAZON_SYNTHESIS_SYSTEM_PROMPT = """You are an expert Amazon Customer Review Insights AI.
Analyze the provided batch of customer reviews for a product and synthesize them into an Amazon-style customer review summary.

You must respond with ONLY a valid JSON object adhering precisely to this structure:
{
  "summary": "A 2-3 sentence executive synthesis starting with 'Customers say...' summarizing the collective customer consensus.",
  "overall_sentiment": "Positive" | "Mixed" | "Negative",
  "average_rating_estimate": float between 1.0 and 5.0,
  "what_customers_liked": [
    {
      "aspect": "Short title (e.g. Sound Quality, Ease of Setup, Battery Life)",
      "description": "1 sentence explanation of why customers appreciated this feature.",
      "customer_quote": "Direct brief quote from one of the positive reviews illustrating this."
    }
  ],
  "what_customers_disliked": [
    {
      "aspect": "Short title (e.g. Durability, Customer Support, Price)",
      "description": "1 sentence explanation of complaints or shortcomings mentioned.",
      "customer_quote": "Direct brief quote from one of the negative reviews illustrating this."
    }
  ],
  "aspect_breakdown": {
    "Quality": "Positive" | "Mixed" | "Negative",
    "Value": "Positive" | "Mixed" | "Negative",
    "Ease of Use": "Positive" | "Mixed" | "Negative",
    "Durability": "Positive" | "Mixed" | "Negative"
  }
}

Guidelines:
- Keep the tone objective, balanced, and customer-centric, exactly like Amazon's AI review highlights.
- Highlight at least 2-3 items in 'what_customers_liked' and 1-3 items in 'what_customers_disliked'.
- If the reviews are predominantly positive or negative, reflect that proportionally.
- Respond with pure JSON only, no surrounding markdown formatting or text.
"""


def _extract_json(raw_text: str) -> dict[str, Any]:
    """Extracts and parses JSON object from model response, stripping think tags and code fences."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def _call_groq_api_http(api_key: str, model: str, prompt: str) -> dict[str, Any]:
    """
    Calls Groq Chat Completions API via standard library HTTPS request.
    Works with zero external dependencies.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": AMAZON_SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        data = json.loads(body)
        raw_content = data["choices"][0]["message"]["content"]
        return _extract_json(raw_content)


def _call_groq_sdk(api_key: str, model: str, prompt: str) -> dict[str, Any]:
    """Calls Groq using the official python SDK if installed."""
    from groq import Groq
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": AMAZON_SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    content = chat_completion.choices[0].message.content
    return json.loads(content)


def _generate_fallback_summary(reviews: list[str], product_id: str | None = None) -> dict[str, Any]:
    """
    Heuristic rule-based fallback when Groq API key is not configured.
    Ensures tests and offline deployments run seamlessly.
    """
    positive_words = {"great", "good", "love", "loved", "best", "excellent", "amazing", "quality", "works", "fine", "perfect", "easy"}
    negative_words = {"bad", "poor", "terrible", "worst", "died", "broke", "stopped", "waste", "complaint", "unhappy", "problem"}

    pos_quotes = []
    neg_quotes = []

    for r in reviews:
        lower = r.lower()
        has_pos = any(w in lower for w in positive_words)
        has_neg = any(w in lower for w in negative_words)
        if has_pos and not has_neg:
            pos_quotes.append(r[:120])
        elif has_neg:
            neg_quotes.append(r[:120])

    total = len(reviews)
    pos_count = len(pos_quotes)
    neg_count = len(neg_quotes)

    if pos_count >= neg_count:
        overall = "Positive"
        rating = round(min(5.0, 3.5 + (pos_count / (total or 1)) * 1.5), 1)
        summary = (
            "Customers say this product performs reliably with great overall satisfaction. "
            "Many reviewers appreciate the build quality and functionality, though occasional complaints exist."
        )
    else:
        overall = "Negative"
        rating = round(max(1.0, 3.0 - (neg_count / (total or 1)) * 1.5), 1)
        summary = (
            "Customers express dissatisfaction with durability and longevity. "
            "A notable portion of reviews highlight failures after initial use."
        )

    liked = [
        {
            "aspect": "Core Performance",
            "description": "Customers appreciate when the product works as expected out of the box.",
            "customer_quote": pos_quotes[0] if pos_quotes else "Great quality and works fine.",
        }
    ]
    if len(pos_quotes) > 1:
        liked.append({
            "aspect": "Value & Usability",
            "description": "Users found the product convenient and easy to adopt.",
            "customer_quote": pos_quotes[1],
        })

    disliked = [
        {
            "aspect": "Reliability & Longevity",
            "description": "Some users reported premature failures or declining battery/hardware performance.",
            "customer_quote": neg_quotes[0] if neg_quotes else "Not as durable as expected.",
        }
    ]

    return {
        "summary": summary,
        "overall_sentiment": overall,
        "average_rating_estimate": rating,
        "total_reviews_analyzed": total,
        "what_customers_liked": liked,
        "what_customers_disliked": disliked,
        "aspect_breakdown": {
            "Quality": "Positive" if overall == "Positive" else "Mixed",
            "Value": "Positive" if pos_count > neg_count else "Mixed",
            "Ease of Use": "Positive",
            "Durability": "Mixed" if neg_count > 0 else "Positive",
        },
        "source": "heuristic_fallback (Configure GROQ_API_KEY in .env for live LLaMA 3.3)",
    }


def generate_amazon_review_insights(
    reviews: list[str],
    product_id: str | None = None,
    max_reviews_to_sample: int = 40,
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Synthesizes a list of reviews into an Amazon-style GenAI summary using Groq LLaMA 3.3.
    Falls back gracefully if Groq API key is not configured.
    """
    if not reviews:
        raise ValueError("Reviews list cannot be empty.")

    # Clean and filter reviews
    cleaned_reviews = [str(r).strip() for r in reviews if str(r).strip()]
    if not cleaned_reviews:
        raise ValueError("No valid review text found.")

    # Sample reviews if large to fit context window
    if len(cleaned_reviews) > max_reviews_to_sample:
        sampled_reviews = cleaned_reviews[:max_reviews_to_sample]
    else:
        sampled_reviews = cleaned_reviews

    groq_key = api_key or get_groq_api_key()
    groq_model = model or get_groq_model()

    if not groq_key or groq_key == "your_groq_api_key_here":
        print("[ReviewPulse GenAI] ℹ️ GROQ_API_KEY not set in .env. Using intelligent local synthesis.")
        insights = _generate_fallback_summary(sampled_reviews, product_id=product_id)
        if product_id:
            insights["product_id"] = product_id
        return insights

    # Prepare formatted prompt
    product_header = f"Product ID: {product_id}\n" if product_id else "Product Reviews:\n"
    formatted_reviews = "\n---\n".join([f"Review #{idx+1}: {r}" for idx, r in enumerate(sampled_reviews)])
    user_prompt = f"{product_header}Total Reviews: {len(sampled_reviews)}\n\nReviews:\n{formatted_reviews}"

    print(f"[ReviewPulse GenAI] 🤖 Querying Groq ({groq_model}) for Amazon-style review synthesis...")
    try:
        # Try SDK first if available, else standard library HTTP
        try:
            insights = _call_groq_sdk(groq_key, groq_model, user_prompt)
        except ImportError:
            insights = _call_groq_api_http(groq_key, groq_model, user_prompt)

        insights["total_reviews_analyzed"] = len(sampled_reviews)
        if product_id:
            insights["product_id"] = product_id
        insights["source"] = f"groq:{groq_model}"
        return insights

    except Exception as e:
        print(f"[ReviewPulse GenAI] ⚠️ Groq API request failed ({e}). Falling back to local synthesis.")
        fallback = _generate_fallback_summary(sampled_reviews, product_id=product_id)
        fallback["error_notice"] = str(e)
        return fallback


def print_amazon_insights_card(insights: dict[str, Any]):
    """Prints a styled Amazon-like Review Highlights card in terminal."""
    prod = insights.get("product_id", "")
    prod_label = f" for {prod}" if prod else ""
    rating = insights.get("average_rating_estimate", 4.0)
    sentiment = insights.get("overall_sentiment", "Positive")
    total = insights.get("total_reviews_analyzed", 0)

    # Convert rating to stars (e.g. 4.2 -> ★★★★☆)
    full_stars = int(rating)
    stars = "★" * full_stars + "☆" * (5 - full_stars)

    badge = "🟢 POSITIVE" if sentiment == "Positive" else ("🔴 NEGATIVE" if sentiment == "Negative" else "🟡 MIXED")

    print("\n" + "=" * 75)
    print(f"       🛒 AMAZON AI REVIEW HIGHLIGHTS{prod_label}")
    print("=" * 75)
    print(f" Rating : {stars} {rating:.1f} out of 5  |  Sentiment: {badge}  |  {total:,} reviews")
    print("-" * 75)
    print(f" 💬 Customers Say:")
    print(f"    \"{insights.get('summary', 'No summary available.')}\"")
    print("-" * 75)

    print(" 👍 What Customers Liked:")
    for item in insights.get("what_customers_liked", []):
        print(f"   • {item.get('aspect', 'Feature')}: {item.get('description', '')}")
        if item.get("customer_quote"):
            print(f"     Quote: \"{item.get('customer_quote')}\"")

    print("\n 👎 What Customers Disliked:")
    for item in insights.get("what_customers_disliked", []):
        print(f"   • {item.get('aspect', 'Issue')}: {item.get('description', '')}")
        if item.get("customer_quote"):
            print(f"     Quote: \"{item.get('customer_quote')}\"")

    aspects = insights.get("aspect_breakdown", {})
    if aspects:
        print("\n 🔍 Aspect Breakdown:")
        aspect_badges = []
        for asp, val in aspects.items():
            sym = "🟢" if val == "Positive" else ("🔴" if val == "Negative" else "🟡")
            aspect_badges.append(f"{sym} {asp}: {val}")
        print("   " + "  |  ".join(aspect_badges))

    print("=" * 75 + "\n")


if __name__ == "__main__":
    sample_reviews = [
        "Great CD: My lovely Pat has one of the GREAT voices of her generation. I have listened to this CD for YEARS and I still LOVE IT. Vocals are stunning.",
        "One of the best game music soundtracks. There is an incredible mix of fun, epic, and emotional songs. Led me to purchase the album immediately.",
        "Batteries died within a year ... I bought this charger in Jul 2003 and it worked OK for a while, but after a year the batteries would not hold a charge.",
        "Works fine, but Maha Energy is better. Powerex charger works faster with option for slower charge.",
        "Great for the non-audiophile: Well built, easy to setup, resolution and effects are great. Cons: No universal remote.",
    ]
    print("Testing Amazon AI Review Insights:")
    result = generate_amazon_review_insights(sample_reviews, product_id="PROD-DEMO-101")
    print_amazon_insights_card(result)
