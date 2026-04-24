"""
Food-Based Restaurant Recommender — Majitar, Sikkim
====================================================
Uses sentiment analysis (VADER, TextBlob, Logistic Regression) on Google
reviews to recommend the best restaurants for a specific food/dish.

Usage:  python food_recommender.py
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdin  = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# NLP Libraries
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Download required NLTK data
for pkg in ["vader_lexicon", "stopwords", "punkt", "punkt_tab",
            "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

# ─── Import dataset ───────────────────────────────────────────────────────────
from reviews_data import REVIEWS, FOOD_KEYWORDS

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Build sentiment-annotated DataFrame
# ══════════════════════════════════════════════════════════════════════════════
df = pd.DataFrame(REVIEWS)

def rating_to_sentiment(r):
    if r >= 4: return "Positive"
    elif r == 3: return "Neutral"
    else:       return "Negative"

df["true_label"] = df["rating"].apply(rating_to_sentiment)

# Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

df["clean_review"] = df["review"].apply(preprocess)

# ── VADER ──────────────────────────────────────────────────────────────────────
sia = SentimentIntensityAnalyzer()

def vader_label(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:  return "Positive"
    elif score <= -0.05: return "Negative"
    else:               return "Neutral"

df["vader_label"] = df["review"].apply(vader_label)
df["vader_score"] = df["review"].apply(lambda t: sia.polarity_scores(t)["compound"])

# ── TextBlob ───────────────────────────────────────────────────────────────────
def textblob_label(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.05:   return "Positive"
    elif score < -0.05: return "Negative"
    else:               return "Neutral"

df["tb_label"] = df["review"].apply(textblob_label)
df["tb_score"] = df["review"].apply(lambda t: TextBlob(t).sentiment.polarity)

# ── Logistic Regression ───────────────────────────────────────────────────────
label_map = {"Positive": 2, "Neutral": 1, "Negative": 0}
inv_map   = {v: k for k, v in label_map.items()}

X = df["clean_review"]
y = df["true_label"].map(label_map)

tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.25, random_state=42, stratify=y)

lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr.fit(X_train, y_train)
df["lr_label"] = [inv_map[p] for p in lr.predict(X_vec)]

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Food-matching utilities
# ══════════════════════════════════════════════════════════════════════════════

def find_matching_food(query):
    """Return the canonical food name that best matches the user query."""
    query = query.strip().lower()

    # Exact match on canonical name
    if query in FOOD_KEYWORDS:
        return query

    # Check if query matches any search term
    for food, terms in FOOD_KEYWORDS.items():
        for term in terms:
            if query in term or term.strip() in query:
                return food

    # Partial / substring match on canonical names
    for food in FOOD_KEYWORDS:
        if query in food or food in query:
            return food

    return None


def get_reviews_for_food(food_name):
    """Return a subset of df where the review mentions the food."""
    terms = FOOD_KEYWORDS[food_name]
    mask = df["review"].str.lower().apply(
        lambda text: any(t.strip() in text for t in terms)
    )
    return df[mask].copy()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Scoring & ranking
# ══════════════════════════════════════════════════════════════════════════════

def score_restaurants(food_df):
    """Compute per-restaurant sentiment scores and combined ranking."""
    results = []
    for rest, grp in food_df.groupby("restaurant"):
        n = len(grp)

        # VADER
        vader_avg = grp["vader_score"].mean()
        vader_pos = (grp["vader_label"] == "Positive").sum()
        vader_neg = (grp["vader_label"] == "Negative").sum()

        # TextBlob
        tb_avg = grp["tb_score"].mean()
        tb_pos = (grp["tb_label"] == "Positive").sum()
        tb_neg = (grp["tb_label"] == "Negative").sum()

        # Logistic Regression
        lr_pos = (grp["lr_label"] == "Positive").sum()
        lr_neg = (grp["lr_label"] == "Negative").sum()
        lr_ratio = lr_pos / n  # fraction predicted positive

        # Normalize VADER (−1…+1) → 0…1
        vader_norm = (vader_avg + 1) / 2
        # Normalize TextBlob (−1…+1) → 0…1
        tb_norm = (tb_avg + 1) / 2

        # Combined score (0–10 scale)
        combined = (0.4 * vader_norm + 0.3 * tb_norm + 0.3 * lr_ratio) * 10

        # Pick the best positive review as a snippet
        pos_reviews = grp[grp["vader_label"] == "Positive"]
        if len(pos_reviews) > 0:
            snippet = pos_reviews.iloc[0]["review"]
        else:
            snippet = grp.iloc[0]["review"]
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."

        results.append({
            "restaurant": rest,
            "reviews_found": n,
            "avg_rating": grp["rating"].mean(),
            "vader_avg": vader_avg,
            "vader_pos": vader_pos,
            "vader_neg": vader_neg,
            "tb_avg": tb_avg,
            "tb_pos": tb_pos,
            "tb_neg": tb_neg,
            "lr_pos": lr_pos,
            "lr_neg": lr_neg,
            "lr_ratio": lr_ratio,
            "combined_score": combined,
            "snippet": snippet,
        })

    results.sort(key=lambda r: r["combined_score"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Display helpers
# ══════════════════════════════════════════════════════════════════════════════

BOLD  = "\033[1m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
YELLOW = "\033[93m"
RED   = "\033[91m"
MAG   = "\033[95m"
DIM   = "\033[2m"
RESET = "\033[0m"

BANNER = f"""
{CYAN}{'='*62}
   MAJITAR FOOD RECOMMENDER  --  Sentiment-Powered
{'='*62}{RESET}
{DIM}  Analyses Google reviews using VADER, TextBlob & Logistic
  Regression to recommend the best restaurants for any dish.{RESET}
"""

def print_banner():
    print(BANNER)


def print_available_foods():
    foods = sorted(FOOD_KEYWORDS.keys())
    print(f"\n{YELLOW}[Searchable foods]{RESET}")
    line = "   "
    for i, f in enumerate(foods):
        line += f"{f}, "
        if (i + 1) % 6 == 0:
            print(line.rstrip(", "))
            line = "   "
    if line.strip():
        print(line.rstrip(", "))
    print()


def display_results(food_name, ranked):
    """Pretty-print the ranked restaurant results."""
    print(f"\n{CYAN}{'-'*62}{RESET}")
    print(f"{BOLD}{GREEN}  >> Results for: \"{food_name.upper()}\"{RESET}")
    print(f"{CYAN}{'-'*62}{RESET}")

    for rank, r in enumerate(ranked, 1):
        # Medal emoji for top 3
        medal = {1: "[1st]", 2: "[2nd]", 3: "[3rd]"}.get(rank, f"[#{rank}]")

        print(f"\n  {BOLD}{medal}  {r['restaurant']}{RESET}")
        print(f"  {DIM}{'-'*50}{RESET}")

        # Combined score (large)
        score = r["combined_score"]
        if score >= 7:
            sc_color = GREEN
        elif score >= 4:
            sc_color = YELLOW
        else:
            sc_color = RED
        print(f"  {BOLD}Combined Score: {sc_color}{score:.1f} / 10{RESET}"
              f"   ({r['reviews_found']} review(s) matched)")

        # Average star rating
        stars = "*" * round(r["avg_rating"]) + "." * (5 - round(r["avg_rating"]))
        print(f"  Avg Rating:    {YELLOW}[{stars}] ({r['avg_rating']:.1f}){RESET}")

        # Per-model breakdown
        print(f"\n  {MAG}+-- VADER{RESET}")
        print(f"  {MAG}|{RESET}  Avg Compound: {r['vader_avg']:+.3f}"
              f"   [+{r['vader_pos']} positive, -{r['vader_neg']} negative]")

        print(f"  {MAG}+-- TextBlob{RESET}")
        print(f"  {MAG}|{RESET}  Avg Polarity:  {r['tb_avg']:+.3f}"
              f"   [+{r['tb_pos']} positive, -{r['tb_neg']} negative]")

        print(f"  {MAG}+-- Logistic Regression{RESET}")
        print(f"  {MAG}|{RESET}  Positive Ratio: {r['lr_ratio']:.0%}"
              f"   [+{r['lr_pos']} positive, -{r['lr_neg']} negative]")

        print(f"  {MAG}+-- Sample Review{RESET}")
        print(f"     {DIM}\"{r['snippet']}\"{RESET}")

    print(f"\n{CYAN}{'='*62}{RESET}\n")


def display_no_results(query):
    """Handle the case where no reviews match the food query."""
    print(f"\n  {RED}[X] No reviews found mentioning \"{query}\".{RESET}")
    print(f"  {YELLOW}Try one of these instead:{RESET}\n")
    print_available_foods()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Main interactive loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print_banner()
    print(f"  {DIM}Dataset: {len(df)} reviews across"
          f" {df['restaurant'].nunique()} restaurants in Majitar, Sikkim{RESET}")
    print_available_foods()

    while True:
        try:
            query = input(f"{BOLD}>> What do you want to eat? "
                          f"(or 'quit' to exit): {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if query.lower() in ("quit", "exit", "q"):
            print(f"\n{GREEN}Thanks for using Majitar Food Recommender! Enjoy your meal!{RESET}")
            break

        if not query:
            continue

        food = find_matching_food(query)

        if food is None:
            display_no_results(query)
            continue

        food_df = get_reviews_for_food(food)

        if food_df.empty:
            display_no_results(query)
            continue

        ranked = score_restaurants(food_df)
        display_results(food, ranked)


if __name__ == "__main__":
    main()
