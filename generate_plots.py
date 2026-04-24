"""
Generate individual visualization images with white backgrounds.
Each plot is saved as a separate PNG file in a 'plots/' subfolder.
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# Download NLTK data
for pkg in ["vader_lexicon", "stopwords", "punkt", "punkt_tab",
            "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

from reviews_data import REVIEWS

# ─── DATA SETUP (same as sentiment_analyzer.py) ──────────────────────────────
df = pd.DataFrame(REVIEWS)

def rating_to_sentiment(r):
    if r >= 4: return "Positive"
    elif r == 3: return "Neutral"
    else:       return "Negative"

df["true_label"] = df["rating"].apply(rating_to_sentiment)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

df["clean_review"] = df["review"].apply(preprocess)

# VADER
sia = SentimentIntensityAnalyzer()
def vader_label(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:  return "Positive"
    elif score <= -0.05: return "Negative"
    else:               return "Neutral"

df["vader_label"] = df["review"].apply(vader_label)
df["vader_score"] = df["review"].apply(lambda t: sia.polarity_scores(t)["compound"])

# TextBlob
def textblob_label(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.05:   return "Positive"
    elif score < -0.05: return "Negative"
    else:               return "Neutral"

df["tb_label"] = df["review"].apply(textblob_label)
df["tb_score"] = df["review"].apply(lambda t: TextBlob(t).sentiment.polarity)
df["tb_subjectivity"] = df["review"].apply(lambda t: TextBlob(t).sentiment.subjectivity)

# Logistic Regression
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
y_pred_lr = lr.predict(X_test)
df["lr_label"] = [inv_map[p] for p in lr.predict(X_vec)]

CLASSES = ["Negative", "Neutral", "Positive"]
y_true = df["true_label"]
y_true_lr_str = y_test.map(inv_map)
y_pred_lr_str = pd.Series(y_pred_lr).map(inv_map)

# Metrics
def calc_metrics(y_true, y_pred, name):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

results = []
results.append(calc_metrics(y_true, df["vader_label"], "VADER"))
results.append(calc_metrics(y_true, df["tb_label"], "TextBlob"))
results.append(calc_metrics(y_true_lr_str, y_pred_lr_str, "Logistic Regression"))
results_df = pd.DataFrame(results)

cv_scores = cross_val_score(lr, X_vec, y, cv=5, scoring="accuracy")

# ─── OUTPUT FOLDER ────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT, exist_ok=True)

# ─── STYLE: White background, clean look ─────────────────────────────────────
PALETTE = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}
MODEL_COLORS = ["#5C6BC0", "#26A69A", "#EF5350"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.labelcolor": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#222222",
    "axes.titlecolor": "#111111",
    "grid.color": "#DDDDDD",
    "axes.edgecolor": "#CCCCCC",
})

def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: plots/{name}")


# ══════════════════════════════════════════════════════════════════════════════
#  1. True Sentiment Distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
counts = y_true.value_counts()
bars = ax.bar(counts.index, counts.values,
              color=[PALETTE[c] for c in counts.index],
              edgecolor="#AAAAAA", linewidth=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(val), ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.set_title("True Sentiment Distribution", fontsize=15, fontweight="bold", pad=12)
ax.set_ylabel("Number of Reviews")
ax.set_ylim(0, counts.max() + 8)
ax.grid(axis="y", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
savefig(fig, "01_sentiment_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
#  2. Sentiment by Restaurant
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
rest_sent = df.groupby(["restaurant", "true_label"]).size().unstack(fill_value=0)
rest_sent = rest_sent.reindex(columns=["Negative", "Neutral", "Positive"], fill_value=0)
rest_sent.index = [r.replace("(Tourist Cafeteria)", "").replace("Restro & Lounge", "R&L")
                   .replace("Restro & Bar", "R&B").strip() for r in rest_sent.index]
rest_sent.plot(kind="bar", ax=ax, color=[PALETTE[c] for c in rest_sent.columns],
               edgecolor="#AAAAAA", linewidth=0.5)
ax.set_title("Sentiment by Restaurant", fontsize=15, fontweight="bold", pad=12)
ax.set_ylabel("Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.grid(axis="y", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
savefig(fig, "02_sentiment_by_restaurant.png")


# ══════════════════════════════════════════════════════════════════════════════
#  3. Average Rating by Restaurant
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
avg_rat = df.groupby("restaurant")["rating"].mean().sort_values(ascending=True)
avg_rat.index = [r.replace("(Tourist Cafeteria)", "").replace("Restro & Lounge", "R&L")
                 .replace("Restro & Bar", "R&B").strip() for r in avg_rat.index]
bars = ax.barh(avg_rat.index, avg_rat.values,
               color=MODEL_COLORS[0], edgecolor="#AAAAAA", linewidth=0.5, height=0.6)
for bar, val in zip(bars, avg_rat.values):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}", va="center", fontsize=10, fontweight="bold", color="#333")
ax.set_title("Average Rating by Restaurant", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Average Star Rating")
ax.set_xlim(0, 5.5)
ax.grid(axis="x", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
savefig(fig, "03_avg_rating_by_restaurant.png")


# ══════════════════════════════════════════════════════════════════════════════
#  4. VADER Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 5))
cm_v = confusion_matrix(y_true, df["vader_label"], labels=CLASSES)
sns.heatmap(cm_v, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=1, linecolor="white",
            annot_kws={"size": 14, "weight": "bold"},
            cbar_kws={"shrink": 0.8})
ax.set_title("VADER - Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
savefig(fig, "04_vader_confusion_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
#  5. TextBlob Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 5))
cm_t = confusion_matrix(y_true, df["tb_label"], labels=CLASSES)
sns.heatmap(cm_t, annot=True, fmt="d", cmap="Greens", ax=ax,
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=1, linecolor="white",
            annot_kws={"size": 14, "weight": "bold"},
            cbar_kws={"shrink": 0.8})
ax.set_title("TextBlob - Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
savefig(fig, "05_textblob_confusion_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
#  6. Logistic Regression Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 5))
cm_lr = confusion_matrix(y_true_lr_str, y_pred_lr_str, labels=CLASSES)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Oranges", ax=ax,
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=1, linecolor="white",
            annot_kws={"size": 14, "weight": "bold"},
            cbar_kws={"shrink": 0.8})
ax.set_title("Logistic Regression - Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
savefig(fig, "06_lr_confusion_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
#  7. Model Performance Comparison
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
metrics_cols = ["Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(metrics_cols))
width = 0.22
for i, (_, row) in enumerate(results_df.iterrows()):
    vals = [row[c] for c in metrics_cols]
    b = ax.bar(x + i*width, vals, width, label=row["Model"],
               color=MODEL_COLORS[i], edgecolor="#AAAAAA", linewidth=0.5)
    for bar, val in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_title("Model Performance Comparison", fontsize=15, fontweight="bold", pad=12)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_cols, fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
ax.grid(axis="y", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
savefig(fig, "07_model_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
#  8. VADER Compound Score Distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
for sentiment in ["Positive", "Neutral", "Negative"]:
    grp = df[df["true_label"] == sentiment]
    ax.hist(grp["vader_score"], bins=12, alpha=0.65,
            label=sentiment, color=PALETTE[sentiment], edgecolor="#AAAAAA")
ax.axvline(0.05, color="#E65100", linestyle="--", linewidth=1.5, label="Threshold +0.05")
ax.axvline(-0.05, color="#E65100", linestyle="--", linewidth=1.5, alpha=0.6, label="Threshold -0.05")
ax.set_title("VADER Compound Score Distribution", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Compound Score")
ax.set_ylabel("Frequency")
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
savefig(fig, "08_vader_score_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
#  9. TextBlob Polarity vs Subjectivity
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
for sentiment in ["Positive", "Neutral", "Negative"]:
    grp = df[df["true_label"] == sentiment]
    ax.scatter(grp["tb_score"], grp["tb_subjectivity"],
               label=sentiment, color=PALETTE[sentiment],
               alpha=0.8, s=70, edgecolors="#555555", linewidths=0.5)
ax.axvline(0, color="#999999", linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_title("TextBlob: Polarity vs Subjectivity", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Polarity")
ax.set_ylabel("Subjectivity")
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
savefig(fig, "09_textblob_polarity_vs_subjectivity.png")


# ══════════════════════════════════════════════════════════════════════════════
#  10. Cross-Validation Scores
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 6), cv_scores, marker="o", color="#5C6BC0",
        linewidth=2.5, markersize=10, markerfacecolor="white",
        markeredgecolor="#5C6BC0", markeredgewidth=2.5)
ax.axhline(cv_scores.mean(), color="#EF5350", linestyle="--",
           linewidth=1.8, label=f"Mean = {cv_scores.mean():.4f}")
ax.fill_between(range(1, 6),
                cv_scores.mean() - cv_scores.std(),
                cv_scores.mean() + cv_scores.std(),
                alpha=0.15, color="#5C6BC0", label=f"$\\pm$1 Std = {cv_scores.std():.4f}")
for i, sc in enumerate(cv_scores):
    ax.text(i+1, sc + 0.015, f"{sc:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("LR 5-Fold Cross-Validation Accuracy", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Fold")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.15)
ax.set_xticks(range(1, 6))
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
savefig(fig, "10_cross_validation.png")


# ══════════════════════════════════════════════════════════════════════════════
#  11. Overall Sentiment Share (Pie)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 7))
overall = y_true.value_counts()
wedges, texts, autotexts = ax.pie(
    overall.values,
    labels=overall.index,
    colors=[PALETTE[c] for c in overall.index],
    autopct="%1.1f%%", startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 2.5},
    textprops={"fontsize": 12})
for at in autotexts:
    at.set_fontweight("bold")
    at.set_fontsize(13)
ax.set_title("Overall Sentiment Share", fontsize=15, fontweight="bold", pad=15)
savefig(fig, "11_sentiment_pie.png")


print(f"\nDone! All 11 plots saved to: {OUT}")
