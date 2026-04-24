"""
Sentiment Analysis of Google Reviews - Majitar, Sikkim Restaurants
Uses: VADER, TextBlob, and Logistic Regression (Machine Learning)
Evaluates on: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter

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
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

# Download required NLTK data
for pkg in ["vader_lexicon", "stopwords", "punkt", "punkt_tab",
            "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

# ─── Import dataset ───────────────────────────────────────────────────────────
from reviews_data import REVIEWS

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.DataFrame(REVIEWS)

def rating_to_sentiment(r):
    if r >= 4: return "Positive"
    elif r == 3: return "Neutral"
    else:       return "Negative"

df["true_label"] = df["rating"].apply(rating_to_sentiment)
print(f"Dataset loaded: {len(df)} reviews across {df['restaurant'].nunique()} restaurants\n")
print(df["true_label"].value_counts())
print()

# ─── 2. TEXT PREPROCESSING ────────────────────────────────────────────────────
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

df["clean_review"] = df["review"].apply(preprocess)

# ─── 3. VADER ─────────────────────────────────────────────────────────────────
sia = SentimentIntensityAnalyzer()

def vader_label(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:  return "Positive"
    elif score <= -0.05: return "Negative"
    else:               return "Neutral"

df["vader_label"]  = df["review"].apply(vader_label)
df["vader_score"]  = df["review"].apply(lambda t: sia.polarity_scores(t)["compound"])

# ─── 4. TEXTBLOB ──────────────────────────────────────────────────────────────
def textblob_label(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.05:   return "Positive"
    elif score < -0.05: return "Negative"
    else:               return "Neutral"

df["tb_label"] = df["review"].apply(textblob_label)
df["tb_score"] = df["review"].apply(lambda t: TextBlob(t).sentiment.polarity)

# ─── 5. LOGISTIC REGRESSION ───────────────────────────────────────────────────
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

# ─── 6. EVALUATION HELPER ─────────────────────────────────────────────────────
CLASSES = ["Negative", "Neutral", "Positive"]
y_true = df["true_label"]

def metrics(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))
    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

results = []
results.append(metrics(y_true, df["vader_label"], "VADER"))
results.append(metrics(y_true, df["tb_label"],    "TextBlob"))

# For LR, evaluate only on test split
y_true_lr_str = y_test.map(inv_map)
y_pred_lr_str = pd.Series(y_pred_lr).map(inv_map)
results.append(metrics(y_true_lr_str, y_pred_lr_str, "Logistic Regression (TF-IDF)"))

cv_scores = cross_val_score(lr, X_vec, y, cv=5, scoring="accuracy")
print(f"\nLogistic Regression 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

results_df = pd.DataFrame(results)
print("\n\nSummary Table:")
print(results_df.to_string(index=False))

# ─── 7. VISUALIZATIONS ────────────────────────────────────────────────────────
PALETTE = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}
MODEL_COLORS = ["#5C6BC0", "#26A69A", "#EF5350"]
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "#1E1E2E",
                     "axes.facecolor": "#1E1E2E", "axes.labelcolor": "white",
                     "xtick.color": "white", "ytick.color": "white",
                     "text.color": "white", "axes.titlecolor": "white",
                     "grid.color": "#333355"})

fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor("#1E1E2E")
fig.suptitle("Sentiment Analysis – Majitar, Sikkim Restaurant Reviews",
             fontsize=20, fontweight="bold", color="white", y=0.98)

# ── Plot 1: Sentiment Distribution (true labels) ──────────────────────────────
ax1 = fig.add_subplot(4, 3, 1)
counts = y_true.value_counts()
bars = ax1.bar(counts.index, counts.values,
               color=[PALETTE[c] for c in counts.index], edgecolor="#333355", linewidth=0.5)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(val), ha="center", va="bottom", fontsize=12, fontweight="bold", color="white")
ax1.set_title("True Sentiment Distribution", fontsize=12, fontweight="bold")
ax1.set_ylabel("Number of Reviews")
ax1.set_ylim(0, counts.max() + 5)
ax1.grid(axis="y", alpha=0.3)

# ── Plot 2: Sentiment by Restaurant ───────────────────────────────────────────
ax2 = fig.add_subplot(4, 3, 2)
rest_sent = df.groupby(["restaurant", "true_label"]).size().unstack(fill_value=0)
rest_sent = rest_sent.reindex(columns=["Negative", "Neutral", "Positive"], fill_value=0)
rest_sent.index = [r.replace("(Tourist Cafeteria)", "").replace("Restro & Lounge", "R&L")
                   .replace("Restro & Bar", "R&B").strip() for r in rest_sent.index]
rest_sent.plot(kind="bar", ax=ax2, color=[PALETTE[c] for c in rest_sent.columns],
               edgecolor="#333355", linewidth=0.5)
ax2.set_title("Sentiment by Restaurant", fontsize=12, fontweight="bold")
ax2.set_ylabel("Count")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=8)
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(axis="y", alpha=0.3)

# ── Plot 3: Average Rating by Restaurant ──────────────────────────────────────
ax3 = fig.add_subplot(4, 3, 3)
avg_rat = df.groupby("restaurant")["rating"].mean().sort_values(ascending=False)
avg_rat.index = [r.replace("(Tourist Cafeteria)", "").replace("Restro & Lounge", "R&L")
                 .replace("Restro & Bar", "R&B").strip() for r in avg_rat.index]
bars3 = ax3.barh(avg_rat.index, avg_rat.values,
                 color=MODEL_COLORS[0], edgecolor="#333355", linewidth=0.5)
for bar, val in zip(bars3, avg_rat.values):
    ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va="center", fontsize=9, color="white")
ax3.set_title("Average Rating by Restaurant", fontsize=12, fontweight="bold")
ax3.set_xlabel("Average Star Rating")
ax3.set_xlim(0, 6)
ax3.grid(axis="x", alpha=0.3)

# ── Plot 4: VADER Confusion Matrix ────────────────────────────────────────────
ax4 = fig.add_subplot(4, 3, 4)
cm_v = confusion_matrix(y_true, df["vader_label"], labels=CLASSES)
sns.heatmap(cm_v, annot=True, fmt="d", cmap="Blues", ax=ax4,
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, linecolor="#333355",
            annot_kws={"size": 12, "weight": "bold"})
ax4.set_title("VADER – Confusion Matrix", fontsize=12, fontweight="bold")
ax4.set_xlabel("Predicted"); ax4.set_ylabel("Actual")

# ── Plot 5: TextBlob Confusion Matrix ─────────────────────────────────────────
ax5 = fig.add_subplot(4, 3, 5)
cm_t = confusion_matrix(y_true, df["tb_label"], labels=CLASSES)
sns.heatmap(cm_t, annot=True, fmt="d", cmap="Greens", ax=ax5,
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, linecolor="#333355",
            annot_kws={"size": 12, "weight": "bold"})
ax5.set_title("TextBlob – Confusion Matrix", fontsize=12, fontweight="bold")
ax5.set_xlabel("Predicted"); ax5.set_ylabel("Actual")

# ── Plot 6: LR Confusion Matrix ───────────────────────────────────────────────
ax6 = fig.add_subplot(4, 3, 6)
cm_lr = confusion_matrix(y_true_lr_str, y_pred_lr_str, labels=CLASSES)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Reds", ax=ax6,
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, linecolor="#333355",
            annot_kws={"size": 12, "weight": "bold"})
ax6.set_title("Logistic Regression – Confusion Matrix", fontsize=12, fontweight="bold")
ax6.set_xlabel("Predicted"); ax6.set_ylabel("Actual")

# ── Plot 7: Model Comparison Bar Chart ────────────────────────────────────────
ax7 = fig.add_subplot(4, 3, (7, 8))
metrics_cols = ["Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(metrics_cols))
width = 0.25
for i, (_, row) in enumerate(results_df.iterrows()):
    vals = [row[c] for c in metrics_cols]
    bars7 = ax7.bar(x + i*width, vals, width, label=row["Model"],
                    color=MODEL_COLORS[i], edgecolor="#333355", linewidth=0.5)
    for bar, val in zip(bars7, vals):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="white")
ax7.set_title("Model Performance Comparison", fontsize=12, fontweight="bold")
ax7.set_xticks(x + width); ax7.set_xticklabels(metrics_cols, fontsize=10)
ax7.set_ylim(0, 1.15); ax7.set_ylabel("Score")
ax7.legend(fontsize=9); ax7.grid(axis="y", alpha=0.3)

# ── Plot 8: VADER Score Distribution ─────────────────────────────────────────
ax8 = fig.add_subplot(4, 3, 9)
for sentiment, grp in df.groupby("true_label"):
    ax8.hist(grp["vader_score"], bins=12, alpha=0.7,
             label=sentiment, color=PALETTE[sentiment], edgecolor="#333355")
ax8.axvline(0.05, color="yellow", linestyle="--", linewidth=1, label="Threshold +0.05")
ax8.axvline(-0.05, color="orange", linestyle="--", linewidth=1, label="Threshold -0.05")
ax8.set_title("VADER Compound Score Distribution", fontsize=12, fontweight="bold")
ax8.set_xlabel("Compound Score"); ax8.set_ylabel("Frequency")
ax8.legend(fontsize=7); ax8.grid(alpha=0.3)

# ── Plot 9: TextBlob Polarity vs Subjectivity ─────────────────────────────────
ax9 = fig.add_subplot(4, 3, 10)
df["tb_subjectivity"] = df["review"].apply(lambda t: TextBlob(t).sentiment.subjectivity)
for sentiment, grp in df.groupby("true_label"):
    ax9.scatter(grp["tb_score"], grp["tb_subjectivity"],
                label=sentiment, color=PALETTE[sentiment], alpha=0.8, s=60, edgecolors="white", linewidths=0.3)
ax9.axvline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
ax9.set_title("TextBlob: Polarity vs Subjectivity", fontsize=12, fontweight="bold")
ax9.set_xlabel("Polarity"); ax9.set_ylabel("Subjectivity")
ax9.legend(fontsize=8); ax9.grid(alpha=0.3)

# ── Plot 10: Cross-Validation Scores ─────────────────────────────────────────
ax10 = fig.add_subplot(4, 3, 11)
ax10.plot(range(1, 6), cv_scores, marker="o", color="#5C6BC0",
          linewidth=2, markersize=8, markerfacecolor="white", markeredgecolor="#5C6BC0", markeredgewidth=2)
ax10.axhline(cv_scores.mean(), color="#EF5350", linestyle="--",
             linewidth=1.5, label=f"Mean = {cv_scores.mean():.4f}")
ax10.fill_between(range(1, 6),
                  cv_scores.mean() - cv_scores.std(),
                  cv_scores.mean() + cv_scores.std(),
                  alpha=0.2, color="#5C6BC0", label=f"±1 Std = {cv_scores.std():.4f}")
for i, sc in enumerate(cv_scores):
    ax10.text(i+1, sc + 0.01, f"{sc:.3f}", ha="center", fontsize=8, color="white")
ax10.set_title("LR 5-Fold Cross-Validation Accuracy", fontsize=12, fontweight="bold")
ax10.set_xlabel("Fold"); ax10.set_ylabel("Accuracy")
ax10.set_ylim(0, 1.2); ax10.legend(fontsize=8); ax10.grid(alpha=0.3)

# ── Plot 11: Pie – Sentiment Share per Restaurant ─────────────────────────────
ax11 = fig.add_subplot(4, 3, 12)
overall = y_true.value_counts()
wedges, texts, autotexts = ax11.pie(
    overall.values,
    labels=overall.index,
    colors=[PALETTE[c] for c in overall.index],
    autopct="%1.1f%%", startangle=140,
    wedgeprops={"edgecolor": "#1E1E2E", "linewidth": 2},
    textprops={"color": "white", "fontsize": 10})
for at in autotexts:
    at.set_fontweight("bold")
ax11.set_title("Overall Sentiment Share", fontsize=12, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("sentiment_results.png", dpi=150, bbox_inches="tight",
            facecolor="#1E1E2E")
print("\nPlot saved to sentiment_results.png")
print("\n💡 TIP: Run  python food_recommender.py  for interactive food-based restaurant recommendations!")

