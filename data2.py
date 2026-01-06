# 1️⃣ Imports
import json
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 2️⃣ Load JSONL data
data = []
with open("problems_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# 3️⃣ Convert sample_io (list) → text
def sample_io_to_text(x):
    if isinstance(x, list):
        return " ".join(
            f"Input: {item.get('input', '')} Output: {item.get('output', '')}"
            for item in x
        )
    return ""

df["sample_io_text"] = df["sample_io"].apply(sample_io_to_text)

# 4️⃣ Create full_text
df["full_text"] = (
    df["title"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["input_description"].fillna("") + " " +
    df["output_description"].fillna("") + " " +
    df["sample_io_text"].fillna("")
)

# Text length
df["text_length"] = df["full_text"].apply(len)

# Keyword-based features
keywords = [
    "graph", "tree", "dp", "dynamic programming",
    "greedy", "binary search", "segment tree",
    "recursion", "bitmask"
]

for kw in keywords:
    df[f"kw_{kw}"] = df["full_text"].str.lower().str.count(kw)

# Constraint indicator (e.g., 10^5, 10^6)
df["has_constraints"] = df["full_text"].str.contains(r"10\^", regex=True).astype(int)


# 5️⃣ Prepare inputs & labels
X_text = df["full_text"]
y_class = df["problem_class"]

# 6️⃣ TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)
X_tfidf = tfidf.fit_transform(X_text)

numeric_features = df[
    ["text_length", "has_constraints"] +
    [f"kw_{kw}" for kw in keywords]
].values



# 7️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

# 8️⃣ Train classifier
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# 9️⃣ Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

y_score = df["problem_score"]
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_score,
    test_size=0.2,
    random_state=42
)

reg = LinearRegression()


reg.fit(X_train, y_train)
pred = reg.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
