import re
import string

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

df = pd.read_csv("support_tickets.csv")
print("Dataset shape:", df.shape)
print(df.head())

STOPWORDS = set("""a an the is are was were be been being to of in on for with and or but if
this that these those i you he she it we they my your his her its our their
me him us them do does did have has had not no nor so than too very can
will just don should now""".split())

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[0-9]+", " ", text)             
    text = text.translate(str.maketrans("", "", string.punctuation))  
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOPWORDS]
    return " ".join(tokens)


df["clean_text"] = df["ticket_text"].apply(clean_text)
print(df[["ticket_text", "clean_text"]].head())

vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["clean_text"])

y_category = df["category"]
y_priority = df["priority"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_category, test_size=0.2, random_state=42, stratify=y_category)

category_model = LogisticRegression(max_iter=1000)
category_model.fit(X_train, y_train)
cat_preds = category_model.predict(X_test)

print("\n===== CATEGORY MODEL RESULTS =====")
print("Accuracy :", round(accuracy_score(y_test, cat_preds), 3))
print("Precision:", round(precision_score(y_test, cat_preds, average="weighted"), 3))
print("Recall   :", round(recall_score(y_test, cat_preds, average="weighted"), 3))
print("F1 score :", round(f1_score(y_test, cat_preds, average="weighted"), 3))
print(classification_report(y_test, cat_preds))

cm_cat = confusion_matrix(y_test, cat_preds, labels=category_model.classes_)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_cat, annot=True, fmt="d", cmap="Blues",
            xticklabels=category_model.classes_, yticklabels=category_model.classes_)
plt.title("Confusion Matrix - Category Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_category.png")
plt.close()

Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X, y_priority, test_size=0.2, random_state=42, stratify=y_priority)

priority_model = MultinomialNB()
priority_model.fit(Xp_train, yp_train)
pri_preds = priority_model.predict(Xp_test)

print("\n===== PRIORITY MODEL RESULTS =====")
print("Accuracy :", round(accuracy_score(yp_test, pri_preds), 3))
print("Precision:", round(precision_score(yp_test, pri_preds, average="weighted"), 3))
print("Recall   :", round(recall_score(yp_test, pri_preds, average="weighted"), 3))
print("F1 score :", round(f1_score(yp_test, pri_preds, average="weighted"), 3))
print(classification_report(yp_test, pri_preds))

cm_pri = confusion_matrix(yp_test, pri_preds, labels=priority_model.classes_)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_pri, annot=True, fmt="d", cmap="Oranges",
            xticklabels=priority_model.classes_, yticklabels=priority_model.classes_)
plt.title("Confusion Matrix - Priority Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_priority.png")
plt.close()

def predict_new_ticket(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    category = category_model.predict(vec)[0]
    priority = priority_model.predict(vec)[0]
    return category, priority


sample_tickets = ["my payment got deducted twice this is urgent please refund asap",
    "hey just wondering if you guys have a student discount",
    "the app crashes every single time i open it, production is down",
    "i cant login, forgot my password and reset email never comes",]

print("\n===== SAMPLE PREDICTIONS =====")
for t in sample_tickets:
    cat, pri = predict_new_ticket(t)
    print(f"Ticket: {t}\n  -> Category: {cat} | Priority: {pri}\n")