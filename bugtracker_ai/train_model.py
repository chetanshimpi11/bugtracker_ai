import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 🔸 Create ml_models folder if not exists
os.makedirs("ml_models", exist_ok=True)

# ================== PRIORITY MODEL ==================

print("🚀 Training Priority Model...")

# Load priority_data.csv
df_pri = pd.read_csv("priority_data.csv", encoding='latin1')  # if utf-8 fails

# Combine title + description for training
df_pri['text'] = df_pri['title'] + " " + df_pri['description']

# Split data
Xp = df_pri['text']
yp = df_pri['priority']
Xp_train, _, yp_train, _ = train_test_split(Xp, yp, test_size=0.2, random_state=42)

# Vectorize
vectorizer_priority = TfidfVectorizer()
Xp_train_vec = vectorizer_priority.fit_transform(Xp_train)

# Train model
priority_model = MultinomialNB()
priority_model.fit(Xp_train_vec, yp_train)

# Save
joblib.dump(priority_model, "ml_models/bug_priority_model.pkl")
joblib.dump(vectorizer_priority, "ml_models/vectorizer_priority.pkl")
print("✅ Priority model and vectorizer saved to ml_models/")

# ================== CATEGORY MODEL ==================

print("\n🚀 Training Category Model...")

# Load category_data.csv
df_cat = pd.read_csv("category_data.csv", encoding='latin1')  # or utf-8

# Split data
Xc = df_cat['bug_report']
yc = df_cat['category']
Xc_train, _, yc_train, _ = train_test_split(Xc, yc, test_size=0.2, random_state=42)

# Vectorize
vectorizer_category = TfidfVectorizer()
Xc_train_vec = vectorizer_category.fit_transform(Xc_train)

# Train model
category_model = MultinomialNB()
category_model.fit(Xc_train_vec, yc_train)

# Save
joblib.dump(category_model, "ml_models/bug_model.pkl")
joblib.dump(vectorizer_category, "ml_models/vectorizer.pkl")
print("✅ Category model and vectorizer saved to ml_models/")

print("\n🎉 All Models Trained Successfully!")
