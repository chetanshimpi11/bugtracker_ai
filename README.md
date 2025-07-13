# 🐞 AI-Powered Bug Tracking System

An AI-integrated Django web application to auto-predict bug **category** (UI Bug, Crash Bug, etc.) 
and **priority** (High/Medium/Low) using Machine Learning models.

---

## 🚀 Features

- 🧠 AI-based prediction of bug category & priority using Multinomial Naive Bayes
- 📄 Bug submission form with title, description & assignee
- 🎯 Real-time predictions integrated using `.pkl` models
- 🎨 Responsive UI using Bootstrap
- ✅ Bug status tracking (Open, In Progress, Closed)
- 🔐 Django admin panel for bug/user management

---

## 🧪 Machine Learning Models

- **Model 1**: Bug Category Prediction
- **Model 2**: Bug Priority Prediction
- **Vectorization**: `TfidfVectorizer`
- Trained using labeled bug report CSVs

---


---

## 🛠️ Tech Stack

- Python, Django, HTML, CSS, Bootstrap
- Pandas, NumPy, Scikit-learn, Joblib
- SQLite, Django ORM

---

## 📦 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/bugtracker_ai.git
cd bugtracker_ai

# 2. (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train ML models
python train_model.py

# 5. Run server
python manage.py runserver

