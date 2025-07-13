import os
import joblib
from django.conf import settings
from django.shortcuts import render, redirect
from .models import Bug
from .forms import BugForm

# Load ML models and vectorizers
BASE_DIR = settings.BASE_DIR
vectorizer = joblib.load(os.path.join(BASE_DIR, 'ml_models/vectorizer.pkl'))
category_model = joblib.load(os.path.join(BASE_DIR, 'ml_models/bug_model.pkl'))
vectorizer_priority = joblib.load(os.path.join(BASE_DIR, 'ml_models/vectorizer_priority.pkl'))
priority_model = joblib.load(os.path.join(BASE_DIR, 'ml_models/bug_priority_model.pkl'))

# ========== Submit Bug View ==========
def submit_bug(request):
    if request.method == 'POST':
        form = BugForm(request.POST)
        if form.is_valid():
            bug = form.save(commit=False)

            # Combine title + description for prediction
            text = bug.title + " " + bug.description

            # Predict Category
            X1 = vectorizer.transform([text])
            bug.category = category_model.predict(X1)[0]

            # Predict Priority
            X2 = vectorizer_priority.transform([text])
            bug.priority = priority_model.predict(X2)[0]

            bug.save()
            return redirect('bug_list')
    else:
        form = BugForm()
    return render(request, 'submit_bug.html', {'form': form})

# ========== Bug List View ==========
def bug_list(request):
    bugs = Bug.objects.all().order_by('-id')
    return render(request, 'bug_list.html', {'bugs': bugs})
