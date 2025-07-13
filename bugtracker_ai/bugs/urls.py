from django.urls import path
from . import views

urlpatterns = [
    path('submit/', views.submit_bug, name='submit_bug'),
    path('bugs/', views.bug_list, name='bug_list'),
]
