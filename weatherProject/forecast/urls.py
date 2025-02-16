# forecast/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.weather_view, name='Weather View'),  # Call the correct function here
]
