from django.urls import path
from .views import dashboard, home

app_name = 'segmentation_app'

urlpatterns = [
    path('', dashboard, name='dashboard'),
    path('sth', home, name='home')
]
