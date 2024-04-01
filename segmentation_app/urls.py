from django.urls import path
from .views import dashboard, home, login, signup, verify, payment_initiate, semipremium

app_name = 'segmentation_app'

urlpatterns = [
    path('', dashboard, name='dashboard'),
    path('login', login, name='login'),
    path('signup', signup, name='signup'),
    path('customer_segmentation', home, name='home'),
    path('verify', verify, name='verify'),
    path('payment_initiate', payment_initiate, name='payment_initiate'),
    path('semipremium', semipremium, name='semipremium'),

]
