from django.urls import path
from protocol import views

urlpatterns = [
    path(' ',views.protocol, name='protocol'),
    path('vibra_protocol/',views.vibra_protocol, name='vibra_protocol'),
    path('press_protocol/',views.press_protocol, name='press_protocol'),
]