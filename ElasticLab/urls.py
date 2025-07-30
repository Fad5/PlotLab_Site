from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('analysis.urls')),  # Замените 'analysis' на имя вашего приложения
]