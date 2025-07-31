from django.contrib import admin
from django.urls import path, include
from analysis import views

urlpatterns = [
    path('', views.home, name='home'),
    path('moduł-younga/', views.analyze_young_modulus, name='moduł_younga'),
    path('box/', views.box_san, name='box_san'),
    path('about/', views.about, name='about'),
    path('PPU-Testus/', views.vibration_analysis_, name='PPU_Testus'),
    path('servo/', views.Servo, name='Servo'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('chart-data/', views.chart_data, name='chart_data'),
    path('compression/', views.compression, name='compression'),
    path('flex_analysis/', views.flex_analysis_view, name='flex_analysis_view'),
    path('razr/',views.tensile_test_view, name='razr'),
    path('on_load/',views.on_load, name='on_load'),
    path('help/', include('help.urls')), 
    path('protocol/', include('protocol.urls')), 
]