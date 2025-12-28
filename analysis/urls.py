from django.urls import path, include
from analysis import views

urlpatterns = [
    path('', views.home, name='home'),
    path('moduł-younga/', views.analyze_young_modulus, name='moduł_younga'),
    path('box/', views.box_san, name='box_san'),
    path('about/', views.about, name='about'),
    path('servo/', views.Servo, name='Servo'),
    path('compression/', views.compression, name='compression'),
    path('flex_analysis/', views.flex_analysis_view, name='flex_analysis_view'),
    path('on_load/',views.on_load, name='on_load'),
    path('help/', include('help.urls')), 
    path('protocol/', include('protocol.urls')),
    path('aaa/', views.vibration_analysis___, name='vibro'),
]