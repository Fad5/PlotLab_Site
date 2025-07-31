from django.contrib import admin
from django.urls import path
from help import views

urlpatterns = [
    path('',views.help, name='help'),
    path('vibra_table/',views.vibra_table, name='vibra_table'),
    path('servo/',views.servo, name='servo'),
    path('pulsator/',views.pulsator, name='pulsator'),
    path('Gost_70261_2022/',views.PHP, name='PHP'),
    path('Gost_59940_2021/',views.Gost_59940_2021, name='Gost_59940_2021'),

]