from django.urls import path
from .views import (UploadView, ResultsView, download_rar_press_union,download_excel_press_union,
                    download_report, protocol, Press_Protocols_Stubs, download_template, download_excel_press)

urlpatterns = [
    path('', protocol, name='protocol'),
    path('upload/', UploadView.as_view(), name='upload'),
    path('upload/results/<uuid:pk>/', ResultsView.as_view(), name='analysis_results'),
    path('upload/download/<uuid:pk>/', download_report, name='download_report'),
    path('upload//Press_Protocols_Stubs', Press_Protocols_Stubs, name='Press_Protocols_Stubs'),
    path('download-template/', download_template, name='download_template'),
    path('download-template-excel-press/', download_excel_press, name='download_excel_press'),
    path('download-template-excel-press-union/', download_excel_press_union, name='download_excel_press_union'),
    path('download-template-zip-union/', download_rar_press_union, name='download_rar_press_union'),
]