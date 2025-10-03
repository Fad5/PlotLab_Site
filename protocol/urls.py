from django.urls import path
from .views import (UploadView, ResultsView, download_rar_press_union,download_excel_press_union,
                    download_report, protocol, Press_Protocols_Stubs, download_template, 
                    download_excel_press, generate_and_download_protocols, OD_generate, OD_elone,
                    download_excel_OD_elone, download_template_OD_elone, VibrationAnalysisView)

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
    path('generate/', generate_and_download_protocols, name='generate_protocols'),
    path('download/', generate_and_download_protocols, name='download_protocols'),
    path('OD_generate/', OD_generate, name='OD_generate'),
    path('OD_elone/', OD_elone, name='OD_elone'),
    path('download-excel-OD-elone/', download_excel_OD_elone, name='download_excel_OD_elone'),
    path('download-template-OD-elone/', download_template_OD_elone, name='download_template_OD_elone'),
    path('bbb/', VibrationAnalysisView.as_view(), name='ppu_testus_all')
]