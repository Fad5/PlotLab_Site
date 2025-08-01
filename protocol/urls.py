from django.urls import path
from .views import UploadView, ResultsView, download_report, protocol, Press_Protocols_Stubs, download_template

urlpatterns = [
    path('', protocol, name='protocol'),
    path('upload/', UploadView.as_view(), name='upload'),
    path('upload/results/<uuid:pk>/', ResultsView.as_view(), name='analysis_results'),
    path('upload/download/<uuid:pk>/', download_report, name='download_report'),
    path('upload//Press_Protocols_Stubs', Press_Protocols_Stubs, name='Press_Protocols_Stubs'),
    path('download-template/', download_template, name='download_template'),
]