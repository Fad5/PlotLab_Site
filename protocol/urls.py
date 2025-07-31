from django.urls import path
from .views import UploadView, ResultsView, download_report

urlpatterns = [
    path('', UploadView.as_view(), name='upload'),
    path('results/<uuid:pk>/', ResultsView.as_view(), name='analysis_results'),
    path('download/<uuid:pk>/', download_report, name='download_report'),
]