from django import forms
from .models import AnalysisTask

class AnalysisForm(forms.ModelForm):
    class Meta:
        model = AnalysisTask
        fields = ['data_archive', 'excel_file']
        widgets = {
            'data_archive': forms.FileInput(attrs={
                'accept': '.zip,.rar,.7z',
                'class': 'form-control'
            }),
            'excel_file': forms.FileInput(attrs={
                'accept': '.xlsx,.xls,.csv',
                'class': 'form-control'
            })
        }