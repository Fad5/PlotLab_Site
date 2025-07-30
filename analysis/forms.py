from django import forms

class AnalysisForm(forms.Form):
    data_file = forms.FileField(label='Data File (TXT/CSV)')
    width = forms.FloatField(label='Width (mm)', initial=100.54)
    length = forms.FloatField(label='Length (mm)', initial=100.83)
    height = forms.FloatField(label='Height (mm)', initial=28.83)
    sample_name = forms.CharField(label='Sample Name', max_length=100, initial="Sample")