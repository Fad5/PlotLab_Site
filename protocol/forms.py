from django import forms

class SimpleUploadForm(forms.Form):
    data_archive = forms.FileField(
        label='Архив с данными',
        widget=forms.FileInput(attrs={
            'accept': '.zip,.rar,.7z',
            'class': 'form-control'
        })
    )
    
    excel_file = forms.FileField(
        label='Excel файл с параметрами',
        widget=forms.FileInput(attrs={
            'accept': '.xlsx,.xls,.csv',
            'class': 'form-control'
        })
    )
    
    def clean(self):
        cleaned_data = super().clean()
        data_archive = cleaned_data.get('data_archive')
        excel_file = cleaned_data.get('excel_file')
        
        if data_archive and excel_file:
            # Проверка расширения архива
            archive_ext = data_archive.name.split('.')[-1].lower()
            if archive_ext not in ['zip', 'rar', '7z']:
                raise forms.ValidationError("Поддерживаемые форматы архивов: .zip, .rar, .7z")
            
            # Проверка расширения Excel файла
            excel_ext = excel_file.name.split('.')[-1].lower()
            if excel_ext not in ['xlsx', 'xls', 'csv']:
                raise forms.ValidationError("Поддерживаемые форматы файлов: .xlsx, .xls, .csv")
        
        return cleaned_data