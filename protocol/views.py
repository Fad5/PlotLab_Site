import os
import zipfile
import tempfile
from django.shortcuts import render, redirect
from django.views.generic import FormView, DetailView
from django.urls import reverse_lazy
from django.http import FileResponse
from django.contrib import messages
from .forms import AnalysisForm
from .models import AnalysisTask
from .utils.protocol import genaretion_plot
import pandas as pd

class UploadView(FormView):
    template_name = 'protocol/upload.html'
    form_class = AnalysisForm
    success_url = reverse_lazy('analysis_list')
    
    def form_valid(self, form):
        task = form.save()
        self.process_task(task)
        messages.success(self.request, 'Файлы загружены. Идет обработка...')
        return redirect('analysis_results', pk=task.id)
    
    def process_task(self, task):
        try:
            task.status = 'processing'
            task.save()
            
            # Создаем временные файлы
            with tempfile.TemporaryDirectory() as temp_dir:
                # 1. Обработка архива с данными
                data_files = []
                try:
                    with zipfile.ZipFile(task.data_archive.path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                        data_files = [
                            os.path.join(temp_dir, f) 
                            for f in zip_ref.namelist() 
                            if f.endswith('.txt')
                        ]
                        
                    if not data_files:
                        raise ValueError("В архиве не найдены файлы .txt")
                        
                except Exception as e:
                    task.status = 'failed'
                    task.save()
                    print(f"Ошибка обработки архива: {str(e)}")
                    return

                # 2. Обработка Excel файла
                try:
                    excel_data = pd.read_excel(task.excel_file.path)
                    required_columns = ['Образец', 'Ширина', 'Длина', 'Высота', 'Масса']
                    
                    if not all(col in excel_data.columns for col in required_columns):
                        missing = [col for col in required_columns if col not in excel_data.columns]
                        raise ValueError(f"В Excel файле отсутствуют обязательные колонки: {missing}")
                        
                except Exception as e:
                    task.status = 'failed'
                    task.save()
                    print(f"Ошибка чтения Excel файла: {str(e)}")
                    return

                # 3. Генерация отчета
                try:
                    output_path = os.path.join(temp_dir, 'report.docx')
                    
                    # Добавим логирование перед вызовом genaretion_plot
                    print(f"Начало генерации отчета. Файлов данных: {len(data_files)}")
                    
                    success = genaretion_plot(
                        data_files, 
                        excel_data, 
                        output_filename=output_path
                    )
                    
                    if not success:
                        raise ValueError("Не удалось сгенерировать отчет")
                    
                    # Сохраняем результат
                    with open(output_path, 'rb') as f:
                        task.result_file.save('report.docx', f)
                    
                    task.status = 'completed'
                    
                except Exception as e:
                    task.status = 'failed'
                    print(f"Ошибка генерации отчета: {str(e)}")
                    
        except Exception as e:
            task.status = 'failed'
            print(f"Неожиданная ошибка: {str(e)}")
            
        finally:
            task.save()
            print(f"Статус задачи обновлен: {task.status}")

class ResultsView(DetailView):
    model = AnalysisTask
    template_name = 'protocol/results.html'
    context_object_name = 'task'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['ready'] = self.object.status == 'completed'
        return context

def download_report(request, pk):
    task = AnalysisTask.objects.get(pk=pk)
    if task.result_file:
        response = FileResponse(task.result_file.open('rb'))
        response['Content-Disposition'] = f'attachment; filename="elastic_modulus_report.docx"'
        return response
    return redirect('analysis_results', pk=pk)