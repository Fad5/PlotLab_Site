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
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from docxtpl import DocxTemplate
import os
from io import BytesIO

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


def protocol(request):
    return render(request, 'protocol/protocol.html')



def Press_Protocols_Stubs(request):
    if request.method == 'POST':
            # Получаем загруженные файлы
            excel_file = request.FILES['excelFile']
            doc_file = request.FILES['docFile']
            
            # Создаем временную директорию
            with tempfile.TemporaryDirectory() as temp_dir:
                # Сохраняем файлы во временную директорию
                excel_path = os.path.join(temp_dir, excel_file.name)
                doc_path = os.path.join(temp_dir, doc_file.name)
                
                with open(excel_path, 'wb+') as f:
                    for chunk in excel_file.chunks():
                        f.write(chunk)
                
                with open(doc_path, 'wb+') as f:
                    for chunk in doc_file.chunks():
                        f.write(chunk)
                
                # Читаем Excel файл
                df = pd.read_excel(excel_path)
                
                # Создаем архив в памяти
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Обрабатываем каждую строку в Excel
                    for index, row in df.iterrows():
                        # Создаем контекст для шаблона
                        context = {
                            'Образец': row['Образец'],
                            'Ширина': row['Ширина'],
                            'Длина': row['Длина'],
                            'Высота': row['Высота'],
                            'Масса': row['Масса'],
                            'Номер_протокола': row['Номер протокола'],
                            'Дата': row['Дата']
                        }
                        
                        # Загружаем шаблон
                        doc = DocxTemplate(doc_path)
                        doc.render(context)
                        
                        # Сохраняем документ во временный файл
                        temp_doc_path = os.path.join(temp_dir, f"Протокол_{row['Образец']}.docx")
                        doc.save(temp_doc_path)
                        
                        # Добавляем документ в архив
                        zipf.write(temp_doc_path, os.path.basename(temp_doc_path))
                
                # Возвращаем архив пользователю
                zip_buffer.seek(0)
                response = HttpResponse(zip_buffer, content_type='application/zip')
                response['Content-Disposition'] = 'attachment; filename="protocols_archive.zip"'
                return response
    
    return render(request, 'protocol/3.html')