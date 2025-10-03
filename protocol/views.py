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
from .utils.protocol import genaretion_plot, generate_individual_protocols
from .utils.help_fun import reformat_date, dolg, generate_random_float, str_to_float, float_to_str
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from docxtpl import DocxTemplate
import os
from io import BytesIO
from django.core.files.uploadedfile import UploadedFile
from django.conf import settings

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
    """
    Функия для создания заглушек протоколов пресс
    """
    if request.method == 'POST':
 
            # Получаем загруженные файлы
            excel_file = request.FILES['excelFile']

            # Получаем данные с полей 
            percent_10_min = float(request.POST.get('percent_10_min'))
            percent_10_max = float(request.POST.get('percent_10_max'))
            percent_20_min = float(request.POST.get('percent_20_min'))
            percent_20_max = float(request.POST.get('percent_20_max'))
            percent_40_min = float(request.POST.get('percent_40_min'))
            percent_40_max = float(request.POST.get('percent_40_max'))
            
            sample_type = request.POST.get('sample_type')

            if sample_type  == 'square_plate':
                type_sample = 'Квадратная пластина'
            if sample_type  == 'rectangular_plate':
                type_sample = 'Прямоугольная пластина'

            # Генерация случайного числа 
            precent_10  = generate_random_float(percent_10_min, percent_10_max)
            precent_20  = generate_random_float(percent_20_min, percent_20_max)
            precent_40  = generate_random_float(percent_40_min, percent_40_max)
            
            # Обработка шаблона документа
            if 'docFile' in request.FILES:
                doc_file = request.FILES['docFile']
            else:
                # Используем шаблон по умолчанию
                default_template_path = os.path.join('templates_doc', 'template_press.docx')
                doc_file = open(default_template_path, 'rb')
            
            # Создаем временную директорию
            with tempfile.TemporaryDirectory() as temp_dir:
                # Сохраняем файлы во временную директорию
                excel_path = os.path.join(temp_dir, excel_file.name)
                
                # Для doc_file используем либо имя загруженного файла, либо имя файла по умолчанию
                if isinstance(doc_file, UploadedFile):  # Если это загруженный файл
                    doc_path = os.path.join(temp_dir, doc_file.name)
                else:  # Если это файл по умолчанию
                    doc_path = os.path.join(temp_dir, 'template_press.docx')
                
                # Сохраняем excel файл
                with open(excel_path, 'wb+') as f:
                    for chunk in excel_file.chunks():
                        f.write(chunk)
                
                # Сохраняем doc файл
                with open(doc_path, 'wb+') as f:
                    if isinstance(doc_file, UploadedFile):
                        for chunk in doc_file.chunks():
                            f.write(chunk)
                    else:
                        f.write(doc_file.read())
                        doc_file.close()  # Закрываем файл по умолчанию после чтения
                
                # Читаем Excel файл
                df = pd.read_excel(excel_path)
                
                # Создаем архив в памяти
                zip_buffer = BytesIO()
                
                try:
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        # Обрабатываем каждую строку в Excel
                        for index, row in df.iterrows():
                            # Получаем дату с excel файла
                            excel_data = row['Дата']
                            # Преобразуем дату 03.03.2000 в 3 марта 2000г.
                            data_prot = reformat_date(excel_data)
                            # Отображаем правильную должность в зависимости от даты 
                            dolg_ = dolg(excel_data)
                            print(dolg_)

                            context = {
                                'name_sample': row['Образец'],
                                'type_sample': type_sample,
                                'width': row['Ширина'],
                                'length': row['Длина'],
                                'height': row['Высота'],
                                'mass': row['Масса'],
                                'num_prot': row['Номер протокола'],
                                'data_prot': data_prot,
                                'dol': dolg_,
                                'precent_10':precent_10,
                                'precent_20':precent_20,
                                'precent_40':precent_40,
                            }
                            
                            # Загружаем шаблон
                            doc = DocxTemplate(doc_path)
                            doc.render(context)
                            
                            name_protocol = str(row['Номер протокола']).split('/')
                            if len(name_protocol) == 2:
                                name_protocol = name_protocol[0] + '-' + name_protocol[1]
                            else:
                                name_protocol = name_protocol[0]
                            
                            # Сохраняем документ во временный файл
                            temp_doc_path = os.path.join(temp_dir, f"{name_protocol}.docx")
                            doc.save(temp_doc_path)
                            
                            # Добавляем документ в архив
                            zipf.write(temp_doc_path, os.path.basename(temp_doc_path))
                    
                    # После выхода из блока with zipfile.ZipFile, буфер остается открытым
                    zip_buffer.seek(0)
                    
                    # Создаем ответ
                    response = HttpResponse(zip_buffer.getvalue(), content_type='application/zip')
                    response['Content-Disposition'] = 'attachment; filename="protocols_archive.zip"'
                    return response
                
                finally:
                    # Явно закрываем буфер после использования
                    zip_buffer.close()
        
    return render(request, 'protocol/3.html')


def download_template(request):
    """
    Функция для скачивания doc-шаблона,
    заглушки пресс
    """
    # Путь к файлу шаблона
    template_path = os.path.join(settings.BASE_DIR,'templates_doc', 'template_press.docx')
    print(template_path)
    # Открываем файл и возвращаем как ответ
    try:
        file = open(template_path, 'rb')
        response = FileResponse(file)
        response['Content-Type'] = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        response['Content-Disposition'] = 'attachment; filename="protocol_template.docx"'
        return response
    except FileNotFoundError:
        return HttpResponse("Файл шаблона не найден", status=404)

def download_excel_press(request):
    """
    Функция для скачивания excel файла в котором находятся геометрические харасктеристики,
    заглушки пресс
    """
    file_path = os.path.join(settings.BASE_DIR, 'templates_doc', 'excel_press.xlsx')
    return FileResponse(open(file_path, 'rb'), as_attachment=True)

def download_rar_press_union(request):
    """
    Функция для скачивания zip-архива в котором находятся файлы с испытаний,
    обеденненный пресс
    """
    file_path = os.path.join(settings.BASE_DIR, 'templates_doc', 'union.zip')
    return FileResponse(open(file_path, 'rb'), as_attachment=True)

def download_excel_press_union(request):
    """
    Функция для скачивания excel файла в котором находятся геометрические харасктеристики,
    обеденненный пресс
    """
    file_path = os.path.join(settings.BASE_DIR, 'templates_doc', 'union.xlsx')
    return FileResponse(open(file_path, 'rb'), as_attachment=True)


from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods

@require_http_methods(["GET", "POST"])  # Разрешаем оба метода
def generate_and_download_protocols(request):
    if request.method == 'GET':
        # Возвращаем просто HTML страницу для GET-запросов
        return render(request, 'protocol/upload_form.html')
    
    if request.method == 'POST':
        try:
            if 'excelFile' not in request.FILES:
                return JsonResponse({'error': 'Excel файл не был загружен'}, status=400)
            
            excel_file = request.FILES['excelFile']
            data_files = request.FILES.getlist('data_files')
            
            # Читаем Excel
            try:
                df = pd.read_excel(excel_file)
            except Exception as e:
                return JsonResponse({'error': f'Ошибка чтения Excel файла: {str(e)}'}, status=400)
            
            # Сохраняем файлы данных временно
            temp_files = []
            for f in data_files:
                temp_path = os.path.join(tempfile.gettempdir(), f.name)
                try:
                    with open(temp_path, 'wb+') as dest:
                        for chunk in f.chunks():
                            dest.write(chunk)
                    temp_files.append(temp_path)
                except Exception as e:
                    # Удаляем уже созданные временные файлы в случае ошибки
                    for temp_file in temp_files:
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    return JsonResponse({'error': f'Ошибка сохранения файла {f.name}: {str(e)}'}, status=500)
            
            # Генерируем протоколы
            try:
                response = generate_individual_protocols(
                    data_list=temp_files,
                    data_excel=df,
                    template_path='templates_doc/template_press.docx',
                    zip_response=True
                )
                
                # Если получили HttpResponse (архив)
                if isinstance(response, HttpResponse):
                    # Удаляем временные файлы
                    for f in temp_files:
                        try:
                            os.remove(f)
                        except:
                            pass
                    return response
                else:
                    raise Exception("Не удалось сгенерировать протоколы")
                
            except Exception as e:
                # Удаляем временные файлы в случае ошибки
                for f in temp_files:
                    try:
                        os.remove(f)
                    except:
                        pass
                return JsonResponse({'error': f'Ошибка генерации протоколов: {str(e)}'}, status=500)
                
        except Exception as e:
            return JsonResponse({'error': f'Неожиданная ошибка: {str(e)}'}, status=500)
        

def OD_generate(request):
    """
    Функия для создания заглушек протоколов пресс
    """
    if request.method == 'POST':
 
            # Получаем загруженные файлы
            excel_file = request.FILES['excelFile']

            # Получаем данные с полей 
            percent_min = float(request.POST.get('percent_min'))
            percent_max = float(request.POST.get('percent_max'))
            sample_type = request.POST.get('sample_type')

            if sample_type  == 'square_plate':
                type_sample = 'Квадратная пластина'
            if sample_type  == 'rectangular_plate':
                type_sample = 'Прямоугольная пластина'

            # Генерация случайного числа 
            procent  = generate_random_float(percent_min, percent_max)

            
            # Обработка шаблона документа
            if 'docFile' in request.FILES:
                doc_file = request.FILES['docFile']
            else:
                # Используем шаблон по умолчанию
                default_template_path = os.path.join('templates_doc', 'template_OD.docx')
                doc_file = open(default_template_path, 'rb')
            
            # Создаем временную директорию
            with tempfile.TemporaryDirectory() as temp_dir:
                # Сохраняем файлы во временную директорию
                excel_path = os.path.join(temp_dir, excel_file.name)
                
                # Для doc_file используем либо имя загруженного файла, либо имя файла по умолчанию
                if isinstance(doc_file, UploadedFile):  # Если это загруженный файл
                    doc_path = os.path.join(temp_dir, doc_file.name)
                else:  # Если это файл по умолчанию
                    doc_path = os.path.join(temp_dir, 'template_OD.docx')
                
                # Сохраняем excel файл
                with open(excel_path, 'wb+') as f:
                    for chunk in excel_file.chunks():
                        f.write(chunk)
                
                # Сохраняем doc файл
                with open(doc_path, 'wb+') as f:
                    if isinstance(doc_file, UploadedFile):
                        for chunk in doc_file.chunks():
                            f.write(chunk)
                    else:
                        f.write(doc_file.read())
                        doc_file.close()  # Закрываем файл по умолчанию после чтения
                
                # Читаем Excel файл
                df = pd.read_excel(excel_path)
                
                # Создаем архив в памяти
                zip_buffer = BytesIO()
                
                try:
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        # Обрабатываем каждую строку в Excel
                        for index, row in df.iterrows():
                            # Получаем дату с excel файла
                            excel_data = row['Дата']
                            # Преобразуем дату 03.03.2000 в 3 марта 2000г.
                            data_prot = reformat_date(excel_data)
                            # Отображаем правильную должность в зависимости от даты 
                            dolg_ = dolg(excel_data)

                            precent__ = float(procent.replace(',', '.'))

                            height_0  = str_to_float(row['Высота'])
                            height_1  =  height_0 - (height_0 * (precent__ / 100))
                            

                            context = {
                                'name_sample': row['Образец'],
                                'type_sample': type_sample,
                                'width': row['Ширина'],
                                'length': row['Длина'],
                                'height': row['Высота'],
                                'height_1': float_to_str(height_1),
                                'mass': row['Масса'],
                                'num_prot': row['Номер протокола'],
                                'data_prot': data_prot,
                                'dol': dolg_,
                                'procent': (procent),
                            }
                            
                            # Загружаем шаблон
                            doc = DocxTemplate(doc_path)
                            doc.render(context)
                            
                            name_protocol = str(row['Номер протокола']).split('/')
                            if len(name_protocol) == 2:
                                name_protocol = name_protocol[0] + '-' + name_protocol[1]
                            else:
                                name_protocol = name_protocol[0]
                            
                            # Сохраняем документ во временный файл
                            temp_doc_path = os.path.join(temp_dir, f"{name_protocol}.docx")
                            doc.save(temp_doc_path)
                            
                            # Добавляем документ в архив
                            zipf.write(temp_doc_path, os.path.basename(temp_doc_path))
                    
                    # После выхода из блока with zipfile.ZipFile, буфер остается открытым
                    zip_buffer.seek(0)
                    
                    # Создаем ответ
                    response = HttpResponse(zip_buffer.getvalue(), content_type='application/zip')
                    response['Content-Disposition'] = 'attachment; filename="protocols_archive.zip"'
                    return response
                
                finally:
                    # Явно закрываем буфер после использования
                    zip_buffer.close()
        
    return render(request, 'protocol/OD/OD.html')


def OD_elone(request):
    """
    Функия для создания протоколов остаточной деформации 
    """
    if request.method == 'POST':
 
            # Получаем загруженные файлы
            excel_file = request.FILES['excelFile']

            # Получаем данные с полей 
            sample_type = request.POST.get('sample_type')

            if sample_type  == 'square_plate':
                type_sample = 'Квадратная пластина'
            if sample_type  == 'rectangular_plate':
                type_sample = 'Прямоугольная пластина'

            # Генерация случайного числа 

            
            # Обработка шаблона документа
            if 'docFile' in request.FILES:
                doc_file = request.FILES['docFile']
            else:
                # Используем шаблон по умолчанию
                default_template_path = os.path.join('templates_doc', 'template_OD.docx')
                doc_file = open(default_template_path, 'rb')
            
            # Создаем временную директорию
            with tempfile.TemporaryDirectory() as temp_dir:
                # Сохраняем файлы во временную директорию
                excel_path = os.path.join(temp_dir, excel_file.name)
                
                # Для doc_file используем либо имя загруженного файла, либо имя файла по умолчанию
                if isinstance(doc_file, UploadedFile):  # Если это загруженный файл
                    doc_path = os.path.join(temp_dir, doc_file.name)
                else:  # Если это файл по умолчанию
                    doc_path = os.path.join(temp_dir, 'template_OD.docx')
                
                # Сохраняем excel файл
                with open(excel_path, 'wb+') as f:
                    for chunk in excel_file.chunks():
                        f.write(chunk)
                
                # Сохраняем doc файл
                with open(doc_path, 'wb+') as f:
                    if isinstance(doc_file, UploadedFile):
                        for chunk in doc_file.chunks():
                            f.write(chunk)
                    else:
                        f.write(doc_file.read())
                        doc_file.close()  # Закрываем файл по умолчанию после чтения
                
                # Читаем Excel файл
                df = pd.read_excel(excel_path)
                
                # Создаем архив в памяти
                zip_buffer = BytesIO()
                
                try:
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        # Обрабатываем каждую строку в Excel
                        for index, row in df.iterrows():
                            # Получаем дату с excel файла
                            excel_data = row['Дата']
                            # Преобразуем дату 03.03.2000 в 3 марта 2000г.
                            data_prot = reformat_date(excel_data)
                            # Отображаем правильную должность в зависимости от даты 
                            dolg_ = dolg(excel_data)

                            print(row['Высота_после'], type(row['Высота_после']), "row['Высота_после']")

                            height_0 = str_to_float(row['Высота'])
                            height_1 = str_to_float(row['Высота_после'])

                            procent = (1 - height_1 / height_0) * 100
                            

                            context = {
                                'name_sample': row['Образец'],
                                'type_sample': type_sample,
                                'width': row['Ширина'],
                                'length': row['Длина'],
                                'height': row['Высота'],
                                'height_1': float_to_str(row['Высота_после']),
                                'mass': row['Масса'],
                                'num_prot': row['Номер протокола'],
                                'data_prot': data_prot,
                                'dol': dolg_,
                                'procent': float_to_str(procent),
                            }
                            
                            # Загружаем шаблон
                            doc = DocxTemplate(doc_path)
                            doc.render(context)
                            
                            name_protocol = str(row['Номер протокола']).split('/')
                            if len(name_protocol) == 2:
                                name_protocol = name_protocol[0] + '-' + name_protocol[1]
                            else:
                                name_protocol = name_protocol[0]
                            
                            # Сохраняем документ во временный файл
                            temp_doc_path = os.path.join(temp_dir, f"{name_protocol}.docx")
                            doc.save(temp_doc_path)
                            
                            # Добавляем документ в архив
                            zipf.write(temp_doc_path, os.path.basename(temp_doc_path))
                    
                    # После выхода из блока with zipfile.ZipFile, буфер остается открытым
                    zip_buffer.seek(0)
                    
                    # Создаем ответ
                    response = HttpResponse(zip_buffer.getvalue(), content_type='application/zip')
                    response['Content-Disposition'] = 'attachment; filename="protocols_archive.zip"'
                    return response
                
                finally:
                    # Явно закрываем буфер после использования
                    zip_buffer.close()
        
    return render(request, 'protocol/OD/OD_elone.html')


def download_excel_OD_elone(request):
    """
    Функция для скачивания excel файла в котором находятся геометрические харасктеристики,
    заглушки пресс
    """
    file_path = os.path.join(settings.BASE_DIR, 'templates_doc', 'OD_elone.xlsx')
    return FileResponse(open(file_path, 'rb'), as_attachment=True)


def download_template_OD_elone(request):
    """
    Функция для скачивания doc-шаблона,
    заглушки пресс
    """
    # Путь к файлу шаблона
    template_path = os.path.join(settings.BASE_DIR,'templates_doc', 'template_OD.docx')
    print(template_path)
    # Открываем файл и возвращаем как ответ
    try:
        file = open(template_path, 'rb')
        response = FileResponse(file)
        response['Content-Type'] = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        response['Content-Disposition'] = 'attachment; filename="protocol_template.docx"'
        return response
    except FileNotFoundError:
        return HttpResponse("Файл шаблона не найден", status=404)
    

import shutil
import os
import tempfile
from django.shortcuts import render
from django.http import HttpResponse, FileResponse
from django.views import View
from django.core.files.storage import FileSystemStorage
import pandas as pd
from .vibro_table_all import process_excel_file, extract_archive, get_archive_files_list, get_file, vibraTableOne, create_full_report

class VibrationAnalysisView(View):
    def get(self, request):
        return render(request, 'protocol/pputestus_all.html')
    
    def post(self, request):
        try:
            # Получаем файлы из формы
            excel_file = request.FILES.get('excel_file')
            archive_file = request.FILES.get('archive_file')
            
            if not excel_file or not archive_file:
                return render(request, 'protocol/pputestus_all.html', {
                    'error': 'Пожалуйста, загрузите оба файла'
                })
            
            # Создаем временные файлы
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_excel:
                for chunk in excel_file.chunks():
                    tmp_excel.write(chunk)
                excel_path = tmp_excel.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_archive:
                for chunk in archive_file.chunks():
                    tmp_archive.write(chunk)
                archive_path = tmp_archive.name
            
            try:
                # Обрабатываем Excel файл
                samples_data, mass_columns = process_excel_file(excel_path)
                
                # Извлекаем архив
                temp_path = extract_archive(archive_path)
                
                # Получаем список файлов в архиве
                list_file = get_archive_files_list(archive_path)
                
                # Проверяем наличие всех необходимых файлов
                missing_files = []
                for sample_id, data in samples_data.items():
                    for i in data['name_files']:
                        if i not in list_file:
                            missing_files.append(i)
                
                if missing_files:
                    return render(request, 'protocol/pputestus_all.html', {
                        'error': f'В архиве отсутствуют файлы: {", ".join(missing_files)}'
                    })
                
                # Обрабатываем данные для каждого образца
                for sample_id, data in samples_data.items():
                    files = data['name_files'] 
                    list_files = get_file(temp_path, files)
                    
                    a = data['geometric_params']['length']
                    b = data['geometric_params']['width']
                    h = data['geometric_params']['height']
                    loads = mass_columns
                    heights = [item['value'] for item in data['masses'].values()]
                    
                    images, datas, results = vibraTableOne(sample_id, list_files, a, b, h, heights, loads)
                    
                    samples_data[sample_id]['images'] = images
                    samples_data[sample_id]['datas'] = datas
                    samples_data[sample_id]['results'] = results
                
                # Создаем отчет
                output_filename = 'vibration_analysis_report.docx'
                output_path = os.path.join(tempfile.gettempdir(), output_filename)
                create_full_report(samples_data, output_path)
                
                # Отправляем файл пользователю
                response = FileResponse(
                    open(output_path, 'rb'),
                    as_attachment=True,
                    filename=output_filename
                )
                
                # Очистка временных файлов (можно сделать через celery или background task)
                self.cleanup_temp_files([excel_path, archive_path, output_path])
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path, ignore_errors=True)
                
                return response
                
            except Exception as e:
                # Очистка в случае ошибки
                self.cleanup_temp_files([excel_path, archive_path])
                return render(request, 'protocol/pputestus_all.html', {
                    'error': f'Ошибка обработки: {str(e)}'
                })
                
        except Exception as e:
            return render(request, 'protocol/pputestus_all.html', {
                'error': f'Ошибка: {str(e)}'
            })
    
    def cleanup_temp_files(self, file_paths):
        """Очистка временных файлов"""

        for path in file_paths:
            try:
                if os.path.isfile(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
            except:
                pass