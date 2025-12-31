from django.shortcuts import render, redirect
from django.views.generic import FormView, DetailView
from django.urls import reverse_lazy
from django.contrib import messages
from .forms import SimpleUploadForm
from .utils.protocol import genaretion_plot, generate_individual_protocols, genaretion_plot_with_saved_plots
from .utils.help_fun import reformat_date, dolg, generate_random_float, str_to_float, float_to_str
from docxtpl import DocxTemplate
from io import BytesIO
from django.core.files.uploadedfile import UploadedFile
from django.conf import settings
import shutil
import os
import tempfile
import zipfile
import xlsxwriter
from django.http import HttpResponse, FileResponse
from django.views import View
from .vibro_table_all import process_excel_file, extract_archive, get_archive_files_list, get_file, vibraTableOne, create_full_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class UploadView(View):
    template_name = 'protocol/upload.html'
    
    def get(self, request):
        form = SimpleUploadForm()
        return render(request, self.template_name, {'form': form})
    
    def post(self, request):
        form = SimpleUploadForm(request.POST, request.FILES)
        
        if not form.is_valid():
            return render(request, self.template_name, {'form': form})
        
        excel_file = request.FILES.get('excel_file')
        data_archive = request.FILES.get('data_archive')
        
        if not excel_file or not data_archive:
            messages.error(request, 'Пожалуйста, загрузите оба файла')
            return render(request, self.template_name, {'form': form})
        
        try:
            # Создаем временную директорию
            temp_dir = tempfile.mkdtemp()
            print(f"Создана временная директория: {temp_dir}")
            
            # 1. Обработка архива с данными
            data_files = []
            
            archive_path = os.path.join(temp_dir, 'uploaded_archive.zip')
            with open(archive_path, 'wb') as f:
                for chunk in data_archive.chunks():
                    f.write(chunk)
            
            try:
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                    data_files = [
                        os.path.join(temp_dir, f) 
                        for f in zip_ref.namelist() 
                        if f.endswith('.txt') or f.endswith('.csv')
                    ]
                
                if not data_files:
                    raise ValueError("В архиве не найдены файлы данных (.txt или .csv)")
                    
            finally:
                if os.path.exists(archive_path):
                    os.unlink(archive_path)
            
            # 2. Обработка Excel файла
            excel_path = os.path.join(temp_dir, 'uploaded_excel.xlsx')
            with open(excel_path, 'wb') as f:
                for chunk in excel_file.chunks():
                    f.write(chunk)
            
            try:
                excel_data = pd.read_excel(excel_path)
                required_columns = ['Образец', 'Ширина', 'Длина', 'Высота', 'Масса']
                
                if not all(col in excel_data.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in excel_data.columns]
                    raise ValueError(f"В Excel файле отсутствуют обязательные колонки: {missing}")
                    
            finally:
                if os.path.exists(excel_path):
                    os.unlink(excel_path)
            
            # 3. Генерация отчета и графиков
            print(f"Начало генерации отчета. Файлов данных: {len(data_files)}")
            
            # Создаем поддиректорию для выходных файлов
            output_dir = os.path.join(temp_dir, 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            # Путь для Word-протокола
            protocol_path = os.path.join(output_dir, 'vibration_protocol.docx')
            
            # Генерируем отчет и получаем пути к графикам и Excel файлам
            success, plot_dirs, excel_files = genaretion_plot_with_saved_plots(
                data_files, 
                excel_data, 
                output_dir=output_dir,
                output_filename=protocol_path
            )
            
            if not success:
                raise ValueError("Не удалось сгенерировать отчет")
            
            # 4. Создаем финальный архив
            archive_name = 'vibration_analysis_results.zip'
            archive_path = os.path.join(temp_dir, archive_name)
            
            # Создаем ZIP архив с сохранением структуры папок
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Добавляем протокол в корень архива
                if os.path.exists(protocol_path):
                    zipf.write(protocol_path, arcname='vibration_protocol.docx')
                
                # Добавляем папки с графиками и Excel файлами
                if plot_dirs:
                    for sample_name, plot_dir in plot_dirs.items():
                        if os.path.exists(plot_dir):
                            for root, dirs, files in os.walk(plot_dir):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.relpath(file_path, output_dir)
                                    zipf.write(file_path, arcname=arcname)
            
            # 5. Читаем архив в память и создаем ответ
            with open(archive_path, 'rb') as f:
                archive_content = f.read()
            
            # Создаем HttpResponse с содержимым архива
            response = HttpResponse(
                archive_content,
                content_type='application/zip'
            )
            response['Content-Disposition'] = f'attachment; filename="{archive_name}"'
            response['Content-Length'] = len(archive_content)
            
            # 6. Удаляем временные файлы вручную после отправки ответа
            def cleanup_temp_files():
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        print(f"Очищена временная директория: {temp_dir}")
                except Exception as e:
                    print(f"Ошибка при очистке временных файлов: {e}")
            
            return response
            
        except Exception as e:
            messages.error(request, f'Ошибка при обработке файлов: {str(e)}')
            # Очищаем временную директорию при ошибке
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            return render(request, self.template_name, {'form': form})
            
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
    

def download_excel_vibrotable_all(request):
    """
    Функция для скачивания excel файла в котором находятся геометрические харасктеристики,
    обеденненный вибростол
    """
    file_path = os.path.join(settings.BASE_DIR, 'templates_doc', 'Exemple_vibrotable_all.xlsx')
    return FileResponse(open(file_path, 'rb'), as_attachment=True)

def download_rar_vibrotable_all(request):
    """
    Функция для скачивания zip-архива в котором находятся файлы с испытаний,
    обеденненный вибростол
    """
    file_path = os.path.join(settings.BASE_DIR, 'templates_doc', 'Exemple_vibrotable_all.zip')
    return FileResponse(open(file_path, 'rb'), as_attachment=True)


class VibrationAnalysisView(View):
    def get(self, request):
        return render(request, 'protocol/pputestus_all.html')
    
    def post(self, request):

            # Получаем файлы из формы
            excel_file = request.FILES.get('excel_file')
            archive_file = request.FILES.get('archive_file')
            limit_HZ = int(request.POST.get('limit_HZ'))

            
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

                # Создаем временную папку для всех файлов
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                temp_dir = tempfile.mkdtemp(prefix=f'vibration_report_{timestamp}_')
                
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
                    
                    images, datas, results = vibraTableOne(sample_id, list_files, a, b, h, heights, loads, limits=(0, limit_HZ))
                    
                    samples_data[sample_id]['images'] = images
                    samples_data[sample_id]['datas'] = datas
                    samples_data[sample_id]['results'] = results
                
                # 1. СОЗДАЕМ ПРОТОКОЛ В WORD
                docx_filename = 'vibration_analysis_report.docx'
                docx_path = os.path.join(temp_dir, docx_filename)
                create_full_report(samples_data, docx_path)
                
                # 2. СОХРАНЯЕМ ГРАФИКИ
                saved_graphs = self.save_sample_graphs(samples_data, temp_dir)
                print(f"✓ Сохранено {len(saved_graphs)} графиков")
                
                # 3. СОЗДАЕМ EXCEL ФАЙЛЫ
                saved_excel_files = self.save_sample_excel(samples_data, temp_dir)
                print(f"✓ Создано {len(saved_excel_files)} Excel файлов")
                
                # 4. СОЗДАЕМ ZIP АРХИВ
                zip_filename = f'vibration_analysis_report_{timestamp}.zip'
                zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Добавляем все файлы из временной папки
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Относительный путь внутри архива
                            arcname = os.path.relpath(file_path, temp_dir)
                            zipf.write(file_path, arcname)
                
                # Отправляем ZIP архив пользователю
                response = FileResponse(
                    open(zip_path, 'rb'),
                    as_attachment=True,
                    filename=zip_filename
                )
                
                # Очистка временных файлов
                self.cleanup_temp_files([excel_path, archive_path, zip_path, temp_dir])
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path, ignore_errors=True)
                
                return response

    
    def save_sample_graphs(self, samples_data, output_dir):
        """
        Сохраняет графики для каждого образца
        """
        graphs_dir = os.path.join(output_dir, "Данные")
        os.makedirs(graphs_dir, exist_ok=True)
        self.graphs_dir = graphs_dir
        saved_graphs = []
        
        for sample_id, data in samples_data.items():
            if 'images' not in data:
                continue
            
            # Создаем папку для данного образца
            sample_dir = os.path.join(graphs_dir, f"Образец_{sample_id}")
            os.makedirs(sample_dir, exist_ok=True)
            
            for image_name, fig in data['images'].items():
                try:
                    # Сохраняем график
                    graph_path = os.path.join(sample_dir, f"{image_name}.png")
                    fig.savefig(graph_path, dpi=300, bbox_inches='tight')
                    saved_graphs.append(graph_path)
                    
                    # Закрываем figure чтобы освободить память
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"Ошибка при сохранении графика {image_name}: {e}")
        
        return saved_graphs
    
    def save_sample_excel(self, samples_data, output_dir):
        """
        Сохраняет Excel файлы с данными для каждого образца
        """
        excel_dir = os.path.join(output_dir, self.graphs_dir)
        os.makedirs(excel_dir, exist_ok=True)
        
        saved_excel_files = []
        
        # 2. Excel файлы для каждого образца (графики)
        for sample_id, data in samples_data.items():
            if 'datas' not in data:
                continue
            
            try:
                # Создаем Excel файл для данного образца
                excel_path = os.path.join(excel_dir, f"Образец_{sample_id}", f"Данные_образец_{sample_id}.xlsx")
                os.makedirs(os.path.dirname(excel_path), exist_ok=True)
                workbook = xlsxwriter.Workbook(excel_path)
                
                # Для каждой массы создаем отдельный лист
                for mass_str, graph_data in data['datas'].items():
                    try:
                        if len(graph_data) >= 3:
                            freq, tf_module, isolation_eff = graph_data[:3]
                            
                            # Создаем лист с именем массы
                            sheet_name = f"Масса_{mass_str}кг"
                            worksheet = workbook.add_worksheet(sheet_name)
                            
                            # Заголовки
                            headers = ['Частота, Гц', 'Передаточная функция', 'Эффективность, дБ']
                            for col, header in enumerate(headers):
                                worksheet.write(0, col, header)
                            
                            # Данные
                            for row in range(min(len(freq), len(tf_module), len(isolation_eff))):
                                worksheet.write(row + 1, 0, float(freq[row]))
                                worksheet.write(row + 1, 1, float(tf_module[row]))
                                worksheet.write(row + 1, 2, float(isolation_eff[row]))
                            
                            # Если есть результаты для этой массы - размещаем справа
                            if 'results' in data and mass_str in data['results']:
                                pressure, Fpeak, Ed, damp = data['results'][mass_str]
                                
                                # Определяем начальную позицию для результатов (справа от таблицы)
                                results_col_start = 5  # Начинаем с колонки E (индекс 4) для отступа
                                
                                worksheet.write(0, results_col_start, 'Результаты испытаний:')
                                worksheet.write(1, results_col_start, 'Удельное давление, кПа')
                                worksheet.write(1, results_col_start + 1, float(pressure))
                                worksheet.write(2, results_col_start, 'Резонансная частота, Гц')
                                worksheet.write(2, results_col_start + 1, float(Fpeak))
                                worksheet.write(3, results_col_start, 'Динамический модуль упругости, Н/мм²')
                                worksheet.write(3, results_col_start + 1, float(Ed))
                                worksheet.write(4, results_col_start, 'Коэффициент потерь')
                                worksheet.write(4, results_col_start + 1, float(damp))
                                
                    except (ValueError, TypeError, IndexError) as e:
                        print(f"Ошибка при обработке данных для образца {sample_id}, масса {mass_str}: {e}")
                        continue
                
                workbook.close()
                saved_excel_files.append(excel_path)
                print(f"✓ Создан Excel файл: {excel_path}")
                
            except Exception as e:
                print(f"Ошибка при создании Excel файла для образца {sample_id}: {e}")

        return saved_excel_files
    
    def cleanup_temp_files(self, file_paths):
        """Очистка временных файлов"""
        for path in file_paths:
            try:
                if path and os.path.exists(path):
                    if os.path.isfile(path):
                        os.unlink(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
            except:
                pass


class VibrationUploadView(View):
    template_name = 'protocol/ww.html'
    
    def get(self, request):
        return render(request, self.template_name)
    
    def analyze_vibration_data(self, data_file, sample_name, output_dir):
        """Анализ данных вибрационных испытаний (полный аналог MATLAB скрипта)"""
        
        try:
            print(f"\n{'='*60}")
            print(f"Начинаем анализ файла: {data_file}")
            print(f"Название образца: {sample_name}")
            print(f"Выходная директория: {output_dir}")
            print(f"{'='*60}")
            
            # ЧТЕНИЕ ДАННЫХ ИЗ ФАЙЛА
            # Пробуем разные методы чтения для обработки различных форматов
            
            def clean_number_string(num_str):
                """Очистка строки с числом"""
                if not isinstance(num_str, str):
                    return str(num_str)
                
                # Убираем лишние пробелы
                cleaned = num_str.strip()
                
                # Если строка содержит табуляцию, берем первую часть
                if '\t' in cleaned:
                    cleaned = cleaned.split('\t')[0]
                
                # Заменяем запятую на точку для десятичных чисел
                cleaned = cleaned.replace(',', '.')
                
                # Убираем лишние нули в начале
                if cleaned.startswith('0') and len(cleaned) > 1 and cleaned[1] != '.':
                    # Находим первую ненулевую цифру
                    for i, char in enumerate(cleaned):
                        if char != '0' and char != '.':
                            cleaned = cleaned[i:]
                            break
                    if cleaned.startswith('.'):
                        cleaned = '0' + cleaned
                
                # Если после очистки пустая строка, возвращаем '0'
                if cleaned == '':
                    cleaned = '0'
                    
                return cleaned
            
            # Метод 1: Чтение с помощью pandas с обработкой различных разделителей
            try:
                print("Попытка 1: Чтение с pandas...")
                
                # Сначала пробуем определить разделитель
                with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline()
                    second_line = f.readline()
                
                # Определяем разделитель
                if '\t' in first_line and '\t' in second_line:
                    sep = '\t'
                    print("Определен разделитель: табуляция (\\t)")
                elif ',' in first_line and ',' in second_line:
                    sep = ','
                    print("Определен разделитель: запятая (,)")
                else:
                    sep = None  # Автоопределение
                    print("Разделитель: автоопределение")
                
                # Пробуем прочитать файл
                data = pd.read_csv(
                    data_file, 
                    sep=sep,
                    decimal='.',  # Будем заменять запятые на точки
                    header=None,
                    engine='python',
                    encoding='utf-8',
                    on_bad_lines='skip',
                    dtype=str,  # Читаем как строки для последующей обработки
                    skip_blank_lines=True
                )
                
                print(f"Успешно прочитано строк: {data.shape[0]}, колонок: {data.shape[1]}")
                
                # Конвертируем строки в числа
                numeric_data = []
                for i in range(data.shape[0]):
                    row = []
                    for j in range(min(data.shape[1], 7)):  # Берем максимум 7 колонок
                        try:
                            value_str = str(data.iloc[i, j])
                            cleaned = clean_number_string(value_str)
                            value = float(cleaned)
                            row.append(value)
                        except:
                            row.append(np.nan)
                    numeric_data.append(row)
                
                data = pd.DataFrame(numeric_data)
                
            except Exception as e1:
                print(f"Ошибка при чтении pandas: {e1}")
                print("Попытка 2: Ручное чтение файла...")
                
                # Метод 2: Ручное чтение файла
                with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                parsed_data = []
                for line_num, line in enumerate(lines):
                    if line_num > 50000:  # Ограничение для производительности
                        break
                        
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Разбиваем строку на части
                    parts = []
                    
                    # Пробуем разные методы разделения
                    if '\t' in line:
                        parts = line.split('\t')
                    elif ',' in line:
                        parts = line.split(',')
                    elif ';' in line:
                        parts = line.split(';')
                    else:
                        # Разделяем по пробелам (несколько подряд)
                        parts = line.split()
                    
                    # Конвертируем части в числа
                    row = []
                    for part in parts:
                        try:
                            cleaned = clean_number_string(part)
                            value = float(cleaned)
                            row.append(value)
                        except:
                            row.append(np.nan)
                    
                    # Если в строке есть хотя бы 4 числа, добавляем её
                    if len([x for x in row if not np.isnan(x)]) >= 4:
                        # Заполняем недостающие значения NaN
                        while len(row) < 4:
                            row.append(np.nan)
                        parsed_data.append(row[:4])  # Берем только первые 4 колонки
                
                data = pd.DataFrame(parsed_data)
                print(f"Ручное чтение: строк - {data.shape[0]}, колонок - {data.shape[1]}")
            
            # Проверяем, что данные прочитаны
            if data.shape[0] == 0:
                raise ValueError("Не удалось прочитать данные из файла")
            
            print(f"\nДанные успешно загружены:")
            print(f"Количество строк: {data.shape[0]}")
            print(f"Количество колонок: {data.shape[1]}")
            
            # ВЫБОР ПРАВИЛЬНЫХ КОЛОНОК
            # Пробуем определить, какие колонки содержат нужные данные
            
            # Определяем возможные колонки
            potential_force_cols = []
            potential_disp_cols = []
            potential_time_cols = []
            
            for col_idx in range(min(data.shape[1], 7)):  # Проверяем первые 7 колонок
                col_data = data.iloc[:, col_idx].values
                
                # Убираем NaN
                valid_indices = ~np.isnan(col_data)
                if np.sum(valid_indices) < 100:  # Нужно достаточно данных
                    continue
                    
                clean_data = col_data[valid_indices]
                
                # Анализируем данные в колонке
                
                # 1. Проверяем на колонку ВРЕМЕНИ (должна монотонно увеличиваться)
                if len(clean_data) > 100:
                    # Выбираем подвыборку для анализа
                    sample_size = min(1000, len(clean_data))
                    sample_indices = np.linspace(0, len(clean_data)-1, sample_size, dtype=int)
                    sample_data = clean_data[sample_indices]
                    
                    diffs = np.diff(sample_data)
                    non_decreasing = np.sum(diffs >= -0.0001) / len(diffs)  # Допускаем небольшие колебания
                    
                    if non_decreasing > 0.95 and np.max(sample_data) > 1.0:
                        potential_time_cols.append((col_idx, non_decreasing, np.max(sample_data)))
                
                # 2. Проверяем на колонку СИЛЫ (должна иметь циклы, пики и спады)
                if len(clean_data) > 200:
                    # Вычисляем статистики
                    data_range = np.max(clean_data) - np.min(clean_data)
                    data_mean = np.mean(clean_data)
                    data_std = np.std(clean_data)
                    
                    # Сила должна иметь значительные изменения
                    if data_range > 5 and data_mean > 1 and data_std > 0.5:
                        # Проверяем наличие циклов (производная меняет знак)
                        sample_size = min(500, len(clean_data))
                        sample_indices = np.linspace(0, len(clean_data)-1, sample_size, dtype=int)
                        sample_data = clean_data[sample_indices]
                        
                        diffs = np.diff(sample_data)
                        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                        
                        if sign_changes > 5:  # Должны быть изменения направления
                            potential_force_cols.append((col_idx, data_range, sign_changes))
                
                # 3. Проверяем на колонку ПЕРЕМЕЩЕНИЯ (обычно увеличивается, значения в мм)
                if len(clean_data) > 100:
                    data_range = np.max(clean_data) - np.min(clean_data)
                    data_max = np.max(clean_data)
                    
                    # Перемещение обычно в диапазоне 0-50 мм
                    if 0.1 < data_max < 100 and data_range > 0.5:
                        # Проверяем общий тренд увеличения
                        sample_size = min(500, len(clean_data))
                        sample_indices = np.linspace(0, len(clean_data)-1, sample_size, dtype=int)
                        sample_data = clean_data[sample_indices]
                        
                        # Подгоняем линейную модель
                        x = np.arange(len(sample_data))
                        slope = np.polyfit(x, sample_data, 1)[0]
                        
                        if slope > 0:  # Общий тренд увеличения
                            potential_disp_cols.append((col_idx, slope, data_max))
            
            # Выбираем лучшие колонки
            def select_best(col_list, sort_idx=1, reverse=True):
                if not col_list:
                    return None
                col_list.sort(key=lambda x: x[sort_idx], reverse=reverse)
                return col_list[0][0]
            
            # Выбор колонок
            force_col = select_best(potential_force_cols, sort_idx=1)  # Сортировка по data_range
            disp_col = select_best(potential_disp_cols, sort_idx=1)    # Сортировка по slope
            time_col = select_best(potential_time_cols, sort_idx=1)    # Сортировка по non_decreasing
            
            # Если не удалось определить, используем эвристику
            if force_col is None or disp_col is None or time_col is None:
                print("\nНе удалось автоматически определить колонки, используем эвристику...")
                
                # Из вашего примера данных видно:
                # Колонка 0: Сила (20.543, 22.370 и т.д.)
                # Колонка 2: Перемещение (0.418, 0.438 и т.д.)
                # Колонка 3: Время (1.270, 1.330 и т.д.)
                
                # Проверяем гипотезу, что это стандартный формат
                test_force = data.iloc[:, 0].values
                test_disp = data.iloc[:, 2].values if data.shape[1] > 2 else None
                test_time = data.iloc[:, 3].values if data.shape[1] > 3 else None
                
                # Проверяем диапазоны
                if len(test_force) > 100:
                    force_range = np.max(test_force) - np.min(test_force)
                    if force_range > 5:
                        force_col = 0
                        print(f"Выбрана колонка 0 для силы (диапазон: {force_range:.2f})")
                
                if test_disp is not None and len(test_disp) > 100:
                    disp_range = np.max(test_disp) - np.min(test_disp)
                    if 0.1 < disp_range < 50:
                        disp_col = 2
                        print(f"Выбрана колонка 2 для перемещения (диапазон: {disp_range:.2f})")
                
                if test_time is not None and len(test_time) > 100:
                    time_range = np.max(test_time) - np.min(test_time)
                    if time_range > 1:
                        time_col = 3
                        print(f"Выбрана колонка 3 для времени (диапазон: {time_range:.2f})")
            
            # Если всё ещё не определены, используем первые доступные колонки
            if force_col is None:
                force_col = 0
                print("Используем колонку 0 для силы (по умолчанию)")
            if disp_col is None:
                disp_col = min(1, data.shape[1]-1) if data.shape[1] > 1 else 0
                print(f"Используем колонку {disp_col} для перемещения (по умолчанию)")
            if time_col is None:
                time_col = min(2, data.shape[1]-1) if data.shape[1] > 2 else min(1, data.shape[1]-1)
                print(f"Используем колонку {time_col} для времени (по умолчанию)")
            
            print(f"\nВыбранные колонки:")
            print(f"  Сила: колонка {force_col}")
            print(f"  Перемещение: колонка {disp_col}")
            print(f"  Время: колонка {time_col}")
            
            # ИЗВЛЕЧЕНИЕ ДАННЫХ
            force_raw = data.iloc[:, force_col].values
            displacement_raw = data.iloc[:, disp_col].values
            time_raw = data.iloc[:, time_col].values
            
            # Убираем NaN значения
            valid_mask = ~(np.isnan(force_raw) | np.isnan(displacement_raw) | np.isnan(time_raw))
            force_raw = force_raw[valid_mask]
            displacement_raw = displacement_raw[valid_mask]
            time_raw = time_raw[valid_mask]
            
            print(f"\nДанные после очистки от NaN:")
            print(f"  Сила: {len(force_raw)} точек, диапазон: {np.min(force_raw):.3f} - {np.max(force_raw):.3f} Н")
            print(f"  Перемещение: {len(displacement_raw)} точек, диапазон: {np.min(displacement_raw):.3f} - {np.max(displacement_raw):.3f} мм")
            print(f"  Время: {len(time_raw)} точек, диапазон: {np.min(time_raw):.3f} - {np.max(time_raw):.3f} с")
            
            if len(force_raw) < 100:
                raise ValueError(f"Слишком мало данных после очистки: {len(force_raw)} точек")
            
            # ОБРАБОТКА ДАННЫХ (как в MATLAB скрипте)
            
            # 1. Удаление начальных нулей (find(force_raw~=0, 1, 'first'))
            non_zero_indices = np.where(force_raw != 0)[0]
            if len(non_zero_indices) == 0:
                # Если все нули, ищем первую ненулевую точку по другому критерию
                non_zero_indices = np.where(force_raw > 0.1)[0]
                if len(non_zero_indices) == 0:
                    non_zero_indices = np.where(displacement_raw > 0.1)[0]
            
            if len(non_zero_indices) == 0:
                raise ValueError("Не найдены ненулевые значения")
            
            k = non_zero_indices[0]
            print(f"\nПервый ненулевой индекс: {k} (значение силы: {force_raw[k]:.3f} Н)")
            
            # Нормализуем время, чтобы начиналось с 0
            time = time_raw[k:] - time_raw[k]
            force = force_raw[k:]
            displacement = displacement_raw[k:]
            
            print(f"После обрезки начальных нулей:")
            print(f"  Время: {len(time)} точек")
            print(f"  Сила: {len(force)} точек")
            print(f"  Перемещение: {len(displacement)} точек")
            
            # 2. Фильтрация данных: force > 0 и displacement >= 0 (как в MATLAB)
            valid_mask = (force > 0) & (displacement >= 0)
            time = time[valid_mask]
            force = force[valid_mask]
            displacement = displacement[valid_mask]
            
            print(f"После фильтрации (force > 0 и displacement >= 0):")
            print(f"  Время: {len(time)} точек")
            print(f"  Сила: {len(force)} точек")
            print(f"  Перемещение: {len(displacement)} точек")
            
            if len(time) < 50:
                raise ValueError(f"Слишком мало данных после фильтрации: {len(time)} точек")
            
            # 3. Определение циклов сжатия (dforce = diff(force))
            dforce = np.diff(force)
            
            if len(dforce) < 10:
                raise ValueError("Недостаточно данных для расчета производной")
            
            # Порог для определения начала и конца циклов (0.1 * max(dforce))
            max_dforce = np.max(np.abs(dforce))
            if max_dforce == 0:
                raise ValueError("Производная силы равна нулю")
            
            threshold = 0.1 * max_dforce
            print(f"\nОпределение циклов:")
            print(f"  Максимальная производная: {max_dforce:.3f}")
            print(f"  Порог: {threshold:.3f}")
            
            # Начала циклов: dforce > порога
            compression_starts = np.where(dforce > threshold)[0]
            # Окончания циклов: dforce < -порога
            compression_ends = np.where(dforce < -threshold)[0]
            
            print(f"  Найдено начал циклов: {len(compression_starts)}")
            print(f"  Найдено окончаний циклов: {len(compression_ends)}")
            
            if len(compression_starts) == 0 or len(compression_ends) == 0:
                # Пробуем уменьшить порог
                threshold = 0.05 * max_dforce
                compression_starts = np.where(dforce > threshold)[0]
                compression_ends = np.where(dforce < -threshold)[0]
                print(f"  Новый порог: {threshold:.3f}")
                print(f"  Найдено начал циклов: {len(compression_starts)}")
                print(f"  Найдено окончаний циклов: {len(compression_ends)}")
            
            if len(compression_starts) == 0 or len(compression_ends) == 0:
                # Если всё ещё не нашли, используем альтернативный метод
                print("  Используем альтернативный метод определения циклов...")
                
                # Ищем локальные максимумы и минимумы
                from scipy.signal import find_peaks
                
                # Ищем пики силы (максимумы)
                peaks_max, _ = find_peaks(force, height=np.mean(force), distance=50)
                # Ищем впадины силы (минимумы)
                peaks_min, _ = find_peaks(-force, height=-np.mean(force), distance=50)
                
                if len(peaks_max) > 0 and len(peaks_min) > 0:
                    # Создаем циклы от минимума к максимуму
                    compression_cycles = []
                    for i in range(min(len(peaks_min), len(peaks_max))):
                        start_idx = peaks_min[i]
                        end_idx = peaks_max[i] if peaks_max[i] > start_idx else peaks_max[i+1] if i+1 < len(peaks_max) else len(force)-1
                        if end_idx > start_idx and (end_idx - start_idx) >= 10:
                            compression_cycles.append(np.arange(start_idx, end_idx + 1))
                    
                    compression_starts = [cycle[0] for cycle in compression_cycles]
                    compression_ends = [cycle[-1] for cycle in compression_cycles]
                    
                    print(f"  Найдено циклов (альтернативный метод): {len(compression_cycles)}")
                else:
                    raise ValueError("Не удалось определить циклы сжатия")
            else:
                # Убедимся, что у нас одинаковое количество начал и окончаний
                min_length = min(len(compression_starts), len(compression_ends))
                compression_starts = compression_starts[:min_length]
                compression_ends = compression_ends[:min_length]
                
                # Создаем массив индексов для каждого цикла
                compression_cycles = []
                for start_idx, end_idx in zip(compression_starts, compression_ends):
                    if end_idx > start_idx and (end_idx - start_idx) >= 10:  # Минимум 10 точек
                        cycle_indices = np.arange(start_idx, end_idx + 1)
                        compression_cycles.append(cycle_indices)
            
            if not compression_cycles:
                raise ValueError("Не найдено полных циклов (минимум 10 точек)")
            
            print(f"  Всего циклов для анализа: {len(compression_cycles)}")
            
            # 4. Анализ каждого цикла
            cycle_results = []
            
            for i, idx in enumerate(compression_cycles):
                try:
                    if len(idx) < 10:
                        continue  # Пропускаем слишком короткие циклы
                    
                    cycle_force = force[idx]
                    cycle_disp = displacement[idx]
                    
                    max_force_cycle = np.max(cycle_force)
                    min_force_cycle = np.min(cycle_force)
                    max_idx = np.argmax(cycle_force)
                    min_idx = np.argmin(cycle_force)
                    
                    max_disp_cycle = cycle_disp[max_idx]
                    min_disp_cycle = cycle_disp[min_idx]
                    
                    # Расчет жесткости (как в MATLAB)
                    if (max_disp_cycle - min_disp_cycle) > 0.001:  # Избегаем деления на ноль
                        stiffness = (max_force_cycle - min_force_cycle) / (max_disp_cycle - min_disp_cycle)
                    else:
                        stiffness = np.nan
                        print(f"  Цикл {i+1}: нулевое перемещение, жесткость = NaN")
                    
                    cycle_results.append({
                        'CycleNumber': i + 1,
                        'MaxForce': float(max_force_cycle),
                        'MinForce': float(min_force_cycle),
                        'MaxDisp': float(max_disp_cycle),
                        'MinDisp': float(min_disp_cycle),
                        'Stiffness': float(stiffness) if not np.isnan(stiffness) else None
                    })
                    
                    print(f"  Цикл {i+1}: Fmax={max_force_cycle:.2f} Н, Fmin={min_force_cycle:.2f} Н, "
                        f"Dmax={max_disp_cycle:.3f} мм, Dmin={min_disp_cycle:.3f} мм, "
                        f"Жесткость={stiffness:.2f} Н/мм" if not np.isnan(stiffness) else "Жесткость=NaN")
                        
                except Exception as e:
                    print(f"  Ошибка при анализе цикла {i+1}: {e}")
                    continue
            
            # Удаляем циклы с NaN жесткостью (как в MATLAB: valid_cycles = ~isnan([cycle_results.Stiffness]))
            cycle_results = [cr for cr in cycle_results if cr['Stiffness'] is not None]
            
            if not cycle_results:
                raise ValueError("Не удалось рассчитать жесткость ни для одного цикла")
            
            print(f"\nУспешно проанализировано циклов: {len(cycle_results)}")
            
            # ГРАФИК 1: Сила-время-перемещение (как в MATLAB)
            plt.figure(figsize=(15, 7))
            
            # Левая ось Y - перемещение (синий)
            ax1 = plt.gca()
            ax1.plot(time, displacement, '-b', linewidth=2)
            ax1.set_xlabel('Время, с', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Перемещение, мм', fontsize=12, fontweight='bold', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.grid(True, alpha=0.3)
            
            # Правая ось Y - сила (красный)
            ax2 = ax1.twinx()
            ax2.plot(time, force, '-r', linewidth=2, alpha=0.7)
            ax2.set_ylabel('Нагрузка, Н', fontsize=12, fontweight='bold', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            plt.title(f'Образец: {sample_name} - Сила, время, перемещение', fontsize=14, fontweight='bold')
            
            # Добавляем легенду
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='b', lw=2, label='Перемещение'),
                Line2D([0], [0], color='r', lw=2, label='Сила', alpha=0.7)
            ]
            ax1.legend(handles=legend_elements, loc='upper left')
            
            # Сохранение графика
            time_plot_png = os.path.join(output_dir, f'{sample_name}_time_plot.png')
            plt.savefig(time_plot_png, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Сохранен график 1: {time_plot_png}")
            
            # ГРАФИК 2: Сила-перемещение для циклов (как в MATLAB)
            plt.figure(figsize=(15, 7))
            
            # Создаем цветовую карту для циклов (jet как в MATLAB)
            if len(compression_cycles) > 0:
                colors = plt.cm.jet(np.linspace(0, 1, len(compression_cycles)))
            else:
                colors = ['b']  # Запасной цвет
            
            # Рисуем все циклы
            legend_handles = []
            legend_labels = []
            
            for i, idx in enumerate(compression_cycles):
                if i < len(colors):
                    cycle_disp = displacement[idx]
                    cycle_force = force[idx]
                    color = colors[i]
                    line, = plt.plot(cycle_disp, cycle_force, color=color, linewidth=2, alpha=0.7)
                    
                    # Добавляем в легенду только первые 5 циклов
                    if i < 5:
                        legend_handles.append(line)
                        legend_labels.append(f'Цикл {i+1}')
            
            # Выделение целевого цикла (4-го или последнего, как в MATLAB)
            if len(cycle_results) >= 4:
                target_cycle_idx = 3  # 4-й цикл (индекс 3)
                target_cycle_label = "Цикл 4 (целевой)"
            else:
                target_cycle_idx = len(cycle_results) - 1  # Последний цикл
                target_cycle_label = f"Цикл {target_cycle_idx + 1} (целевой)"
            
            if len(compression_cycles) > target_cycle_idx:
                target_disp = displacement[compression_cycles[target_cycle_idx]]
                target_force = force[compression_cycles[target_cycle_idx]]
                target_line, = plt.plot(target_disp, target_force, 'k-', linewidth=3)
                legend_handles.append(target_line)
                legend_labels.append(target_cycle_label)
                
                # Отметки точек максимума и минимума для целевого цикла
                target_result = cycle_results[target_cycle_idx]
                max_point = plt.scatter(target_result['MaxDisp'], target_result['MaxForce'], 
                        s=100, c='red', marker='o', zorder=5, edgecolors='black', linewidth=1)
                min_point = plt.scatter(target_result['MinDisp'], target_result['MinForce'], 
                        s=100, c='green', marker='o', zorder=5, edgecolors='black', linewidth=1)
                
                legend_handles.extend([max_point, min_point])
                legend_labels.extend(['Макс. сила', 'Мин. сила'])
                
                # Добавляем аннотации
                plt.annotate(f'Fmax = {target_result["MaxForce"]:.1f} Н\nDmax = {target_result["MaxDisp"]:.2f} мм',
                            xy=(target_result['MaxDisp'], target_result['MaxForce']),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                            fontsize=9)
                
                plt.annotate(f'Fmin = {target_result["MinForce"]:.1f} Н\nDmin = {target_result["MinDisp"]:.2f} мм',
                            xy=(target_result['MinDisp'], target_result['MinForce']),
                            xytext=(10, -30), textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                            fontsize=9)
            
            plt.xlabel('Перемещение, мм', fontsize=12, fontweight='bold')
            plt.ylabel('Нагрузка, Н', fontsize=12, fontweight='bold')
            plt.title(f'Образец: {sample_name} - Циклы нагружения', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Добавляем легенду
            if legend_handles:
                plt.legend(handles=legend_handles, labels=legend_labels, loc='best', fontsize=10)
            
            # Сохранение второго графика
            cycles_plot_png = os.path.join(output_dir, f'{sample_name}_cycles_plot.png')
            plt.savefig(cycles_plot_png, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Сохранен график 2: {cycles_plot_png}")
            
            # СОХРАНЕНИЕ В EXCEL (как в MATLAB)
            excel_path = os.path.join(output_dir, f'{sample_name}_results.xlsx')
            
            # Создаем DataFrame с результатами (как results_table в MATLAB)
            results_df = pd.DataFrame(cycle_results)
            results_df.columns = ['Cycle', 'MaxForce_N', 'MinForce_N', 
                                'MaxDisplacement_mm', 'MinDisplacement_mm', 'Stiffness_N_mm']
            
            # Создаем Excel writer
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Лист с результатами циклов
                results_df.to_excel(writer, sheet_name='Результаты циклов', index=False)
                
                # Лист с информацией о целевом цикле
                if target_cycle_idx < len(cycle_results):
                    target_result = cycle_results[target_cycle_idx]
                    target_info = {
                        'Параметр': [
                            'Название образца',
                            'Номер целевого цикла',
                            'Максимальная сила, Н',
                            'Минимальная сила, Н',
                            'Максимальное перемещение, мм',
                            'Минимальное перемещение, мм',
                            'Жесткость, Н/мм'
                        ],
                        'Значение': [
                            sample_name,
                            target_result['CycleNumber'],
                            f"{target_result['MaxForce']:.3f}",
                            f"{target_result['MinForce']:.3f}",
                            f"{target_result['MaxDisp']:.3f}",
                            f"{target_result['MinDisp']:.3f}",
                            f"{target_result['Stiffness']:.3f}"
                        ]
                    }
                    target_df = pd.DataFrame(target_info)
                    target_df.to_excel(writer, sheet_name='Целевой цикл', index=False)
                    
                    # Вывод в консоль (как fprintf в MATLAB)
                    print(f"\n{'='*60}")
                    print(f"РЕЗУЛЬТАТЫ ДЛЯ {target_cycle_idx + 1}-ГО ЦИКЛА:")
                    print(f"{'='*60}")
                    print(f"Максимальная сила: {target_result['MaxForce']:.2f} Н")
                    print(f"Минимальная сила: {target_result['MinForce']:.2f} Н")
                    print(f"Максимальное перемещение: {target_result['MaxDisp']:.3f} мм")
                    print(f"Минимальное перемещение: {target_result['MinDisp']:.3f} мм")
                    print(f"Жесткость: {target_result['Stiffness']:.2f} Н/мм")
                    print(f"{'='*60}")
                
                # Лист с общей информацией
                info_data = {
                    'Информация о файле': [
                        'Имя файла',
                        'Название образца',
                        'Дата обработки',
                        'Всего точек данных',
                        'Всего циклов',
                        'Успешно обработано циклов'
                    ],
                    'Значение': [
                        os.path.basename(data_file),
                        sample_name,
                        pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        len(time),
                        len(compression_cycles),
                        len(cycle_results)
                    ]
                }
                info_df = pd.DataFrame(info_data)
                info_df.to_excel(writer, sheet_name='Информация', index=False)
            
            print(f"Сохранен Excel файл: {excel_path}")
            
            # Пути к созданным файлам
            plot_paths = {
                'time_plot': time_plot_png,
                'cycles_plot': cycles_plot_png
            }
            
            return True, excel_path, plot_paths, target_result if 'target_result' in locals() else None
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"ОШИБКА ПРИ АНАЛИЗЕ {sample_name}:")
            print(f"{'='*60}")
            print(f"Тип ошибки: {type(e).__name__}")
            print(f"Сообщение: {str(e)}")
            print(f"{'='*60}")
            import traceback
            traceback.print_exc()
            return False, None, None, None
    
    def post(self, request):
        data_archive = request.FILES.get('data_archive')
        
        if not data_archive:
            messages.error(request, 'Пожалуйста, загрузите архив с данными')
            return render(request, self.template_name)
        
        try:
            # Создаем временную директорию
            temp_dir = tempfile.mkdtemp()
            print(f"Создана временная директория: {temp_dir}")
            
            # 1. Обработка архива с данными
            data_files = []
            archive_path = os.path.join(temp_dir, 'uploaded_archive.zip')
            
            with open(archive_path, 'wb') as f:
                for chunk in data_archive.chunks():
                    f.write(chunk)
            
            try:
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    # Сначала получаем список всех файлов
                    file_list = zip_ref.namelist()
                    print(f"Файлов в архиве: {len(file_list)}")
                    
                    # Фильтруем только .txt файлы
                    txt_files = [f for f in file_list if f.lower().endswith('.txt')]
                    print(f"Текстовых файлов (.txt): {len(txt_files)}")
                    
                    if not txt_files:
                        raise ValueError("В архиве не найдены файлы данных (.txt)")
                    
                    # Извлекаем все файлы
                    zip_ref.extractall(temp_dir)
                    
                    # Получаем полные пути к извлеченным .txt файлам
                    data_files = [os.path.join(temp_dir, f) for f in txt_files]
                    
                    print(f"Извлеченные файлы: {data_files}")
                    
            except Exception as e:
                raise ValueError(f"Ошибка при распаковке архива: {str(e)}")
            finally:
                if os.path.exists(archive_path):
                    os.unlink(archive_path)
            
            # 2. Создаем директорию для результатов
            output_dir = os.path.join(temp_dir, 'results')
            os.makedirs(output_dir, exist_ok=True)
            print(f"Создана директория для результатов: {output_dir}")
            
            # 3. Обработка каждого файла данных
            results_info = {}
            processed_count = 0
            summary_data = []  # Для сводной таблицы
            
            for data_file in data_files:
                try:
                    # Извлекаем имя образца из имени файла (без расширения)
                    sample_name = os.path.splitext(os.path.basename(data_file))[0]
                    print(f"\n{'='*50}")
                    print(f"Обработка образца: {sample_name}")
                    print(f"Файл: {data_file}")
                    print(f"{'='*50}")
                    
                    # Создаем поддиректорию для образца
                    sample_dir = os.path.join(output_dir, sample_name)
                    os.makedirs(sample_dir, exist_ok=True)
                    print(f"Создана директория для образца: {sample_dir}")
                    
                    # Выполняем анализ (полный аналог MATLAB скрипта)
                    success, excel_path, plot_paths, target_result = self.analyze_vibration_data(
                        data_file, sample_name, sample_dir
                    )
                    
                    if success:
                        results_info[sample_name] = {
                            'excel': excel_path,
                            'plots': plot_paths,
                            'directory': sample_dir
                        }
                        processed_count += 1
                        
                        # Добавляем данные для сводной таблицы
                        if target_result:
                            summary_data.append({
                                'Образец': sample_name,
                                'Цикл': target_result['CycleNumber'],
                                'Макс.сила_Н': target_result['MaxForce'],
                                'Мин.сила_Н': target_result['MinForce'],
                                'Макс.перемещ_мм': target_result['MaxDisp'],
                                'Мин.перемещ_мм': target_result['MinDisp'],
                                'Жесткость_Н_мм': target_result['Stiffness']
                            })
                        
                        print(f"✓ Образец {sample_name} успешно обработан")
                    else:
                        print(f"✗ Ошибка при обработке образца {sample_name}")
                        
                except Exception as e:
                    print(f"✗ Критическая ошибка при обработке файла {data_file}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            
            print(f"\n{'='*50}")
            print(f"Обработка завершена. Успешно обработано: {processed_count} из {len(data_files)} файлов")
            print(f"{'='*50}")
            
            # 4. Создаем сводный Excel файл
            if summary_data:
                summary_path = os.path.join(output_dir, 'СВОДНЫЙ_РЕЗУЛЬТАТЫ.xlsx')
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(summary_path, index=False)
                print(f"Создан сводный файл: {summary_path}")
            
            # 5. Создаем финальный архив
            archive_name = 'vibration_analysis_results.zip'
            final_archive_path = os.path.join(temp_dir, archive_name)
            
            with zipfile.ZipFile(final_archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Добавляем сводный файл (если есть)
                if os.path.exists(summary_path):
                    zipf.write(summary_path, arcname='СВОДНЫЙ_РЕЗУЛЬТАТЫ.xlsx')
                
                # Добавляем все папки с результатами
                for sample_name, info in results_info.items():
                    sample_dir = info['directory']
                    if os.path.exists(sample_dir):
                        for root, dirs, files in os.walk(sample_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                # Сохраняем структуру: results/Образец/файлы
                                arcname = os.path.relpath(file_path, temp_dir)
                                zipf.write(file_path, arcname=arcname)
                                print(f"Добавлен в архив: {arcname}")
            
            print(f"Создан финальный архив: {final_archive_path}")
            print(f"Размер архива: {os.path.getsize(final_archive_path) / 1024 / 1024:.2f} МБ")
            
            # 6. Читаем архив в память
            with open(final_archive_path, 'rb') as f:
                archive_content = f.read()
            
            # 7. Создаем ответ
            response = HttpResponse(
                archive_content,
                content_type='application/zip'
            )
            response['Content-Disposition'] = f'attachment; filename="{archive_name}"'
            
            # 8. Удаляем временные файлы после отправки ответа
            def cleanup_temp_files():
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        print(f"Очищена временная директория: {temp_dir}")
                except Exception as e:
                    print(f"Ошибка при очистке временных файлов: {e}")
            
            # Для Django 3.0+
            response._resource_closers.append(cleanup_temp_files)
            
            return response
            
        except Exception as e:
            messages.error(request, f'Ошибка при обработке файлов: {str(e)}')
            print(f"Ошибка в основном потоке: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Очищаем временную директорию при ошибке
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            return render(request, self.template_name)    
        


class TensileTestUploadView(View):
    template_name = 'protocol/tensile_test.html'
    
    def get(self, request):
        return render(request, self.template_name)
    
    def analyze_tensile_test_data(self, data_file, sample_name, output_dir):
        """Функция анализа (приведена выше)"""
        # Вставьте здесь полную функцию analyze_tensile_test_data
    
    def post(self, request):
        data_archive = request.FILES.get('data_archive')
        
        if not data_archive:
            messages.error(request, 'Пожалуйста, загрузите архив с данными')
            return render(request, self.template_name)
        
        try:
            # Создаем временную директорию
            temp_dir = tempfile.mkdtemp()
            print(f"Создана временная директория: {temp_dir}")
            
            # 1. Обработка архива с данными
            data_files = []
            archive_path = os.path.join(temp_dir, 'uploaded_archive.zip')
            
            with open(archive_path, 'wb') as f:
                for chunk in data_archive.chunks():
                    f.write(chunk)
            
            try:
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    print(f"Файлов в архиве: {len(file_list)}")
                    
                    txt_files = [f for f in file_list if f.lower().endswith('.txt')]
                    print(f"Текстовых файлов (.txt): {len(txt_files)}")
                    
                    if not txt_files:
                        raise ValueError("В архиве не найдены файлы данных (.txt)")
                    
                    zip_ref.extractall(temp_dir)
                    data_files = [os.path.join(temp_dir, f) for f in txt_files]
                    
                    print(f"Извлеченные файлы: {data_files}")
                    
            except Exception as e:
                raise ValueError(f"Ошибка при распаковке архива: {str(e)}")
            finally:
                if os.path.exists(archive_path):
                    os.unlink(archive_path)
            
            # 2. Создаем директорию для результатов
            output_dir = os.path.join(temp_dir, 'results')
            os.makedirs(output_dir, exist_ok=True)
            print(f"Создана директория для результатов: {output_dir}")
            
            # 3. Обработка каждого файла данных
            results_info = {}
            processed_count = 0
            summary_data = []
            
            for data_file in data_files:
                try:
                    sample_name = os.path.splitext(os.path.basename(data_file))[0]
                    print(f"\n{'='*50}")
                    print(f"Обработка образца: {sample_name}")
                    print(f"{'='*50}")
                    
                    sample_dir = os.path.join(output_dir, sample_name)
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # Анализ данных
                    success, excel_path, plot_paths, result_data = self.analyze_tensile_test_data(
                        data_file, sample_name, sample_dir
                    )
                    
                    if success:
                        results_info[sample_name] = {
                            'excel': excel_path,
                            'plots': plot_paths,
                            'directory': sample_dir
                        }
                        processed_count += 1
                        
                        # Добавляем в сводную таблицу
                        summary_data.append({
                            'Образец': sample_name,
                            'Макс_сила_Н': result_data['max_force'],
                            'Перемещение_мм': result_data['disp_max'],
                            'Время_с': result_data['time_max'],
                            'Энергия_Дж': result_data['energy'],
                            'Макс_перемещ_мм': result_data['max_displacement']
                        })
                        
                        print(f"✓ Образец {sample_name} успешно обработан")
                    else:
                        print(f"✗ Ошибка при обработке образца {sample_name}")
                        
                except Exception as e:
                    print(f"✗ Критическая ошибка: {str(e)}")
                    continue
            
            if processed_count == 0:
                raise ValueError("Не удалось обработать ни одного файла данных")
            
            print(f"\n{'='*50}")
            print(f"Обработка завершена. Успешно: {processed_count}/{len(data_files)}")
            print(f"{'='*50}")
            
            # 4. Создаем сводный Excel файл
            if summary_data:
                summary_path = os.path.join(output_dir, 'СВОДНЫЙ_РЕЗУЛЬТАТЫ.xlsx')
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(summary_path, index=False)
                print(f"Создан сводный файл: {summary_path}")
            
            # 5. Создаем финальный архив
            archive_name = 'tensile_test_results.zip'
            final_archive_path = os.path.join(temp_dir, archive_name)
            
            with zipfile.ZipFile(final_archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if os.path.exists(summary_path):
                    zipf.write(summary_path, arcname='СВОДНЫЙ_РЕЗУЛЬТАТЫ.xlsx')
                
                for sample_name, info in results_info.items():
                    sample_dir = info['directory']
                    if os.path.exists(sample_dir):
                        for root, dirs, files in os.walk(sample_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, temp_dir)
                                zipf.write(file_path, arcname=arcname)
            
            print(f"Создан финальный архив: {final_archive_path}")
            
            # 6. Читаем архив и создаем ответ
            with open(final_archive_path, 'rb') as f:
                archive_content = f.read()
            
            response = HttpResponse(
                archive_content,
                content_type='application/zip'
            )
            response['Content-Disposition'] = f'attachment; filename="{archive_name}"'
            
            # Очистка временных файлов
            def cleanup_temp_files():
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        print(f"Очищена временная директория: {temp_dir}")
                except Exception as e:
                    print(f"Ошибка при очистке: {e}")
            
            response._resource_closers.append(cleanup_temp_files)
            
            return response
            
        except Exception as e:
            messages.error(request, f'Ошибка при обработке файлов: {str(e)}')
            print(f"Ошибка: {str(e)}")
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            return render(request, self.template_name)




    def analyze_tensile_test_data(self, data_file, sample_name, output_dir):
        """Анализ данных испытаний на разрыв пластика (трёхточечный изгиб)"""
        
        try:
            print(f"\n{'='*60}")
            print(f"Анализ испытаний на разрыв: {sample_name}")
            print(f"Файл: {data_file}")
            print(f"Выходная директория: {output_dir}")
            print(f"{'='*60}")
            
            # ЧТЕНИЕ ДАННЫХ ИЗ ФАЙЛА
            def clean_number_string(num_str):
                """Очистка строки с числом"""
                if not isinstance(num_str, str):
                    return str(num_str)
                
                cleaned = num_str.strip()
                
                # Если строка содержит табуляцию, берем первую часть
                if '\t' in cleaned:
                    cleaned = cleaned.split('\t')[0]
                
                # Заменяем запятую на точку для десятичных чисел
                cleaned = cleaned.replace(',', '.')
                
                # Убираем лишние нули в начале
                if cleaned.startswith('0') and len(cleaned) > 1 and cleaned[1] != '.':
                    for i, char in enumerate(cleaned):
                        if char != '0' and char != '.':
                            cleaned = cleaned[i:]
                            break
                    if cleaned.startswith('.'):
                        cleaned = '0' + cleaned
                
                if cleaned == '':
                    cleaned = '0'
                    
                return cleaned
            
            # Чтение файла с учетом различных форматов
            try:
                # Сначала пробуем определить разделитель
                with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
                    first_lines = [f.readline() for _ in range(5)]
                
                # Определяем разделитель
                sep = None
                for line in first_lines:
                    if '\t' in line:
                        sep = '\t'
                        break
                    elif ',' in line:
                        sep = ','
                        break
                
                if sep is None:
                    sep = None  # Автоопределение
                
                # Читаем файл
                data = pd.read_csv(
                    data_file,
                    sep=sep,
                    decimal='.',
                    header=None,
                    engine='python',
                    encoding='utf-8',
                    on_bad_lines='skip',
                    dtype=str,
                    skip_blank_lines=True
                )
                
                # Конвертируем строки в числа
                numeric_data = []
                for i in range(data.shape[0]):
                    row = []
                    for j in range(min(data.shape[1], 7)):  # Берем максимум 7 колонок
                        try:
                            value_str = str(data.iloc[i, j])
                            cleaned = clean_number_string(value_str)
                            value = float(cleaned)
                            row.append(value)
                        except:
                            row.append(np.nan)
                    numeric_data.append(row)
                
                data = pd.DataFrame(numeric_data)
                print(f"Успешно прочитано: {data.shape[0]} строк, {data.shape[1]} колонок")
                
            except Exception as e:
                print(f"Ошибка при чтении pandas: {e}")
                # Ручное чтение файла
                with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                parsed_data = []
                for line_num, line in enumerate(lines):
                    if line_num > 50000:  # Ограничение
                        break
                        
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Разбиваем строку
                    parts = []
                    if '\t' in line:
                        parts = line.split('\t')
                    elif ',' in line:
                        parts = line.split(',')
                    elif ';' in line:
                        parts = line.split(';')
                    else:
                        parts = line.split()
                    
                    # Конвертируем в числа
                    row = []
                    for part in parts:
                        try:
                            cleaned = clean_number_string(part)
                            value = float(cleaned)
                            row.append(value)
                        except:
                            row.append(np.nan)
                    
                    # Нужно минимум 4 значения
                    valid_values = [x for x in row if not np.isnan(x)]
                    if len(valid_values) >= 4:
                        while len(row) < 4:
                            row.append(np.nan)
                        parsed_data.append(row[:4])
                
                data = pd.DataFrame(parsed_data)
                print(f"Ручное чтение: {data.shape[0]} строк, {data.shape[1]} колонок")
            
            # Проверка данных
            if data.shape[0] < 10:
                raise ValueError(f"Слишком мало данных: {data.shape[0]} строк")
            
            if data.shape[1] < 4:
                print(f"Предупреждение: только {data.shape[1]} колонок, ожидается 4")
                # Добавляем недостающие колонки с NaN
                while data.shape[1] < 4:
                    data[len(data.columns)] = np.nan
            
            # ИЗВЛЕЧЕНИЕ ДАННЫХ (как в MATLAB скрипте)
            # 1-й столбец — сила, Н
            # 3-й столбец — перемещение, мм
            # 4-й столбец — время, с
            
            force_raw = data.iloc[:, 0].values
            displacement_raw = data.iloc[:, 2].values
            time_raw = data.iloc[:, 3].values
            
            # Убираем NaN
            valid_mask = ~(np.isnan(force_raw) | np.isnan(displacement_raw) | np.isnan(time_raw))
            force_raw = force_raw[valid_mask]
            displacement_raw = displacement_raw[valid_mask]
            time_raw = time_raw[valid_mask]
            
            print(f"\nДанные после очистки:")
            print(f"  Сила: {len(force_raw)} точек, диапазон: {np.min(force_raw):.3f} - {np.max(force_raw):.3f} Н")
            print(f"  Перемещение: {len(displacement_raw)} точек, диапазон: {np.min(displacement_raw):.3f} - {np.max(displacement_raw):.3f} мм")
            print(f"  Время: {len(time_raw)} точек, диапазон: {np.min(time_raw):.3f} - {np.max(time_raw):.3f} с")
            
            if len(force_raw) < 10:
                raise ValueError("Слишком мало данных после очистки")
            
            # ОБРАБОТКА ДАННЫХ (как в MATLAB скрипте)
            
            # Удаление начальных нулей: k = find(force_raw~=0, 1, 'first')
            non_zero_indices = np.where(force_raw != 0)[0]
            if len(non_zero_indices) == 0:
                non_zero_indices = np.where(force_raw > 0.1)[0]
                if len(non_zero_indices) == 0:
                    non_zero_indices = np.where(displacement_raw > 0.1)[0]
            
            if len(non_zero_indices) == 0:
                raise ValueError("Не найдены ненулевые значения")
            
            k = non_zero_indices[0]
            print(f"Первый ненулевой индекс: {k}")
            
            # Извлечение данных с позиции k
            time = time_raw[k:] - time_raw[k]  # Нормализуем время
            force = force_raw[k:]
            displacement = displacement_raw[k:]
            
            # Фильтрация: valid = (force > 0) & (displacement >= 0)
            valid = (force > 0) & (displacement >= 0)
            time = time[valid]
            force = force[valid]
            displacement = displacement[valid]
            
            print(f"После фильтрации (force > 0 и displacement >= 0):")
            print(f"  Время: {len(time)} точек")
            print(f"  Сила: {len(force)} точек")
            print(f"  Перемещение: {len(displacement)} точек")
            
            if len(time) < 10:
                raise ValueError("Слишком мало данных после фильтрации")
            
            # Нахождение максимальной силы: [max_force, idx_max] = max(force)
            max_force = np.max(force)
            idx_max = np.argmax(force)
            time_max = time[idx_max]
            disp_max = displacement[idx_max]
            
            print(f"\nРезультаты анализа:")
            print(f"  Максимальная сила: {max_force:.2f} Н")
            print(f"  Время при максимальной силе: {time_max:.2f} с")
            print(f"  Перемещение при максимальной силе: {disp_max:.3f} мм")
            
            # ГРАФИК 1: Сила-время-перемещение (как в MATLAB)
            fig1 = plt.figure(figsize=(15, 7), facecolor='white')
            
            # Левая ось Y - перемещение (синий)
            ax1 = plt.gca()
            p1, = ax1.plot(time, displacement, '-b', linewidth=2, label='Перемещение')
            h_disp = ax1.scatter(time_max, disp_max, s=200, c='black', marker='o', 
                            zorder=5, label='Макс. перемещение')
            
            # Аннотация для максимального перемещения
            ax1.annotate(f'{disp_max:.2f} мм',
                        xy=(time_max, disp_max),
                        xytext=(10, -10),
                        textcoords='offset points',
                        fontsize=14,
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax1.set_xlabel('Время, с', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Перемещение, мм', fontsize=14, fontweight='bold', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.grid(True, alpha=0.3)
            
            # Правая ось Y - сила (красный)
            ax2 = ax1.twinx()
            p2, = ax2.plot(time, force, '-r', linewidth=2, alpha=0.8, label='Нагрузка')
            h_force = ax2.scatter(time_max, max_force, s=200, c='black', marker='o',
                                zorder=5, label='Макс. нагрузка')
            
            # Аннотация для максимальной силы
            ax2.annotate(f'{max_force:.2f} Н',
                        xy=(time_max, max_force),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=14,
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax2.set_ylabel('Нагрузка, Н', fontsize=14, fontweight='bold', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Объединенная легенда
            lines = [p1, h_disp, p2, h_force]
            labels = ['Перемещение', 'Макс. перемещение', 'Нагрузка', 'Макс. нагрузка']
            ax1.legend(lines, labels, loc='upper left', fontsize=12)
            
            plt.title(f'Образец: {sample_name} - Трёхточечный изгиб', fontsize=16, fontweight='bold')
            
            # Сохранение графика
            time_plot_png = os.path.join(output_dir, f'{sample_name}_время.png')
            plt.savefig(time_plot_png, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Сохранен график 1: {time_plot_png}")
            
            # ГРАФИК 2: Сила-перемещение (как в MATLAB)
            fig2 = plt.figure(figsize=(15, 7), facecolor='white')
            
            plt.plot(displacement, force, '-r', linewidth=2, label='Нагрузка vs Перемещение')
            plt.scatter(disp_max, max_force, s=200, c='black', marker='o', 
                    zorder=5, label='Максимальная нагрузка')
            
            # Аннотация
            plt.annotate(f'({max_force:.2f} Н, {disp_max:.2f} мм)',
                        xy=(disp_max, max_force),
                        xytext=(10, -30),
                        textcoords='offset points',
                        fontsize=14,
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.xlabel('Перемещение, мм', fontsize=14, fontweight='bold')
            plt.ylabel('Нагрузка, Н', fontsize=14, fontweight='bold')
            plt.title(f'Образец: {sample_name} - Диаграмма "Нагрузка-Перемещение"', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=12)
            
            # Сохранение графика
            force_disp_plot_png = os.path.join(output_dir, f'{sample_name}_нагрузка_перемещение.png')
            plt.savefig(force_disp_plot_png, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Сохранен график 2: {force_disp_plot_png}")
            
            # СОХРАНЕНИЕ В EXCEL
            excel_path = os.path.join(output_dir, f'{sample_name}_результаты.xlsx')
            
            # Основные результаты
            results_data = {
                'Параметр': [
                    'Название образца',
                    'Максимальная сила разрушения, Н',
                    'Перемещение при максимальной силе, мм',
                    'Время достижения максимальной силы, с',
                    'Максимальное перемещение, мм',
                    'Максимальная сила (относительная), %',
                    'Жёсткость на участке упругости, Н/мм',
                    'Энергия разрушения, Дж'
                ],
                'Значение': [
                    sample_name,
                    f"{max_force:.3f}",
                    f"{disp_max:.3f}",
                    f"{time_max:.3f}",
                    f"{np.max(displacement):.3f}",
                    f"{100:.1f}",  # Базовое значение
                    f"{(max_force/disp_max if disp_max > 0 else 0):.2f}",
                    f"{np.trapz(force, displacement):.3f}"  # Интеграл силы по перемещению
                ]
            }
            
            # Детальные данные для графика
            detailed_data = {
                'Время_с': time[:1000],  # Ограничиваем количество точек
                'Сила_Н': force[:1000],
                'Перемещение_мм': displacement[:1000]
            }
            
            # Создаем Excel writer
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Лист с основными результатами
                results_df = pd.DataFrame(results_data)
                results_df.to_excel(writer, sheet_name='Основные результаты', index=False)
                
                # Лист с детальными данными
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_excel(writer, sheet_name='Данные испытания', index=False)
                
                # Лист со статистикой
                stats_data = {
                    'Статистика': [
                        'Количество точек данных',
                        'Средняя сила, Н',
                        'Стандартное отклонение силы, Н',
                        'Минимальная сила, Н',
                        'Максимальная сила, Н',
                        'Среднее перемещение, мм',
                        'Максимальное перемещение, мм'
                    ],
                    'Значение': [
                        len(force),
                        f"{np.mean(force):.3f}",
                        f"{np.std(force):.3f}",
                        f"{np.min(force):.3f}",
                        f"{np.max(force):.3f}",
                        f"{np.mean(displacement):.3f}",
                        f"{np.max(displacement):.3f}"
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Статистика', index=False)
            
            print(f"Сохранен Excel файл: {excel_path}")
            
            # Вывод результатов в консоль
            print(f"\n{'='*60}")
            print("РЕЗУЛЬТАТЫ ИСПЫТАНИЙ НА РАЗРЫВ:")
            print(f"{'='*60}")
            print(f"Образец: {sample_name}")
            print(f"Максимальная сила разрушения: {max_force:.2f} Н")
            print(f"Перемещение в момент разрушения: {disp_max:.3f} мм")
            print(f"Время до разрушения: {time_max:.2f} с")
            print(f"{'='*60}")
            
            # Пути к созданным файлам
            plot_paths = {
                'time_plot': time_plot_png,
                'force_disp_plot': force_disp_plot_png
            }
            
            result_data = {
                'sample_name': sample_name,
                'max_force': max_force,
                'disp_max': disp_max,
                'time_max': time_max,
                'max_displacement': np.max(displacement),
                'energy': np.trapz(force, displacement)
            }
            
            return True, excel_path, plot_paths, result_data
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"ОШИБКА ПРИ АНАЛИЗЕ {sample_name}:")
            print(f"{'='*60}")
            print(f"Тип ошибки: {type(e).__name__}")
            print(f"Сообщение: {str(e)}")
            print(f"{'='*60}")
            import traceback
            traceback.print_exc()
            return False, None, None, None