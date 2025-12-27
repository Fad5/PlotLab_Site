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