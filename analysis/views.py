from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import numpy as np
from scipy.signal import find_peaks, spectrogram
from .utils import YoungModulusAnalyzer
import tempfile
from django.shortcuts import render
import json
from plotly.offline import plot
import pandas as pd
import re
from matplotlib.mlab import specgram
from django.template.loader import get_template
import logging
import os 
import pandas as pd
from django.http import HttpResponse
from io import BytesIO
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time
from .handler_db import log_upload_to_db
logger = logging.getLogger(__name__)

def main_(request):
    error = None
    
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        
        try:
            # Пробуем разные разделители
            df = pd.read_csv(csv_file, sep="\t", header=None)
            df.replace(",", ".", regex=True, inplace=True)
            df = df.astype(float)

            force = df[0].values  # нагрузка (Н)
            displacement = df[2].values  # перемещение (мм)
            time_ = df[3].values  # время (с)

            context = {
                'time': json.dumps(time_.tolist()),
                'disp': json.dumps(displacement.tolist()),
                'forse': json.dumps(force.tolist()),
                'error': error
            }   
            print(context)
        except Exception as e:
            error = f"Ошибка при обработке файла: {str(e)}"
            context = {
                'time': [],
                'disp': [],
                'forse': [],
                'error': error
            }
        
        return render(request, 'analysis/on_load.html', context)

    # <- ВАЖНО: сюда попадает любой GET-запрос или если файл не загружен
    context = {
        'time': [],
        'disp': [],
        'forse': [],
        'error': error
    }
    return render(request, 'analysis/on_load.html', context)

def box_san(request):
    error = None
    context = {}
    plots = []
    context = {'datasets': []}

    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        

        print(csv_file)
        context['filename'] = {
            'filename': csv_file
        }
        raw_lines = csv_file.read().decode('utf-8').splitlines()

        data = []
        for line in raw_lines:
            # заменяем двойные точки на одну и разбиваем по пробелам/запятым
            numbers = re.findall(r'-?\d+(?:\.\d+)?', line)
            if len(numbers) >= 6:
                data.append([float(n) for n in numbers[:6]])

        if not data:
            error = "Не удалось прочитать данные из файла."
        else:
            df = pd.DataFrame(data)

            chanal1 = df[0].values
            chanal2 = df[1].values
            chanal3 = df[2].values
            chanal4 = df[3].values
            chanal5 = df[4].values
            chanal6 = df[5].values
            x = len(df[0].values)

            plots.append({
            'x': x,
            'y': chanal1.tolist(),
            'title': "Канал 1",
            })

            plots.append({
            'x': x,
            'y': chanal2.tolist(),
            'title': "Канал 2",
            })

            plots.append({
            'x': x,
            'y': chanal3.tolist(),
            'title': "Канал 3",
            })

            plots.append({
            'x': x,
            'y': chanal4.tolist(),
            'title': "Канал 4",
            })
            
            plots.append({
            'x': x,
            'y': chanal5.tolist(),
            'title': "Канал 5",
            })

            plots.append({
            'x': x,
            'y': chanal6.tolist(),
            'title': "Канал 6",
            })
            

            context['datasets'] = json.dumps(plots)
            

    return render(request, 'analysis/box_san.html', context)


def home(request):
    return render(request, 'analysis/home.html')


def about(request):
    return render(request, 'analysis/about.html')



class YoungModulusAnalyzer:
    def __init__(self, file_path, sample_name="Sample"):
        self.file_path = file_path
        self.df = None
        self.sample_name = sample_name
        self.width = 128.07  # мм
        self.length = 108.01  # мм
        self.initial_height = 28.25  # мм
        self.area = self.width * self.length  # мм²
        self.q = self.area / (2 * self.initial_height * (self.width + self.length))
        self.results_dir = tempfile.mkdtemp()
        self.plots = []
        self.error = None

    def load_data(self):
        try:
            log_upload_to_db(self.file_path, 'young_modul')
            self.df = pd.read_csv(self.file_path, sep="\t", header=None, decimal=",")
            self.df = self.df.replace(",", ".", regex=True).astype(float)
            return True
        except Exception as e:
            self.error = f"Ошибка загрузки данных: {e}"
            return False

    def process_data(self):
        if self.df is None:
            self.error = "Данные не загружены!"
            return False

        try:
            k = np.argmax(self.df[0].values > 0)
            M = self.df.values[k:, :4] - self.df.values[k, :4]
            sr = len(M) / M[-1, 3] if M[-1, 3] != 0 else 10

            F = M[:, 0]
            S = M[:, 2]
            T = np.arange(len(F)) / sr

            peaks, _ = find_peaks(S, height=0.5 * np.max(S))
            if len(peaks) < 3:
                self.error = "Недостаточно пиков для анализа (3 минимум)"
                return False

            Defl = S[peaks[0]]
            V = Defl / T[peaks[0]]

            self.create_loading_plots(T, F, S, peaks, Defl)

            if len(peaks) >= 4:
                cycle_index = 3
            else:
                cycle_index = 2

            cycle_length = peaks[cycle_index] - peaks[cycle_index - 1]
            Start = peaks[cycle_index] - cycle_length + 1
            Finish = peaks[cycle_index]

            F1 = F[Start:Finish + 1]
            S1 = S[Start:Finish + 1]

            w = 2 * int(np.ceil(sr))
            n = len(F1) // w

            Pr = np.zeros(n)
            E1 = np.zeros(n)
            Eps1 = np.zeros(n)

            for i in range(n):
                idx1 = i * w
                idx2 = (i + 1) * w - 1
                if idx2 >= len(F1):
                    idx2 = len(F1) - 1

                Pr[i] = (F1[idx1] + F1[idx2]) / 2 / self.area * 1e-6
                delta_F = F1[idx2] - F1[idx1]
                delta_S = S1[idx2] - S1[idx1]

                if delta_S != 0:
                    E1[i] = (delta_F / self.area * 1e-6) / (delta_S / self.initial_height)

                Eps1[i] = (S1[idx1] + S1[idx2]) / 2 / self.initial_height

            if len(Pr) > 0:
                Pr = Pr - Pr[0]

            self.create_results_plots(Pr, E1, Eps1)
            return True

        except Exception as e:
            self.error = f"Ошибка обработки данных: {e}"
            return False

    def create_loading_plots(self, T, F, S, peaks, Defl):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.plots.append({
            'x': T.tolist(),
            'y': S.tolist(),
            'title': f"{self.sample_name} - Перемещение",
            'xaxis_title': 'Время, с',
            'yaxis_title': 'Перемещение, мм',
            'color':  '#1f77b4'})

        cycle_len = peaks[1] - peaks[0]
        labels = ['Цикл 1 Перемещение - Нагрузка', 'Цикл 2 Перемещение - Нагрузка', 'Цикл 3 Перемещение - Нагрузка', 'Цикл 4 Перемещение - Нагрузка', 'Цикл 5 Перемещение - Нагрузка']
        print(peaks)
        print(len(peaks))
        for i in range(min(5, len(peaks))):
            start = peaks[i] - cycle_len + 1 if i > 0 else 0
            end = peaks[i] + cycle_len if i < len(peaks) - 1 else len(S)

            print(F[start:end])
            print(S[start:end])

            area = np.trapezoid(F[start:end], S[start:end])
            self.plots.append({
                'x': S[start:end].tolist(),
                'y': F[start:end].tolist(),
                'title': f"{self.sample_name} - {labels[i]} - {area}",
                'xaxis_title': 'Перемещение, мм',
                'yaxis_title': 'Нагрузка, N',
                'color':  colors[i % len(colors)]})


    def create_loading_plots(self, T, F, S, peaks, Defl):
        # График перемещения (оставляем как было)
        self.plots.append({
            'x': T.tolist(),
            'y': S.tolist(),
            'title': f"{self.sample_name} - Перемещение",
            'xaxis_title': 'Время, с',
            'yaxis_title': 'Перемещение, мм',
            'line': {'color': '#1f77b4'}
        })

        # Цвета для разных циклов
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        cycle_len = peaks[1] - peaks[0]
        
        # Создаем один объединенный график для всех циклов
        combined_plot = {
            'title': f"{self.sample_name} - Все циклы Перемещение-Нагрузка",
            'xaxis_title': 'Перемещение, мм',
            'yaxis_title': 'Нагрузка, N',
            'data': []
        }
        
        for i in range(min(5, len(peaks))):
            start = peaks[i] - cycle_len + 1 if i > 0 else 0
            end = peaks[i] + cycle_len if i < len(peaks) - 1 else len(S)
            
            # Добавляем данные цикла в объединенный график
            combined_plot['data'].append({
                'x': S[start:end].tolist(),
                'y': F[start:end].tolist(),
                'name': f'Цикл {i+1}',
                'line': {'color': colors[i % len(colors)]}
            })
            
            area = np.trapezoid(F[start:end], S[start:end])
            displacement_closed = None

            load  = F[start:end]
            displacement = S[start:end]
                    # Если не совпадают, добавляем первую точку в конец для замыкания
            if (load[0] != load[-1]) or (displacement[0] != displacement[-1]):
                load_closed = np.append(load, load[0])
                displacement_closed = np.append(displacement, displacement[0])
            else:
                load_closed = load
                displacement_closed = displacement

            # Вычисляем площадь методом трапеций для замкнутого контура
            area = 0.5 * np.abs(np.sum(load_closed[:-1] * np.diff(displacement_closed) - 
                                np.sum(displacement_closed[:-1] * np.diff(load_closed)))) 
            # Также сохраняем отдельные графики для каждого цикла (по желанию)
            self.plots.append({
                'x': displacement_closed.tolist(),
                'y': load_closed.tolist(),
                'title': f"{self.sample_name} - Цикл {i+1} Перемещение-Нагрузка {area}",
                'xaxis_title': 'Перемещение, мм',
                'yaxis_title': 'Нагрузка, N',
                'line': {'color': colors[i % len(colors)]},
                'fill': 'tonexty', 
                'fillcolor': 'rgba(0, 100, 200, 0.2)'
            })
        
        # Добавляем объединенный график в список графиков
        self.plots.append(combined_plot)

    def create_results_plots(self, Pr, E1, Eps1):
         
        E1  = E1[E1 != 0]
        Pr_MPa = (Pr - Pr[0]) * 1e6
        half = len(Pr_MPa) // 2
        E1 = E1 * 1_000_000
        xticks = Pr_MPa[half:]
        xtick_labels = np.abs(xticks)
        x = Pr_MPa[:half]
        y = E1[half:] 
        self.plots.append({
            'x': (abs(x)).tolist(),
            'y': (y).tolist(),
            'title': "Модуль упругости - Удельное давление",
            'xaxis_title': 'Нагрузка, MPa',
            'yaxis_title': "Модуль деформации, MPa",
            'xticks': xticks.tolist(),
            'xtick_labels': xtick_labels.tolist(),
            'color':  '#1f77b4' 
        })
        self.plots.append({
            'x': (abs(x)).tolist(),
            'y': (Eps1[half:]* 100).tolist(),
            'color':  '#1f77b4',
            'title': 'Относительная деформация - Удельное давление',
            'xaxis_title': 'Нагрузка , MПa',
            'yaxis_title': 'Относительная деформация, %',
        })

def analyze_young_modulus(request):
    context = {'datasets': []}
    

    if request.method == 'POST' and request.FILES.get('csv_file'):
        uploaded_file = request.FILES['csv_file']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        # Читаем параметры из формы
        sample_name = request.POST.get('sampleName', 'Sample')
        width = float(request.POST.get('width', 100))
        length = float(request.POST.get('length', 100))
        initial_height = float(request.POST.get('height', 20))

        # Передаём параметры в анализатор
        analyzer = YoungModulusAnalyzer(
            file_path=file_path,
            sample_name=sample_name
        )
        analyzer.width = width
        analyzer.length = length
        analyzer.initial_height = initial_height
        analyzer.area = width * length
        analyzer.q = analyzer.area / (2 * initial_height * (width + length))

        context['form_data'] = {
                    'sampleName': sample_name,
                    'width': width,
                    'length': length,
                    'height': int(initial_height)
                 }

        # Основной анализ
        if analyzer.load_data():
            if analyzer.process_data():
                context['datasets'] = json.dumps(analyzer.plots)
            else:
                context['error'] = analyzer.error
        else:
            context['error'] = analyzer.error

        # Удаляем файл
        if os.path.exists(file_path):
            os.remove(file_path)

        # Возвращаем значения обратно в форму
        context.update({
            'sample_name': sample_name,
            'width': width,
            'length': length,
            'initial_height': initial_height
        })

    return render(request, 'analysis/PPU_Testus.html', context)







import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, spectrogram
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
import logging



def find_res_width2(TR, freqs, peak_pos):
    """Нахождение ширины резонанса на половине высоты"""
    try:
        half_height = TR[peak_pos] / 2**0.5

        # Левая граница
        left = np.where(TR[:peak_pos] <= half_height)[0]
        if len(left) > 0 and (peak_pos - left[-1]) >= 1:
            TR_left = TR[left[-1]:peak_pos+1]
            freqs_left = freqs[left[-1]:peak_pos+1]
            if len(TR_left) >= 2 and len(freqs_left) >= 2:
                f1 = np.interp(half_height, TR_left[::-1], freqs_left[::-1])
            else:
                f1 = freqs[left[-1]]
        else:
            f1 = freqs[0]

        # Правая граница
        right = np.where(TR[peak_pos:] <= half_height)[0]
        if len(right) > 0:
            right_end = peak_pos + right[0] + 1
            TR_right = TR[peak_pos:right_end]
            freqs_right = freqs[peak_pos:right_end]
            if len(TR_right) >= 2 and len(freqs_right) >= 2:
                f2 = np.interp(half_height, TR_right, freqs_right)
            else:
                f2 = freqs[right_end-1]
        else:
            f2 = freqs[-1]

        return f1, f2

    except Exception as e:
        print(f"[find_res_width2] Ошибка: {e}")
        return -1, -1

def read_ecofizika(file, axes):
    """Reads data from Ecofizika (Octava)"""
    vibration = pd.read_csv(file, sep='\t', encoding='mbcs', header=None, names=axes,
                            dtype=np.float32,
                            skiprows=4, usecols=range(1,len(axes)+1)).reset_index(drop=True)
    inf = pd.read_csv(file, sep=' ', encoding='mbcs', header=None, names = None,
                           skiprows=2, nrows=1).reset_index(drop=True)
    fs = int(inf.iloc[0, -1])

    return vibration, fs

def downsample_list(data, step=100):
    return data[::step]

import pandas as pd
import os
import time
import json
from django.conf import settings

def process_files(files):
    start_time = time.time()
    
    # 1. Ускоренное чтение файлов
    df_list = []
    for file in files:
        try:
            if file.name.endswith('.alc'):
                # Оптимизация: используем низкоуровневое чтение + C-движок
                df = pd.read_csv(
                    file,
                    sep=';',
                    decimal=',',
                    encoding='utf-8',
                    engine='c',  # Используем более быстрый C-движок
                    usecols=['Время, с', ' Усилие, кН', ' Перемещение, мм'],  # Читаем только нужные колонки
                    dtype={
                        'Время, с': 'float32',
                        ' Усилие, кН': 'float32',
                        ' Перемещение, мм': 'float32'
                    }
                )
            else:
                df = pd.read_excel(
                    file,
                    usecols=['Время, с', ' Усилие, кН', ' Перемещение, мм'],
                    dtype='float32'
                )
            df_list.append(df)
        except Exception as e:
            print(f"Ошибка при чтении {file.name}: {str(e)}")
            continue

    # 2. Быстрое объединение DataFrame'ов
    if not df_list:
        return json.dumps({"error": "Нет данных для обработки"})
    
    full_df = pd.concat(df_list, ignore_index=True, copy=False)  # copy=False для экономии памяти

    # 3. Оптимизированное извлечение данных
    time_ = full_df['Время, с'].values.tolist()  # .values быстрее, чем прямой tolist()
    force = np.abs(full_df[' Усилие, кН'].values).tolist()  # numpy.abs быстрее pandas.abs
    displacement = np.abs(full_df[' Перемещение, мм'].values).tolist()

    # 4. Создание папки и сохранение файла
    if hasattr(settings, 'MEDIA_ROOT'):
        # Создаем путь к папке
        folder_path = os.path.join(settings.MEDIA_ROOT, 'serva', 'full')
        
        # Создаем папки (если не существуют)
        os.makedirs(folder_path, exist_ok=True)
        
        # Полный путь к файлу
        file_path = os.path.join(folder_path, 'full.csv')
        
        # Сохраняем файл
        full_df.to_csv(file_path, index=False, mode='w', encoding='utf-8')
        
        # Возвращаем не только данные, но и путь к файлу для последующего удаления
        return json.dumps({
            'data': [
                {'x': time_, 'y': displacement, 'title': "Перемещение (мм) - Время (с)"},
                {'x': time_, 'y': force, 'title': "Нагрузка (кН) - Время (с)"},
                {'x': displacement, 'y': force, 'title': "Нагрузка (кН) - Перемещение (мм)"},
            ],
            'file_path': file_path
        })
    
    # 5. Логирование времени
    print(f"Обработано {len(time_)} точек за {time.time() - start_time:.2f} сек")

    # 6. Возвращаем результат без пути к файлу
    return json.dumps({
        'data': [
            {'x': time_, 'y': displacement, 'title': "Перемещение (мм) - Время (с)"},
            {'x': time_, 'y': force, 'title': "Нагрузка (кН) - Время (с)"},
            {'x': displacement, 'y': force, 'title': "Нагрузка (кН) - Перемещение (мм)"},
        ],
        'file_path': None
    })


def delete_temp_file(file_path):
    """Функция для удаления временного файла"""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"Файл {file_path} успешно удален")
            
            # Пытаемся удалить пустые родительские папки
            folder_path = os.path.dirname(file_path)
            if os.path.exists(folder_path) and not os.listdir(folder_path):
                os.rmdir(folder_path)
                print(f"Папка {folder_path} удалена")
    except Exception as e:
        print(f"Ошибка при удалении файла {file_path}: {str(e)}")


async def Servo(request):
    context = {'datasets': []}
    if request.method == 'POST':
        uploaded_files = request.FILES.getlist('csv_file')

        # Запускаем процесс обработки в отдельном потоке
        result_json = await asyncio.get_event_loop().run_in_executor(
            None,  # Используем дефолтный executor
            process_files, 
            uploaded_files
        )
        
        # Парсим JSON результат
        result = json.loads(result_json)
        
        # Если есть данные для графиков
        if 'data' in result:
            context['datasets'] = json.dumps(result['data'])
            
            # Если файл был сохранен, удаляем его после отправки данных пользователю
            file_path = result.get('file_path')
            if file_path:
                # Удаляем файл через 1 секунду (даем время на отправку ответа)
                loop = asyncio.get_event_loop()
                loop.call_later(1, delete_temp_file, file_path)
        else:
            context['datasets'] = result_json  # Возвращаем ошибку как есть
        
        return render(request, 'analysis/servo.html', context)
    
    return render(request, 'analysis/servo.html')


# 3-х точечный пластмассы 


from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def find_yield_point_max(strain, stress, sigma_max):
    stress_smooth = savgol_filter(stress, 51, 3)
    max_index = np.argmax(stress_smooth)

    diff = np.diff(stress_smooth[:max_index])
    for i in range(10, len(diff)-10):
        if diff[i] < diff[i-1] and np.all(diff[i:i+5] <= 0):
            sigma_y = stress[i]
            eps_y = strain[i]
            print(sigma_y, 'sigma_y')
            print(eps_y, 'eps_y')
            print(i, 'i')
            if (sigma_max - sigma_y) / sigma_max < 0.02:
                break
            
            return sigma_y, eps_y, i

    return None, None, None

def compression(request):
    context = {}
    
    if request.method == 'POST' and request.FILES.get('csv_file'):
            uploaded_file = request.FILES['csv_file']
            A0 = float(request.POST.get('area', 100))
            L0 = float(request.POST.get('height', 100))
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)
            
            # Анализ данных
            data = pd.read_csv(file_path, sep="\t", header=None)
            data = data.replace(",", ".", regex=True).astype(float)

            force = data[0].values
            displacement = data[2].values
            

            # Фильтрация
            mask = (force > 0)
            force = force[mask]
            displacement = displacement[mask]

            # Расчет напряжений и деформаций
            stress = force / A0
            strain = displacement / L0
            eps0 = strain[0]  
            strain = strain - eps0

            # Поиск ключевых точек
            sigma_max = np.max(stress)
            idx_max = np.argmax(stress)
            eps_max = strain[idx_max]

            sigma_fracture = stress[-1]
            eps_fracture = strain[-1]

            print(strain, "strain")
            print( stress, "stress")
            print( sigma_max,  "sigma_max")

            # Поиск точки текучести
            sigma_y, eps_y, index = find_yield_point_max(strain, stress, sigma_max)
            print(sigma_y,'sigma_y')
            print(eps_y, 'eps_y')
            print(index, 'index')
            if not sigma_y:
                target_sigma = sigma_max * 0.94  # Ищем 94% от sig_u
                # Берём данные только до idx_u
                strain_before_u = strain[:idx_max]
                stress_before_u = stress[:idx_max]
                
                # Находим ближайшее значение к target_sigma
                index = np.argmin(np.abs(stress_before_u - target_sigma))
                sigma_y = stress_before_u[index]
                eps_y = strain_before_u[index]
            if index:
                index = int(index /2)

            sigma_x = stress[index]
            eps_x = strain[index]
            E_yield_GPa = stress[index] / strain[index] / 1000

            if sigma_y is None or (sigma_max - sigma_y) / sigma_max < 0.02:
                target_sigma = sigma_max * 0.98
                mask_left = strain <= eps_max
                idx = np.argmin(np.abs(stress[mask_left] - target_sigma))
                sigma_y = stress[mask_left][idx]
                eps_y = strain[mask_left][idx]

            # Подготовка данных для графика
            datasets = [{
                'x': strain.tolist(),
                'y': stress.tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': 'σ(ε)',
                'line': {'width': 2}
            }, {
                'x': [eps_y],
                'y': [sigma_y],
                'mode': 'markers',
                'name': f'σ_y={sigma_y:.1f}МПа<br>ε_y={eps_y*100:.2f}%',
                'marker': {'color': 'green', 'size': 10}
            },{
                'x': [eps_x],
                'y': [sigma_x],
                'mode': 'markers',
                'name': f'σ_x={sigma_x:.1f}МПа<br>ε_x={eps_x*100:.2f}%, Точка нахождения модуля юнга',
                'marker': {'color': 'grey', 'size': 10}
            }, {
                'x': [eps_max],
                'y': [sigma_max],
                'mode': 'markers',
                'name': f'σ_max={sigma_max:.1f}МПа<br>ε_max={eps_max*100:.2f}%',
                'marker': {'color': 'red', 'size': 10}
            }, {
                'x': [eps_fracture],
                'y': [sigma_fracture],
                'mode': 'markers',
                'name': f'σ_fracture={sigma_fracture:.1f}МПа<br>ε_fracture={eps_fracture*100:.2f}%',
                'marker': {'color': 'purple', 'size': 10}
            }]

            # Вертикальные линии
            shapes = [
                {'type': 'line', 'x0': eps_x, 'y0': 0, 'x1': eps_x, 'y1': sigma_x,
                 'line': {'color': 'grey', 'dash': 'dash'}, 'opacity': 0.7},
                {'type': 'line', 'x0': eps_y, 'y0': 0, 'x1': eps_y, 'y1': sigma_y,
                 'line': {'color': 'green', 'dash': 'dash'}, 'opacity': 0.7},
                {'type': 'line', 'x0': eps_max, 'y0': 0, 'x1': eps_max, 'y1': sigma_max,
                 'line': {'color': 'red', 'dash': 'dash'}, 'opacity': 0.7},
                {'type': 'line', 'x0': eps_fracture, 'y0': 0, 'x1': eps_fracture, 'y1': sigma_fracture,
                 'line': {'color': 'purple', 'dash': 'dash'}, 'opacity': 0.7}
            ]

            # Сохраняем данные для передачи в шаблон
            context['datasets'] = json.dumps(datasets)
            context['layout'] = json.dumps({
                'title': f"Диаграмма сжатия: {uploaded_file.name}",
                'xaxis': {'title': 'ε (относительная деформация)'},
                'yaxis': {'title': 'σ (МПа)'},
                'shapes': shapes,
                'legend': {'x': 0.7, 'y': 0.1},
                'hovermode': 'x unified'
            })
            
            context['filename'] = {'filename': uploaded_file.name}
            context['results'] = {
                'sigma_y_MPa': f"{sigma_y:.1f}",
                'eps_y_percent': f"{eps_y*100:.2f}",
                'E_yield_GPa': f"{E_yield_GPa:.2f}",
                'sigma_max_MPa': f"{sigma_max:.1f}",
                'eps_max_percent': f"{eps_max*100:.2f}",
                'sigma_fracture_MPa': f"{sigma_fracture:.1f}",
                'eps_fracture_percent': f"{eps_fracture*100:.2f}"
            }
            
            # Удаляем временный файл
            fs.delete(filename)
            
    
    return render(request, 'analysis/3.html', context)



class FlexureAnalyzer:
    def __init__(self, L=64, b=10, h=4):
        self.L = L
        self.b = b
        self.h = h
        self.epsilon_f1 = 0.0005
        self.epsilon_f2 = 0.0025

    def calculate_stress(self, F):
        return (3 * F * self.L) / (2 * self.b * self.h**2)

    def calculate_strain(self, s, percent=False):
        strain = (6 * s * self.h) / (self.L**2)
        return strain * 100 if percent else strain

    def find_deflection_at_strain(self, epsilon_f):
        return (epsilon_f * self.L**2) / (6 * self.h)

    def calculate_elastic_modulus(self, F1, F2):
        sigma_f1 = self.calculate_stress(F1)
        sigma_f2 = self.calculate_stress(F2)
        return (sigma_f2 - sigma_f1) / (self.epsilon_f2 - self.epsilon_f1)

    def smooth_data(self, x, y, window_size=30, polyorder=10):
        """Apply Savitzky-Golay filter for smoothing"""
        from scipy.signal import savgol_filter
        if len(y) > window_size:
            return savgol_filter(y, window_size, polyorder)
        return y

    def process_data(self, force, displacement):
        # Pre-process data to remove dead zone
        threshold = 0.1
        first_idx = np.argmax(force > threshold)
        
        if first_idx > 0:
            d = displacement[first_idx]
            f = force[first_idx]
            d_ = np.linspace(0, d, first_idx)
            f_ = np.linspace(0, f, first_idx)
            force = np.concatenate((f_, force[first_idx:]))
            displacement = np.concatenate((d_, displacement[first_idx:]))
        
        # Apply smoothing
        smoothed_force = self.smooth_data(displacement, force)
        smoothed_displacement = displacement  # Don't smooth x-axis
        
        # Main calculations
        stress = self.calculate_stress(smoothed_force)
        strain = self.calculate_strain(smoothed_displacement)
        
        s1_target = self.find_deflection_at_strain(self.epsilon_f1)
        s2_target = self.find_deflection_at_strain(self.epsilon_f2)
        
        interp_func = interp1d(smoothed_displacement, smoothed_force, fill_value="extrapolate")
        F1 = interp_func(s1_target)
        F2 = interp_func(s2_target)
        
        E_f = self.calculate_elastic_modulus(F1, F2)
        
        return {
            'stress': stress.tolist(),
            'strain': (strain * 1000).tolist(),  # in ‰
            'displacement': smoothed_displacement.tolist(),
            'force': smoothed_force.tolist(),
            'target_points': {
                'epsilon': [self.epsilon_f1 * 1000, self.epsilon_f2 * 1000],
                'deflection': [s1_target, s2_target],
                'force': [F1, F2],
                'stress': [self.calculate_stress(F1), self.calculate_stress(F2)]
            },
            'elastic_modulus': E_f / 1000
        }

def numpy_to_python(value):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    elif isinstance(value, dict):
        return {k: numpy_to_python(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [numpy_to_python(v) for v in value]
    return value

def flex_analysis_view(request):
    context = {}
    
    if request.method == 'POST' and request.FILES.get('data_file'):
        try:
            file = request.FILES['data_file']
            L = float(request.POST.get('L', 64))
            b = float(request.POST.get('b', 10))
            h = float(request.POST.get('h', 4))
            smooth_data = request.POST.get('smooth_data', 'false').lower() == 'true'
            
            df = pd.read_csv(file, sep="\t", header=None, decimal=',')
            if len(df.columns) < 3:
                raise ValueError("File must have at least 3 columns")
                
            analyzer = FlexureAnalyzer(L, b, h)
            results = analyzer.process_data(df[0].values, df[2].values)
            
            # Get both raw and smoothed data if needed
            raw_stress = analyzer.calculate_stress(df[0].values)
            raw_strain = analyzer.calculate_strain(df[2].values) * 1000
            
            # Convert all numpy data to Python native types
            chart_data = numpy_to_python({
                'stress_strain': {
                    'x': results['strain'],
                    'y': results['stress'],
                    'x_raw': raw_strain.tolist(),
                    'y_raw': raw_stress.tolist(),
                    'target_x': results['target_points']['epsilon'],
                    'target_y': results['target_points']['stress']
                },
                'force_displacement': {
                    'x': df[2].values,
                    'y': df[0].values,
                }
            })
            
            context['chart_data'] = json.dumps(chart_data)
            context['results'] = numpy_to_python({
                'elastic_modulus': round(results['elastic_modulus'], 2),
                'target_points': results['target_points']
            })
            
        except Exception as e:
            context['error'] = f"Error processing file: {str(e)}"
    
    return render(request, 'analysis/flex_analysis.html', context)


def on_load(request):
    error = None
    results = []  # Будет хранить результаты для всех файлов
    
    if request.method == 'POST' and request.FILES.getlist('csv_files'):
        csv_files = request.FILES.getlist('csv_files')
        
        for csv_file in csv_files:
            try:
                # Чтение файла
                df = pd.read_csv(csv_file, sep="\t", header=None)
                df.replace(",", ".", regex=True, inplace=True)
                df = df.astype(float)

                force = df[0].values  # нагрузка (Н)
                displacement = df[2].values  # перемещение (мм)

                # Расчет средних значений
                mask_20_25 = (force >= 20) & (force <= 25)
                avg_displacement_20 = np.mean(displacement[mask_20_25]) if np.any(mask_20_25) else None

                mask_45_52 = (force >= 50) & (force <= 52)
                avg_displacement_50 = np.mean(displacement[mask_45_52]) if np.any(mask_45_52) else None

                mask_90_102 = (force >= 100) & (force <= 102)
                avg_displacement_100 = np.mean(displacement[mask_90_102]) if np.any(mask_90_102) else None

                # Добавляем результаты для этого файла
                filename = csv_file.name.split('.')[0]  # Имя файла без расширения
                results.append({
                    'filename': filename,
                    'avg_20': avg_displacement_20,
                    'avg_50': avg_displacement_50,
                    'avg_100': avg_displacement_100,
                })
                
            except Exception as e:
                error = f"Ошибка обработки файла {csv_file.name}: {str(e)}"
                continue
        
        # Если запрос на скачивание Excel
        if 'download_excel' in request.POST:
            return generate_excel_response(results)
    
    context = {
        'error': error,
        'results': results,
    }
    return render(request, 'analysis/on_load.html', context)

def generate_excel_response(results):
    # Создаем DataFrame из результатов
    df = pd.DataFrame(results)
    df.columns = ['Имя файла', '20Н (мм)', '50Н (мм)', '100Н (мм)']
    
    # Создаем Excel файл в памяти
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Results')
    writer.close()
    output.seek(0)
    
    # Создаем HTTP ответ
    response = HttpResponse(
        output.getvalue(),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = 'attachment; filename=results.xlsx'
    return response

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from scipy.signal import find_peaks, spectrogram
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.conf import settings
import json

def save_plot_to_html(fig):
    """Сохраняет график matplotlib в HTML-совместимый формат"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

def read_ecofizika(file, axes):
    """Reads data from Ecofizika (Octava)"""
    vibration = pd.read_csv(file, sep='\t', encoding='mbcs', header=None, names=axes,
                          dtype=np.float32,
                          skiprows=4, usecols=range(1,len(axes)+1)).reset_index(drop=True)
    inf = pd.read_csv(file, sep=' ', encoding='mbcs', header=None, names=None,
                     skiprows=2, nrows=1).reset_index(drop=True)
    fs = int(inf.iloc[0, -1])
    return vibration, fs

def find_res_width2(TR, freqs, peak_pos):
    """Нахождение ширины резонанса на половине высоты"""
    try:
        half_height = TR[peak_pos] / 2**0.5

        # Левая граница
        left = np.where(TR[:peak_pos] <= half_height)[0]
        if len(left) > 0 and (peak_pos - left[-1]) >= 1:
            TR_left = TR[left[-1]:peak_pos+1]
            freqs_left = freqs[left[-1]:peak_pos+1]
            if len(TR_left) >= 2 and len(freqs_left) >= 2:
                f1 = np.interp(half_height, TR_left[::-1], freqs_left[::-1])
            else:
                f1 = freqs[left[-1]]
        else:
            f1 = freqs[0]

        # Правая граница
        right = np.where(TR[peak_pos:] <= half_height)[0]
        if len(right) > 0:
            right_end = peak_pos + right[0] + 1
            TR_right = TR[peak_pos:right_end]
            freqs_right = freqs[peak_pos:right_end]
            if len(TR_right) >= 2 and len(freqs_right) >= 2:
                f2 = np.interp(half_height, TR_right, freqs_right)
            else:
                f2 = freqs[right_end-1]
        else:
            f2 = freqs[-1]

        return f1, f2

    except Exception as e:
        print(f"[find_res_width2] Ошибка: {e}")
        return -1, -1
    

def vibration_analysis___(request):
    plt.rcParams['figure.facecolor'] = '#ffffff0d'  # Темный фон фигуры
    plt.rcParams['axes.facecolor'] = '#ffffff0d'    # Темный фон областей графиков
    error = None
    data = {
        'tests': [],
        'sample_params': {}
    }
    context = {
        'plots': [],
        'peaks': [],
        'results_table': [],
        'sample_names': [],
        'form_data': {
            'width': 100,
            'length': 100,
            'height_id': 20,
            'Hz': 700,
            'left_lim': 5,
            'show_mean_line': True,
            'tests': []
        }
    }
    axec = ['2', '1']

    # Список для хранения путей к временным файлам
    temp_files_to_delete = []

    if request.method == 'POST':
        # Получаем параметры образца и сохраняем их для отображения
        form_data = {
            'width': float(request.POST.get('width', 100)),
            'length': float(request.POST.get('length', 100)),
            'height_id': float(request.POST.get('height_id', 20)),
            'Hz': float(request.POST.get('Hz', 700)),
            'left_lim': float(request.POST.get('left_lim', 5)),
            'show_mean_line': request.POST.get('show_mean_line') == 'true',
            'tests': []
        }
        context['form_data'] = form_data
        show_mean_line = form_data['show_mean_line']
        print( request.POST.get('show_mean_line', 'true'), ' request.POST.get')
        print(show_mean_line, 'show_mean_line')


        # Обрабатываем испытания
        i = 0
        while True:
            height_key = f'height_{i}'
            mass_key = f'mass_{i}'
            file_key = f'file_{i}'
            is_swap = f'swap_{i}'
            sample_name_key = f'sample_name_{i}'
            existing_file_key = f'existing_file_{i}'
            
            if height_key not in request.POST:
                break
                
            height = request.POST.get(height_key)
            mass = request.POST.get(mass_key)
            file = request.FILES.get(file_key)
            existing_file = request.POST.get(existing_file_key)
            swap = request.POST.get(is_swap, 'false')
            sample_name = request.POST.get(sample_name_key, f'Образец_{i+1}')

            if height and mass and (file or existing_file):
                # Сохраняем данные теста для отображения в форме
                test_data = {
                    'loaded_height': float(height),
                    'mass': float(mass),
                    'swap': swap,
                    'sample_name': sample_name,
                    'file_name': existing_file if existing_file else (file.name if file else '')
                }
                form_data['tests'].append(test_data)
                
                # Обработка файла
                if file:
                    file_path = os.path.join(settings.MEDIA_ROOT, file.name)
                    with open(file_path, 'wb+') as destination:
                        for chunk in file.chunks():
                            destination.write(chunk)
                    file_path_to_use = file_path
                    file_name_to_use = file.name
                    # Добавляем в список для удаления, только если файл был загружен
                    temp_files_to_delete.append(file_path)
                elif existing_file:
                    file_path_to_use = os.path.join(settings.MEDIA_ROOT, existing_file)
                    file_name_to_use = existing_file
                    # Существующие файлы не удаляем
                else:
                    continue
                
                data['tests'].append({
                    'loaded_height': float(height),
                    'mass': float(mass),
                    'file_path': file_path_to_use,
                    'file_name': file_name_to_use,
                    'sample_name': sample_name,
                    'is_new_file': bool(file)  # Флаг, что файл был загружен
                })
                context['sample_names'].append(sample_name)
                
            i += 1

        # Анализ данных
        S = form_data['width'] * form_data['length'] * 1e-6
        limits = (0, int(form_data['Hz']))
        left_lim = form_data['left_lim']

        combined_transfer_data = []
        combined_efficiency_data = []

        for idx, test in enumerate(data['tests']):
            try:
                vibration_list, fs = read_ecofizika(test['file_path'], axec)
                
                # Вычисляем спектрограмму
                Pxx = {}
                freqs_ = {}
                for ax in axec:
                    y = vibration_list[ax].values
                    freqs_[ax], _, Pxx[ax] = spectrogram(
                        y, nperseg=2048, noverlap=256, fs=fs,
                        scaling='spectrum', mode='magnitude'
                    )

                last_index = min(int(limits[1] / freqs_['1'][1]), len(freqs_['1']) - 1)
                freqs = freqs_['1'][1:last_index]
                left_lim_idx = np.argmax(freqs > left_lim) if len(freqs) > 0 else 0

                TR1 = np.mean(Pxx['2'][1:last_index] / Pxx['1'][1:last_index], axis=1)
                TR1mean = pd.Series(TR1).rolling(10, min_periods=1, center=True).mean()
                L = 20 * np.log10(np.mean(Pxx['1'][1:last_index] / Pxx['2'][1:last_index], axis=1))
                Lmean = pd.Series(L).rolling(10, min_periods=1, center=True).mean()

                # Графики для текущего образца
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                plt.suptitle(f"{test['sample_name']}", y=1.02)
                
                ax1.plot(freqs, TR1, label='Исходные данные', alpha=0.5)
                ax1.plot(freqs, TR1mean, label='Сглаженные данные', linewidth=2)
                ax1.set(xlabel='Частота, Гц', ylabel='Модуль передаточной функции')
                ax1.grid(True)
                
                ax2.plot(freqs, L, label='Исходные данные', alpha=0.5)
                ax2.plot(freqs, Lmean, label='Сглаженные данные', linewidth=2)
                ax2.set(xlabel='Частота, Гц', ylabel='Эффективность, дБ')
                ax2.grid(True)
                
                plt.tight_layout()
                context['plots'].append({
                    'title': test['sample_name'],
                    'image': save_plot_to_html(fig),
                    'index': idx
                })

                # Анализ пиков
                if len(TR1mean) > left_lim_idx:
                    max1 = TR1mean[left_lim_idx:].max()
                    f_peaks = find_peaks(TR1mean[left_lim_idx:], distance=100, prominence=0.1*max1)
                    
                    if len(f_peaks[0]) > 0:
                        f_peak_pos = f_peaks[0][0] + left_lim_idx
                        Fpeak = freqs[f_peak_pos]
                        
                        f1, f2 = find_res_width2(TR1mean, freqs, f_peak_pos)
                        if f1 >= 0:
                            damp = (f2 - f1) / Fpeak
                            Ed = 4 * np.pi**2 * Fpeak**2 * test['mass'] * (test['loaded_height']*1e-3) / S * 1e-6
                            
                            context['results_table'].append({
                                'name': test['file_name'],
                                'sample_name': test['sample_name'],
                                'Fpeak': Fpeak,
                                'Ed': Ed,
                                'damp': damp,
                                'DM': TR1mean[f_peak_pos]
                            })
                            context['peaks'].append({
                                'sample_index': idx,
                                'frequency': Fpeak,
                                'position': [Fpeak, TR1mean[f_peak_pos]],
                                'resonance_width': [f1, f2]
                            })

                # Данные для общих графиков
                combined_transfer_data.append({
                    'freqs': freqs,
                    'TR1mean': TR1mean,
                    'name': test['sample_name'],
                    'index': idx
                })
                combined_efficiency_data.append({
                    'freqs': freqs,
                    'Lmean': Lmean,
                    'name': test['sample_name'],
                    'index': idx
                })

            except Exception as e:
                error = f"Ошибка при анализе {test['file_name']}: {str(e)}"
                print(error)

        # Общие графики (СРЕДНЯЯ ЛИНИЯ МЕЖДУ ОБРАЗЦАМИ)
        if combined_transfer_data and combined_efficiency_data:
            # 1. Общий график передаточных функций
            fig_combined, ax = plt.subplots(figsize=(12, 6))
            
            # Рисуем все кривые с индивидуальными названиями
            for data in combined_transfer_data:
                ax.plot(data['freqs'], data['TR1mean'], label=data['name'], linewidth=2.5, alpha=0.8)
            
            # Добавляем среднюю линию между образцами
            if show_mean_line and len(combined_transfer_data) > 1:
                # Находим общий диапазон частот
                min_freq = max([data['freqs'].min() for data in combined_transfer_data])
                max_freq = min([data['freqs'].max() for data in combined_transfer_data])
                
                # Создаем общую сетку частот
                common_freqs = np.linspace(min_freq, max_freq, 1000)
                
                # Интерполируем все данные на общую сетку
                interpolated_data = []
                for data in combined_transfer_data:
                    interp_func = interp1d(data['freqs'], data['TR1mean'], bounds_error=False, fill_value="extrapolate")
                    interpolated_data.append(interp_func(common_freqs))
                
                # Вычисляем среднее значение
                mean_transfer = np.nanmean(interpolated_data, axis=0)
                if show_mean_line:
                # Рисуем среднюю линию
                    ax.plot(common_freqs, mean_transfer, 'k--', linewidth=3, 
                        label='Средняя линия', alpha=0.9)
            
            ax.set(
                xlabel='Частота, Гц',
                ylabel='Модуль передаточной функции'
            )
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            context['plots'].append({
                'title': 'Общий график передаточных функций',
                'image': save_plot_to_html(fig_combined),
                'index': 'combined_transfer'
            })

            # 2. Общий график эффективности
            fig_combined_eff, ax = plt.subplots(figsize=(12, 6))
            
            # Рисуем все кривые с индивидуальными названиями
            for data in combined_efficiency_data:
                ax.plot(data['freqs'], data['Lmean'], label=data['name'], linewidth=2.5, alpha=0.8)
            
            # Добавляем среднюю линию между образцами
            if show_mean_line and len(combined_efficiency_data) > 1:
                # Находим общий диапазон частот
                min_freq = max([data['freqs'].min() for data in combined_efficiency_data])
                max_freq = min([data['freqs'].max() for data in combined_efficiency_data])
                
                # Создаем общую сетку частот
                common_freqs = np.linspace(min_freq, max_freq, 1000)
                
                # Интерполируем все данные на общую сетку
                interpolated_data = []
                for data in combined_efficiency_data:
                    interp_func = interp1d(data['freqs'], data['Lmean'], bounds_error=False, fill_value="extrapolate")
                    interpolated_data.append(interp_func(common_freqs))
                
                # Вычисляем среднее значение
                mean_efficiency = np.nanmean(interpolated_data, axis=0)

                if show_mean_line:
                # Рисуем среднюю линию
                    ax.plot(common_freqs, mean_efficiency, 'k--', linewidth=3, 
                        label='Средняя линия', alpha=0.9)
            
            ax.set(
                xlabel='Частота, Гц',
                ylabel='Эффективность, дБ'
            )
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            context['plots'].append({
                'title': 'Общий график эффективности',
                'image': save_plot_to_html(fig_combined_eff),
                'index': 'combined_efficiency'
            })

    # УДАЛЕНИЕ ВРЕМЕННЫХ ФАЙЛОВ после создания всех графиков
            for file_path in temp_files_to_delete:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Удален временный файл: {file_path}")
                except Exception as e:
                    print(f"Ошибка при удалении файла {file_path}: {e}")

    return render(request, 'analysis/vibro.html', context)