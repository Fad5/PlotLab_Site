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

def read_ecofizika(file, axes = ['2','1']):
    """Reads data from Ecofizika (Octava)"""
    vibration = pd.read_csv(file, sep='\t', encoding='mbcs', header=None, names=axes,
                            dtype=np.float32,
                            skiprows=4, usecols=range(1,len(axes)+1)).reset_index(drop=True)
    inf = pd.read_csv(file, sep=' ', encoding='mbcs', header=None, names = None,
                           skiprows=2, nrows=1).reset_index(drop=True)
    fs = int(inf.iloc[0, -1])

    return vibration, fs









def vibration_analysis_(request):
    error = None
    data = {
        'tests': []  # Initialize as empty list
    }
    context_ = {'datasets': []}
    context_['peaks'] = []
    context_['results_table'] = []
    

    if request.method == 'POST':
            # Получаем параметры из формы
            name = request.POST.get('sampleName', '')
            a = float(request.POST.get('width', 100))
            b = float(request.POST.get('length', 100))
            h = float(request.POST.get('height_id', 20))
            hz = float(request.POST.get('Hz', 700)) 
            left_lim = float(request.POST.get('left_lim', 5))
            limits = (0, int(hz))   
            i = 0
            while True:
                print(i)
                height_key = f'height_{str(i)}'
                mass_key = f'mass_{str(i)}'
                file_key = f'file_{str(i)}' 
                # Проверяем, есть ли такой блок в запросе
                if height_key not in request.POST:
                    break   
                # Получаем данные блока
                height = request.POST.get(height_key)
                mass = request.POST.get(mass_key)
                file = request.FILES.get(file_key)  
                if height and mass and file:
                    # Обработка файла (сохранение)
                    file_path = os.path.join(settings.MEDIA_ROOT, file.name)
                    with open(file_path, 'wb+') as destination:
                        for chunk in file.chunks():
                            destination.write(chunk)    
                    # Добавляем данные теста
                    data['tests'].append({
                        'height': float(height),
                        'mass': float(mass),
                        'file_path': file_path,
                        'file_name': file.name
                    })  
                    i += 1      
            context_['form_data'] = {
                'sampleName': name,
                'width': a,
                'length': b,
                'height_id': h,
                'Hz': hz,
                'left_lim': left_lim
             }

            for i in data['tests']:
                print(i)
            S = a * b * 1e-6  # площадь образца (m2)
            results = {}        
            # Основные графики (аналоги fig_main)
            main_plots = {
                'transfer_function': {'x': [], 'y': [], 'names': [], 'mode': 'lines'},
                'isolation_efficiency': {'x': [], 'y': [], 'names': [], 'mode': 'lines'}
            }       
            i = 0
            all_data = []
            for M in data['tests']:
                print(M)
                project_name = f"{name}_{M['mass']}кг"
                _h = M['height'] * 1e-3  # высота образца (м)      
                fs = FileSystemStorage()  
                vibration_list, fs = read_ecofizika(M['file_path'], ['2', '1'])  
                # Вычисляем спектрограмму
                Pxx = {}
                freqs_ = {}
                try:
                    for ax in ['1', '2']:
                        y = vibration_list[ax].values       
                        if len(y) < 2048:
                            error = f"Недостаточно данных для спектра (ось {ax}, длина: {len(y)})"
                            raise ValueError(error)     
                        freqs_[ax], _, Pxx[ax] = spectrogram(
                            y,
                            nperseg=2048,
                            noverlap=256,
                            fs=fs,
                            scaling='spectrum',
                            mode='magnitude'
                        )       
                    if len(freqs_['1']) < 2:
                        error = "Недостаточно частотных данных после спектра (ось 1)"
                        raise ValueError(error)     
                    if freqs_['1'][1] == 0:
                        error = "Вторая частота равна нулю (ошибка шкалы частот)"
                        raise ValueError(error)     
                    # Ограничение по частотам
                    last_index = min(int(limits[1] / freqs_['1'][1]), len(freqs_['1']) - 1)
                    freqs = freqs_['1'][1:last_index]
                    left_lim_idx = np.argmax(freqs > left_lim) if len(freqs) > 0 else 0     
                    # Вычисления TR, L и их сглаженных версий
                    TR1 = np.mean(Pxx['2'][1:last_index] / Pxx['1'][1:last_index], axis=1)
                    TR = np.mean(Pxx['1'][1:last_index] / Pxx['2'][1:last_index], axis=1)
                    TR1mean = pd.Series(TR1).rolling(10, min_periods=1, center=True).mean()
                    TRmean = pd.Series(TR).rolling(10, min_periods=1, center=True).mean()
                    L = 20 * np.log10(TR)
                    Lmean = 20 * np.log10(TRmean)           
                    plots_1 = [
                        {
                            'x': freqs.tolist(),
                            'y': TR1.tolist(),
                            'name': f'Передаточная функция {M['mass']}',
                            'mode': 'lines',
                            'xaxis_title': 'Частота, Гц',
                            'yaxis_title': 'Модуль передаточной функции',
                        },
                        {
                            'x': freqs.tolist(),
                            'y': TR1mean.tolist(),
                            'mode': 'lines'
                        }
                    ]   
                    plots_2 = [
                        {
                            'x': freqs.tolist(),
                            'y': L.tolist(),
                            'name': 'L',
                            'mode': 'lines'
                        },
                        {
                            'x': freqs.tolist(),
                            'y': Lmean.tolist(),
                            'name': 'L сглаженная',
                            'mode': 'lines'
                        }
                    ]
                    pp = [plots_1, plots_2]
                    all_data.append(pp)   

                except Exception as e:
                    print("Ошибка при спектральном анализе:", str(e))
                    error = f"Ошибка при анализе спектра (нагрузка {M['mass']} кг): {str(e)}"       
                context_['datasets'] = all_data
                # Анализ пиков
                if len(TR1mean) > left_lim_idx:
                    max1 = TR1mean[left_lim_idx:].max()
                    f_peaks = find_peaks(TR1mean[left_lim_idx:], distance=100, prominence=0.1*max1) 
                    if len(f_peaks[0]) > 0:
                        f_peak_pos = f_peaks[0][0] + left_lim_idx
                        Fpeak = freqs[f_peak_pos]   
                        # Добавляем информацию о пике       
                        peak_info = {
                            'frequency': Fpeak,
                            'position': [Fpeak, TR1mean[f_peak_pos]],
                            'efficiency_position': [Fpeak, Lmean[f_peak_pos]],
                            'marker': {'size': 10, 'color': 'red'}
                        }       
                        print(f"Columns: {vibration_list.columns}")
                        print(f"Len ax1: {len(vibration_list['1'])}, Len ax2: {len(vibration_list['2'])}")      
                        f1, f2 = find_res_width2(TR1mean, freqs, f_peak_pos)
                        if f1 >= 0:
                            damp = (f2 - f1) / Fpeak
                            Ed = 4 * np.pi**2 * Fpeak**2 * M['mass'] * _h / S * 1e-6
                            results[i] = (Fpeak, Ed, damp)      
                            peak_info.update({
                                'resonance_width': [f1, f2],
                                'annotation': {
                                    'text': f"Динамический модуль: {Ed:.5f} МПа<br>Демпфирование: {damp:.5f}<br>Частота: {Fpeak:.2f} Гц",
                                    'position': [Fpeak, TR1mean[f_peak_pos]]
                                }
                            }) 

                        peak_info_table = {
                            'name': M['file_name'],
                            'Fpeak': Fpeak,
                            'Ed': Ed,
                            'damp': damp,
                            'DM':TR1mean[f_peak_pos]
                        }       
                        context_['results_table'].append(peak_info_table)
                        context_['peaks'].append(peak_info)

    print(f"Error in vibration_analysis: {error}")
    context_['peaks_json'] = json.dumps(context_.get('peaks', []))
    return render(request, 'analysis/vibration_analysis.html', context_)

executor = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4))

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
                        'Усилие, кН': 'float32',
                        'Перемещение, мм': 'float32'
                    }  # Уменьшаем объём памяти
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

    # 4. Ускоренное сохранение (если обязательно нужно)
    if hasattr(settings, 'MEDIA_ROOT'):
        path_save_full = os.path.join(settings.MEDIA_ROOT, 'serva', 'full', 'full.csv')
        full_df.to_csv(path_save_full, index=False, mode='w', encoding='utf-8')  # mode='w' перезаписывает файл

    # 5. Логирование времени
    print(f"Обработано {len(time_)} точек за {time.time() - start_time:.2f} сек")

    # 6. Возвращаем результат
    return json.dumps([
        {'x': time_, 'y': displacement, 'title': "Перемещение (мм) - Время (с)"},
        {'x': time_, 'y': force, 'title': "Нагрузка (кН) - Время (с)"},
        {'x': displacement, 'y': force, 'title': "Нагрузка (кН) - Перемещение (мм)"},
    ])


async def Servo(request):
    context = {'datasets': []}
    if request.method == 'POST':
        uploaded_files = request.FILES.getlist('csv_file')

        # Запускаем процесс обработки в отдельном потоке, чтобы не блокировать event loop
        plots_json = await asyncio.get_event_loop().run_in_executor(
            executor, process_files, uploaded_files
        )

        context['datasets'] = plots_json
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





def tensile_test_view(request):
    context = {}
    
    if request.method == 'POST' and request.FILES.get('csv_file'):
        try:
            # Получение параметров из формы
            uploaded_file = request.FILES['csv_file']
            L0 = float(request.POST.get('height', 80.0))  # Начальная длина образца (мм)
            A0 = float(request.POST.get('area', 40.0))    # Начальная площадь сечения (мм²)
            
            # Сохранение файла временно
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = fs.path(filename)
            
            # Загрузка и обработка данных согласно ГОСТ 11262-2017
            data = pd.read_csv(file_path, sep="\t", header=None, decimal=',')
            force = data[0].values.astype(float)
            displacement = data[2].values.astype(float)
            
            # Расчет напряжений и деформаций
            stress = force / A0  # МПа
            strain = displacement / L0  # относительная деформация
            
            # Удаление начального участка (до 0.1% деформации)
            threshold = 0.001 * L0
            valid_idx = np.where(displacement > threshold)[0]
            stress = stress[valid_idx]
            strain = strain[valid_idx]
            displacement = displacement[valid_idx]
            
            # Сглаживание данных
            if len(stress) > 50:
                stress_smooth = savgol_filter(stress, 51, 3)
            else:
                stress_smooth = stress
            
            # Преобразование numpy массивов в списки
            strain_list = strain.tolist()
            stress_smooth_list = stress_smooth.tolist()
            
            # Определение ключевых точек согласно ГОСТ 11262-2017
            # 1. Предел прочности (R_m)
            sigma_max = float(np.max(stress_smooth))
            idx_max = int(np.argmax(stress_smooth))
            
            # 2. Предел текучести (R_p0.2)
            epsilon_target = 0.002
            interp_func = interp1d(strain, stress_smooth, fill_value="extrapolate")
            sigma_p02 = float(interp_func(epsilon_target))
            
            # 3. Модуль упругости (E) на участке 0.05% - 0.25% деформации
            epsilon_1 = 0.0005
            epsilon_2 = 0.0025
            sigma_1 = float(interp_func(epsilon_1))
            sigma_2 = float(interp_func(epsilon_2))
            E = float((sigma_2 - sigma_1) / (epsilon_2 - epsilon_1) / 1000)  # ГПа
            
            # 4. Относительное удлинение при разрыве
            epsilon_break = float(strain[-1])
            
            # Подготовка данных для графика
            datasets = [{
                'x': strain_list,
                'y': stress_smooth_list,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Диаграмма растяжения',
                'line': {'width': 2}
            }, {
                'x': [epsilon_target],
                'y': [sigma_p02],
                'mode': 'markers',
                'name': f'R<sub>p0.2</sub> = {sigma_p02:.1f} МПа',
                'marker': {'color': 'green', 'size': 10}
            }, {
                'x': [strain_list[idx_max]],
                'y': [sigma_max],
                'mode': 'markers',
                'name': f'R<sub>m</sub> = {sigma_max:.1f} МПа',
                'marker': {'color': 'red', 'size': 10}
            }]
            
            shapes = [
                {'type': 'line', 'x0': epsilon_1, 'y0': 0, 'x1': epsilon_1, 'y1': sigma_1,
                 'line': {'color': 'grey', 'dash': 'dash'}, 'opacity': 0.5},
                {'type': 'line', 'x0': epsilon_2, 'y0': 0, 'x1': epsilon_2, 'y1': sigma_2,
                 'line': {'color': 'grey', 'dash': 'dash'}, 'opacity': 0.5},
                {'type': 'line', 'x0': epsilon_target, 'y0': 0, 'x1': epsilon_target, 'y1': sigma_p02,
                 'line': {'color': 'green', 'dash': 'dash'}, 'opacity': 0.5},
                {'type': 'line', 'x0': strain_list[idx_max], 'y0': 0, 'x1': strain_list[idx_max], 'y1': sigma_max,
                 'line': {'color': 'red', 'dash': 'dash'}, 'opacity': 0.5}
            ]
            
            # Сохранение результатов
            context['datasets'] = json.dumps(datasets, ensure_ascii=False)
            context['layout'] = json.dumps({
                'title': 'График растяжения по ГОСТ 11262-2017',
                'xaxis': {'title': 'Относительная деформация ε'},
                'yaxis': {'title': 'Напряжение σ, МПа'},
                'shapes': shapes,
                'legend': {'x': 0.7, 'y': 0.1},
                'hovermode': 'x unified'
            }, ensure_ascii=False)
            
            context['results'] = {
                'sigma_max_MPa': f"{sigma_max:.1f}",
                'sigma_p02_MPa': f"{sigma_p02:.1f}",
                'E_modulus_GPa': f"{E:.2f}",
                'epsilon_break_percent': f"{strain_list[idx_max]*100:.2f}"
            }
            
            # Удаление временного файла
            fs.delete(filename)
            
        except Exception as e:
            context['error'] = f"Произошла ошибка при обработке данных: {str(e)}"
            if 'file_path' in locals() and os.path.exists(file_path):
                fs.delete(filename)
    
    return render(request, 'analysis/razr.html', context)


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

