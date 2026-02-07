import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, gaussian_filter1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path 
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Cm
import datetime
import os


class YoungModulusAnalyzer:
    def __init__(self, file_path, params, upload_dir):
        self.file_path = file_path
        self.params = params
        self.upload_dir = Path(upload_dir)
        
        # Инициализация данных
        self.raw_data = None
        self.stress_data = None
        self.strain_data = None
        self.time_data = None
        self.young_modulus = None
        self.form_factor = None
    
    def load_and_process_data(self):
        """Загрузка и первичная обработка данных"""
        try:
            # Загрузка данных
            self.raw_data = pd.read_csv(self.file_path, sep="\t", header=None)
            self.raw_data.replace(",", ".", regex=True, inplace=True)
            self.raw_data = self.raw_data.astype(float)
            
            # Получение параметров образца
            width = self.params['width']
            length = self.params['length']
            height = self.params['height']
            area = width * length * 1e-6  # Переводим в м²
            
            # Извлечение столбцов данных
            force = self.raw_data[0].values  # Нагрузка (Н)
            displacement = self.raw_data[2].values  # Перемещение (мм)
            time = self.raw_data[3].values  # Время (с)
            
            # Расчет основных параметров
            with np.errstate(divide='ignore', invalid='ignore'):
                strain = displacement / height  # Относительная деформация
                stress = (force / area) * 1e-6  # Удельное давление (МПа)
            
            # Фильтрация некорректных значений
            valid = (~np.isnan(stress)) & (~np.isnan(strain)) & (stress > 0) & (strain > 0)
            
            self.stress_data = stress[valid]
            self.strain_data = strain[valid] * 100  # Переводим в проценты
            self.time_data = time[valid]
            
            # Расчет модуля Юнга
            young_modulus = np.gradient(self.stress_data, self.strain_data)
            self.young_modulus = self._smooth_data(young_modulus)
            
            # Расчет коэффициента формы
            self.form_factor = width / height
            
            return True
            
        except Exception as e:
            raise ValueError(f"Ошибка обработки данных: {str(e)}")
    
    def _smooth_data(self, data):
        """Сглаживание данных"""
        window_size = min(50, len(data)//4 or 1)
        data_median = median_filter(data, size=window_size)
        return gaussian_filter1d(data_median, sigma=2)
    
    def create_plots(self):
        """Создание графиков с индивидуальными шкалами"""
        # Основные графики
        fig1 = make_subplots(rows=1, cols=2, subplot_titles=(
            "Модуль Юнга vs Напряжение", 
            "Деформация vs Напряжение"
        ))
        
        fig1.add_trace(
            go.Scatter(
                x=self.stress_data,
                y=self.young_modulus,
                mode='lines',
                line=dict(color='#2e7d32', width=3),
                name='Модуль Юнга (МПа)'
            ),
            row=1, col=1
        )
        
        fig1.add_trace(
            go.Scatter(
                x=self.stress_data,
                y=self.strain_data,
                mode='lines',
                line=dict(color='#2e7d32', width=3),
                name='Деформация (%)'
            ),
            row=1, col=2
        )
        
        # Деформация во времени
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=self.time_data,
                y=self.strain_data,
                mode='lines',
                line=dict(color='#2e7d32', width=3),
                name='Деформация (%)'
            )
        )
        
        # Полные зависимости
        fig3 = make_subplots(rows=1, cols=2, subplot_titles=(
            "Модуль Юнга vs Время", 
            "Деформация vs Напряжение"
        ))
        
        fig3.add_trace(
            go.Scatter(
                x=self.time_data,
                y=self.young_modulus,
                mode='lines',
                line=dict(color='#2e7d32', width=3),
                name='Модуль Юнга (МПа)'
            ),
            row=1, col=1
        )
        
        fig3.add_trace(
            go.Scatter(
                x=self.stress_data,
                y=self.strain_data,
                mode='lines',
                line=dict(color='#2e7d32', width=3),
                name='Деформация (%)'
            ),
            row=1, col=2
        )
        
        # Общие настройки для всех графиков
        for fig in [fig1, fig2, fig3]:
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#2e7d32'),
                hovermode='closest',
                margin=dict(t=40, l=50, r=30, b=50)
            )
            
            fig.update_xaxes(
                showline=True,
                linewidth=1,
                linecolor='lightgray',
                gridcolor='rgba(200, 200, 200, 0.2)'
            )
            
            fig.update_yaxes(
                showline=True,
                linewidth=1,
                linecolor='lightgray',
                gridcolor='rgba(200, 200, 200, 0.2)'
            )
        
        return {
            'main_plot': fig1,
            'time_plot': fig2,
            'full_plot': fig3,
            'stress_data': self.stress_data.tolist(),
            'strain_data': self.strain_data.tolist(),
            'time_data': self.time_data.tolist()
        }
    
    def generate_report(self):
        """Улучшенная версия генерации отчета с проверкой путей"""
        import os
        import sys
        from pathlib import Path
        
        try:
            # 1. Гарантируем существование директории
            self.upload_dir = Path(self.upload_dir)
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Создаем отчет в директории: {self.upload_dir}")
            print(f"Права доступа: {oct(os.stat(self.upload_dir).st_mode)[-3:]}")
            
            # 2. Создаем графики
            plots = self.create_plots()
            plot_files = {}
            
            for name in ['main_plot', 'time_plot', 'full_plot']:
                plot_path = self.upload_dir / f"{name}.png"
                
                # Явно проверяем и создаем директорию
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Для Windows: явно открываем файл с указанием кодировки
                if sys.platform == 'win32':
                    with open(plot_path, 'wb') as f:
                        plots[name].write_image(f, format='png', engine='kaleido')
                else:
                    plots[name].write_image(str(plot_path), engine='kaleido')
                
                # Проверка создания файла
                if not plot_path.exists():
                    raise RuntimeError(
                        f"Файл {plot_path} не создан. "
                        f"Права: {oct(plot_path.parent.stat().st_mode)[-3:]}"
                    )
                
                plot_files[name] = str(plot_path)
                print(f"График {name} сохранен: {plot_path} ({plot_path.stat().st_size} bytes)")

            # 3. Генерация отчета
            template_path = Path(__file__).resolve().parent.parent / 'templates' / 'analysis' / 'report_template.docx'
            if not template_path.exists():
                raise FileNotFoundError(f"Шаблон отчета не найден: {template_path}")
            
            report_path = self.upload_dir / "analysis_report.docx"
            doc = DocxTemplate(str(template_path))
            
            context = {
                'name': self.params['sample_name'],
                'width': self.params['width'],
                'length': self.params['length'],
                'height': self.params['height'],
                'form_factor': self.form_factor,
                'load_pic': InlineImage(doc, plot_files['time_plot'], width=Cm(14)),
                'cycles_pic': InlineImage(doc, plot_files['full_plot'], width=Cm(14)),
                'elastic_pic': InlineImage(doc, plot_files['main_plot'], width=Cm(14)),
                'date': datetime.datetime.now().strftime('%d.%m.%Y')
            }
            
            doc.render(context)
            doc.save(str(report_path))
            
            print(f"Отчет успешно создан: {report_path}")
            return str(report_path)
            
        except Exception as e:
            print(f"Критическая ошибка при генерации отчета: {str(e)}", file=sys.stderr)
            # Удаляем временные файлы
            for plot_path in plot_files.values():
                try:
                    Path(plot_path).unlink(missing_ok=True)
                except:
                    pass
            raise RuntimeError(f"Ошибка генерации отчета: {str(e)}")

    def run_analysis(self):
        """Основной метод выполнения анализа"""
        self.load_and_process_data()
        results = self.create_plots()
        results['form_factor'] = self.form_factor
        results['report_path'] = self.generate_report()
        return results
    

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

# Маленикие функции котороые упрощают код и его понимание 
def read_file(txt_file, sep = "\t"):
    """
    Функция используется для прочтения txt файла с испытательной машины WDW-50
    """
    df = pd.read_csv(txt_file, sep=sep, header=None)
    return df

def in_str_to_float(df):
    """
    Функция заменяет , на . и переводит в тип данных float
    """
    df.replace(",", ".", regex=True, inplace=True)
    df = df.astype(float)
    return df

def get_data_in_file(df):
    """
    Функция забирает данные из нужных столбцов для дальнейшей обработке 
    0 - нагрузка, 2 - перемещение, 3 - время 
    """
    force = df[0].values  # нагрузка (Н)
    displacement = df[2].values  # перемещение (мм)
    time = df[3].values  # время (с)
    return force, displacement, time