import numpy as np
from scipy.signal import find_peaks
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_ALIGN_VERTICAL
import datetime
import tempfile

# Настройки графиков
plt.rcParams['font.family'] = 'Times New Roman'  # Установка шрифта Times New Roman
plt.rcParams['font.style'] = 'normal'  # Обычный шрифт (не italic)
plt.rcParams['font.size'] = 12  # Размер шрифта 12
fontsize = 12
linewidth = 2
fontweight = 'bold'  # Обычный шрифт (не bold)

def save_plot(fig, filename):
    """Сохраняет график в файл и закрывает фигуру"""
    # Создаем временную директорию для графиков, если ее нет
    temp_dir = os.path.join(tempfile.gettempdir(), 'elastic_modulus_plots')
    os.makedirs(temp_dir, exist_ok=True)
    
    full_path = os.path.join(temp_dir, filename)
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return full_path

def load_data(file_path):
    """Загружает данные из файла"""
    try:
        df = pd.read_csv(file_path, sep="\t", header=None)
        df.replace(",", ".", regex=True, inplace=True)
        df = df.astype(float)
        return df
    except Exception as e:
        print(f"Ошибка: Не удалось загрузить файл:\n{str(e)}")
        return None

def data_modul_young(df, width, length, height):
    """Вычисляет данные для построения графиков модуля упругости"""
    if df is None:
        print("Данные не загружены!")
        return None
    
    width = float(width)
    length = float(length)
    initial_height = float(height)
    area = width * length

    k = np.argmax(df[0].values > 0)
    M = df.values[k:, :4] - df.values[k, :4]
    sr = len(M) / M[-1, 3] if M[-1, 3] != 0 else 10

    F = M[:, 0]
    S = M[:, 2]
    T = np.arange(len(F)) / sr

    peaks, _ = find_peaks(S, height=0.5 * np.max(S))
    if len(peaks) < 3:
        print("Недостаточно пиков для анализа (минимум 3)")
        return None

    cycle_index = 1
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

        Pr[i] = (F1[idx1] + F1[idx2]) / 2 / area * 1e-6
        delta_F = F1[idx2] - F1[idx1]
        delta_S = S1[idx2] - S1[idx1]

        if delta_S != 0:
            E1[i] = (delta_F / area * 1e-6) / (delta_S / initial_height)

        Eps1[i] = (S1[idx1] + S1[idx2]) / 2 / initial_height

    if len(Pr) > 0:
        Pr = Pr - Pr[0]

    Pr, E1, Eps1 = Pr[3:], E1[3:], Eps1[3:]
    Eps1 = Eps1[int(len(Eps1)/2):] 
    E1 = E1[int(len(E1)/2):]  
    Pr = Pr[int(len(Pr)/2):] 

    if Pr.size > 2:
        min_Pr = np.min(Pr)
        Eps1 = Eps1 - Eps1[0]
        Pr_ = Pr + (-min_Pr)
    else:
        Pr_ = Pr

    E1 = E1[:] * 1_000_000
    Eps1 = Eps1[:] * 100 
    Pr = Pr_[:] * 1_000_000
    
    return E1, Eps1, Pr

def create_plot_modul_young(E1, Eps1, Pr, name_sample, form_factor):
    """Создает график модуля Юнга"""
    fig, (ax4, ax5) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)

    ax4.plot(Pr, E1, 'k-', linewidth=linewidth)
    # ax4.set_title(f'{name_sample} | Коэффициент формы q = {form_factor:.2f}', 
    #              fontsize=fontsize, fontweight=fontweight)
    ax4.set_xlabel('Удельное давление, МПа', fontsize=fontsize, fontweight=fontweight)
    ax4.set_ylabel('Модуль упругости, МПа', fontsize=fontsize, fontweight=fontweight)
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.set_xlim(left=0)
    ax4.set_ylim(bottom=0)

    ax5.plot(Pr, Eps1, 'k-', linewidth=linewidth)
    ax5.set_xlabel('Удельное давление, МПа', fontsize=fontsize, fontweight=fontweight)
    ax5.set_ylabel('Относительная деформация, %', fontsize=fontsize, fontweight=fontweight)
    ax5.grid(True, linestyle='--', alpha=0.6)
    ax5.set_xlim(left=0)
    ax5.set_ylim(bottom=0)

    plt.tight_layout()
    return save_plot(fig, f"{name_sample}_modul_young.png")

def full_plot(Time, Disp, Forse, name_sample, form_factor):
    """Создает полный график зависимости перемещения и нагрузки от времени"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(Time, Disp, 'b', label='Смещение (мм)', linewidth=linewidth)
    ax1.set_ylabel('Перемещение, мм', fontsize=fontsize, fontweight=fontweight, color='blue')
    ax1.set_xlabel('Время, с', fontsize=fontsize, fontweight=fontweight)
    ax1.grid(True)
    ax1.set_ylim([0, math.ceil(max(Disp) + 0.5)])

    ax1_force = ax1.twinx()
    ax1_force.plot(Time, Forse, 'r', label='Нагрузка (Н)', linewidth=linewidth)
    ax1_force.set_ylabel('Нагрузка, Н', fontsize=fontsize, fontweight=fontweight, color='red')

    # ax1.set_title(f'{name_sample} | Коэффициент формы q = {form_factor}', 
    #              fontsize=fontsize, fontweight=fontweight)

    plt.tight_layout()
    return save_plot(fig, f"{name_sample}_full_plot.png")

def plot_cycles_only(Forse, Disp, locs, name_sample, form_factor):
    """Создает графики циклов нагружения"""
    if len(locs) < 1:
        print(f"Недостаточно данных для построения графика {name_sample}")
        return None

    def colors_labels(n):
        base_colors = ['k', 'g', 'r', 'b', 'm', 'c', 'y']
        n = min(n, len(base_colors))
        labels = [f'Цикл {i+1}' for i in range(n)]
        return base_colors[:n], labels

    n_locs_rang = min(6, len(locs))
    colors, labels = colors_labels(n_locs_rang)
    base_len = locs[0] if len(locs) > 0 else len(Disp)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_locs_rang):
        start = max(locs[i] - base_len + 1, 0)
        end = min(locs[i] + base_len, len(Disp))
        ax.plot(Disp[start:end], Forse[start:end], colors[i], label=labels[i], linewidth=linewidth)

    ax.set_xlabel('Перемещение, мм', fontsize=fontsize, fontweight=fontweight)
    ax.set_ylabel('Нагрузка, Н', fontsize=fontsize, fontweight=fontweight)
    ax.grid(True)
    ax.legend(loc='lower right')
    # ax.set_title(f'Циклы нагружения\n{name_sample} | q = {form_factor:.2f}',
    #             fontsize=fontsize, fontweight=fontweight)

    plt.tight_layout()
    return save_plot(fig, f"{name_sample}_cycles.png")

def find_loading_cycles(df):
    """Находит циклы нагружения в данных"""
    peaks, _ = find_peaks(df[0].values, prominence=np.std(df[0].values)/2)
    return peaks.tolist()

def process_sample_file(filepath, sample_name, width, length, height, mass):
    """Обрабатывает данные одного образца и создает графики"""
    df = load_data(filepath)
    if df is None:
        return None

    force = df[0].values
    displacement = df[2].values
    time = df[3].values
    locs = find_loading_cycles(df)

    full_plot_path = full_plot(time, displacement, force, sample_name, length)
    E1, Eps1, Pr = data_modul_young(df, width, length, height)
    
    if E1 is None:
        return None
        
    modul_plot_path = create_plot_modul_young(E1, Eps1, Pr, sample_name, length)
    cycles_plot_path = plot_cycles_only(force, displacement, locs, sample_name, length)

    return {
        'name': sample_name,
        'width': width,
        'length': length,
        'height': height,
        'mass': mass,
        'full_plot': full_plot_path,
        'modul_plot': modul_plot_path,
        'cycles_plot': cycles_plot_path if cycles_plot_path else None
    }

def add_custom_heading(doc, text, level=1):
    """Добавляет заголовок с ручным форматированием"""
    p = doc.add_paragraph()
    run = p.add_run(text)
    
    # Настройки форматирования в зависимости от уровня заголовка
    if level == 1:
        run.bold = True
        run.font.size = Pt(16)
    elif level == 2:
        run.bold = True
        run.font.size = Pt(14)
    else:
        run.bold = True
        run.font.size = Pt(12)
    
    return p

def insert_samples_table(doc, samples_data, decimal_places={'length': 2, 'width': 2, 'height': 2, 'mass': 2}):
    """Создает таблицу с параметрами образцов с настраиваемым количеством знаков после запятой
    
    Args:
        doc: Объект документа Word
        samples_data: Данные образцов
        decimal_places: Словарь с количеством знаков после запятой для каждого параметра
            По умолчанию: длина/ширина/высота/ масса - 2 знака
    """
    table = doc.add_table(rows=len(samples_data)+1, cols=5)
    table.style = 'Table Grid'
    
    for row in table.rows:
        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    
    # Заголовки таблицы
    hdr_cells = table.rows[0].cells
    headers = ["Образец", "Длина, мм", "Ширина, мм", "Толщина, мм", "Масса, г"]
    
    for i, header in enumerate(headers):
        p = hdr_cells[i].paragraphs[0]
        run = p.add_run(header)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    
    # Данные образцов с форматированием
    for row_idx, sample in enumerate(samples_data, start=1):
        row_cells = table.rows[row_idx].cells
        
        # Форматируем каждое значение с нужным количеством знаков
        length_str = f"{float(sample['length']):.{decimal_places['length']}f}".replace('.', ',')
        width_str = f"{float(sample['width']):.{decimal_places['width']}f}".replace('.', ',')
        height_str = f"{float(sample['height']):.{decimal_places['height']}f}".replace('.', ',')
        mass_str = f"{float(sample['mass']):.{decimal_places['mass']}f}".replace('.', ',')
        
        data = [
            sample['name'],
            length_str,
            width_str,
            height_str,
            mass_str
        ]
        
        for col_idx, value in enumerate(data):
            p = row_cells[col_idx].paragraphs[0]
            run = p.add_run(value)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
            p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER


def insert_samples_graphs(doc, samples_data):
    """Вставляет графики с подписями"""
    figure_counter = 1
    total_samples = len(samples_data)
    total_graphs = sum(1 for sample in samples_data for plot_type, _ in [
        ('full_plot', "График зависимости перемещения и нагрузки от времени"),
        ('modul_plot', "График модуля упругости"),
        ('cycles_plot', "Графики циклов нагружения")
    ] if sample.get(plot_type) and os.path.exists(sample[plot_type]))
    
    for i, sample in enumerate(samples_data):
        # Добавляем заголовок образца

        graphs = [
            ('full_plot', f"График зависимости перемещения и нагрузки от времени образца {sample['name']}"),
            ('modul_plot', f"График модуля упругости образца {sample['name']}"),
            ('cycles_plot', f"Графики циклов нагружения образца {sample['name']}")
        ]
        
        for plot_type, description in graphs:
            if sample.get(plot_type) and os.path.exists(sample[plot_type]):
                # Вставляем график по центру
                para = doc.add_paragraph()
                para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                run = para.add_run()
                run.add_picture(sample[plot_type], width=Inches(8.5))
                
                # Добавляем подпись под графиком
                p = doc.add_paragraph()
                run = p.add_run(f"Рисунок {figure_counter} - {description}")
                run.italic = False
                run.font.size = Pt(12)
                run.font.name = 'Times New Roman'
                p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                
                # Добавляем разрыв страницы
                doc.add_page_break()
                figure_counter += 1

def fill_template(template_path, samples_data, output_filename):
    """Заполняет шаблон документа данными"""
    doc = Document(template_path)
    
    # Замена стандартных плейсхолдеров
    for paragraph in doc.paragraphs:
        if '{ДАТА}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{ДАТА}', datetime.datetime.now().strftime("%d.%m.%Y"))
    
    # Вставка таблицы и графиков
    for paragraph in list(doc.paragraphs):
        if '{TABLE_PLACEHOLDER}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{TABLE_PLACEHOLDER}', '')
            insert_samples_table(doc, samples_data)
            
        elif '{GRAPHS_PLACEHOLDER}' in paragraph.text:
            paragraph.text = paragraph.text.replace('{GRAPHS_PLACEHOLDER}', '')
            insert_samples_graphs(doc, samples_data)
    
    doc.save(output_filename)

def cleanup_temp_files(samples_data):
    """Удаляет временные файлы с графиками"""
    for sample in samples_data:
        for plot_type in ['full_plot', 'modul_plot', 'cycles_plot']:
            if sample.get(plot_type) and os.path.exists(sample[plot_type]):
                try:
                    os.remove(sample[plot_type])
                except:
                    pass

def genaretion_plot(data_list, data_excel, template_path=None, output_filename='Итоговый_протокол.docx'):
    try:
        # Определяем путь к шаблону по умолчанию
        if template_path is None:
            template_path = os.path.join(os.path.dirname(__file__), 'template.docx')
            if not os.path.exists(template_path):
                # Создаем пустой документ, если шаблона нет
                doc = Document()
                doc.add_paragraph('{ДАТА}')
                doc.add_paragraph('{TABLE_PLACEHOLDER}')
                doc.add_paragraph('{GRAPHS_PLACEHOLDER}')
                doc.save(template_path)
        
        # Убедимся, что названия образцов в Excel - строки
        data_excel['Образец'] = data_excel['Образец'].astype(str).str.strip()
        data_excel.columns = data_excel.columns.str.strip()
        samples_data = []
        
        for filepath in data_list:
            filename = os.path.basename(filepath)
            # Извлекаем номер образца из имени файла (удаляем .txt)
            sample_name = os.path.splitext(filename)[0]
            
            # Ищем соответствие в Excel (убедимся, что сравниваем строки)
            row = data_excel[data_excel['Образец'] == sample_name]
            
            if row.empty:
                print(f"Образец {sample_name} не найден в таблице!")
                continue

            try:
                width = float(row['Ширина'].values[0].replace(',', '.'))
                length = float(row['Длина'].values[0].replace(',', '.'))
                height = float(row['Высота'].values[0].replace(',', '.'))
                mass = float(row['Масса'].values[0].replace(',', '.'))

            except (IndexError, ValueError) as e:
                print(f"Ошибка получения параметров для образца {sample_name}: {str(e)}")
                continue

            sample_data = process_sample_file(filepath, sample_name, width, length, height, mass)
            if sample_data:
                samples_data.append(sample_data)
        
        if samples_data:
            fill_template(template_path, samples_data, output_filename)
            cleanup_temp_files(samples_data)
            print(f"Протокол успешно сохранен в файл: {output_filename}")
            return True
        else:
            print("Нет данных для создания протокола!")
            return False
    except Exception as e:
        print(f"Ошибка в genaretion_plot: {str(e)}")
        return False