import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

def get_chart_data(table_name, period='day'):
    """
    Получает данные для построения графиков по разным периодам
    
    Параметры:
    - table_name: имя таблицы в БД
    - period: период группировки ('day', 'week', 'month', 'year')
    
    Возвращает:
    - DataFrame с колонками: period_label, total_files, total_size_mb
    """
    conn = sqlite3.connect("your_database.db")
    
    # Базовый SQL запрос
    if period == 'day':
        group_sql = "strftime('%Y-%m-%d', upload_time)"
        label_sql = "strftime('%Y-%m-%d', upload_time) AS period_label"
    elif period == 'week':
        group_sql = "strftime('%Y-%W', upload_time)"
        label_sql = "strftime('%Y-%W', upload_time) AS period_label"
    elif period == 'month':
        group_sql = "strftime('%Y-%m', upload_time)"
        label_sql = "strftime('%Y-%m', upload_time) AS period_label"
    elif period == 'year':
        group_sql = "strftime('%Y', upload_time)"
        label_sql = "strftime('%Y', upload_time) AS period_label"
    else:
        raise ValueError("Invalid period. Use 'day', 'week', 'month' or 'year'")
    
    query = f"""
        SELECT 
            {label_sql},
            COUNT(*) as total_files,
            ROUND(SUM(size_bytes) / (1024.0 * 1024.0), 2) as total_size_mb
        FROM {table_name}
        GROUP BY {group_sql}
        ORDER BY upload_time
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def plot_upload_stats(table_name, period='day'):
    """
    Строит графики загрузки данных по выбранному периоду
    
    Параметры:
    - table_name: имя таблицы в БД
    - period: период группировки ('day', 'week', 'month', 'year')
    """
    df = get_chart_data(table_name, period)
    
    if df.empty:
        print(f"No data found in table {table_name}")
        return
    
    # Создаем фигуру с двумя subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # График количества файлов
    ax1.bar(df['period_label'], df['total_files'], co   lor='skyblue')
    ax1.set_title(f'Количество загруженных файлов по {period}')
    ax1.set_ylabel('Количество файлов')
    ax1.tick_params(axis='x', rotation=45)
    
    # График объема данных
    ax2.bar(df['period_label'], df['total_size_mb'], color='salmon')
    ax2.set_title(f'Объем загруженных данных (MB) по {period}')
    ax2.set_ylabel('Объем (MB)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Пример использования:
plot_upload_stats('young_modul', 'day')   # по дням
# plot_upload_stats('your_table_name', 'week')  # по неделям
# plot_upload_stats('your_table_name', 'month') # по месяцам
# plot_upload_stats('your_table_name', 'year')  # по годам