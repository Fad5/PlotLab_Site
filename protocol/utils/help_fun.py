import pandas as pd
import datetime
import random

def reformat_date(data):
    if isinstance(data, pd.Timestamp):
        day = data.day
        month = data.month
        year = data.year
    else:
        try:
            # Пробуем разобрать дату в формате 'DD.MM.YYYY'
            dt = datetime.strptime(data, "%d.%m.%Y")
        except ValueError:
            try:
                # Пробуем разобрать дату в формате 'YYYY-MM-DD HH:MM:SS'
                dt = datetime.strptime(data, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise ValueError(f"Неизвестный формат даты: {data}")
        
        day = dt.day
        month = dt.month
        year = dt.year
    
    months = {
        1: "января", 2: "февраля", 3: "марта", 4: "апреля",
        5: "мая", 6: "июня", 7: "июля", 8: "августа",
        9: "сентября", 10: "октября", 11: "ноября", 12: "декабря"
    }
    
    return f"«{day}» {months[month]} {year} г."


def dolg(data_prot):
    """
    Возвращает должность в зависимости от даты.
    
    Параметры:
        data_prot (str или pd.Timestamp): Дата в формате 'DD.MM.YYYY' или Timestamp
        
    Возвращает:
        str: 'Инженер 1 категории' или 'Ведущий инженер'
    """
    # Если data_prot — строка, преобразуем в Timestamp
    if isinstance(data_prot, str):
        data_prot = pd.to_datetime(data_prot, format='%d.%m.%Y')
    
    # Задаем пороговую дату (1 апреля 2025 года)
    porog = pd.Timestamp('2025-04-01 00:00:00')
    
    # Сравниваем даты
    if data_prot > porog:
        return 'Ведущий инженер'
    else:
        return 'Инженер 1 категории'
    

def generate_random_float(start, end, precision=2):
    """
    Генерирует случайное число с плавающей точкой в диапазоне [start, end].
    Возвращает строку, где точка заменена на запятую, и гарантирует два знака после запятой.
    
    Параметры:
        start (float): начало диапазона
        end (float): конец диапазона
        precision (int): количество знаков после запятой (по умолчанию 2)
    
    Возвращает:
        str: случайное число в виде строки с запятой, всегда с двумя знаками после запятой
    """
    number = random.uniform(start, end)
    rounded = round(number, precision)
    
    # Форматируем число, чтобы всегда было 2 знака после запятой
    formatted_num = "{:.2f}".format(rounded)
    
    # Заменяем точку на запятую
    return formatted_num.replace('.', ',')