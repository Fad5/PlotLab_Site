import sqlite3
import os
from datetime import datetime

def log_upload_to_db(filepath, table):
    filename = os.path.basename(filepath)
    size_bytes = os.path.getsize(filepath)
    print(filename, 'filename')
    print(size_bytes, 'size_bytes')
    conn = sqlite3.connect("your_database.db")
    cursor = conn.cursor()
    
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    cursor.execute(f"""
        INSERT INTO {table} (filename, size_bytes, upload_time)
        VALUES (?, ?, ?)
    """, (filename, size_bytes, datetime.now()))
    
    conn.commit()
    conn.close()
