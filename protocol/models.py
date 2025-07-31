from django.db import models
import uuid

class AnalysisTask(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Ожидает обработки'),
        ('processing', 'В обработке'),
        ('completed', 'Завершено'),
        ('failed', 'Ошибка'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    result_file = models.FileField(upload_to='reports/', null=True, blank=True)
    data_archive = models.FileField(upload_to='uploads/data_archives/')
    excel_file = models.FileField(upload_to='uploads/excel_files/')
    
    def __str__(self):
        return f"Анализ #{self.id} - {self.get_status_display()}"