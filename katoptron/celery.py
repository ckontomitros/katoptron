import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'katoptron.settings')

app = Celery('katoptron')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()