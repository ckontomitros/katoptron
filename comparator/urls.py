from django.urls import path
from . import views

app_name = 'comparator'

urlpatterns = [
    path('', views.upload_view, name='upload'),
    path('comparison/<uuid:comparison_id>/', views.comparison_detail, name='detail'),
    path('comparison/<uuid:comparison_id>/status/', views.comparison_status, name='status'),
    path('comparisons/', views.comparison_list, name='list'),
]