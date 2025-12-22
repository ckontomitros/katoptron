from django.contrib import admin
from .models import Comparison


@admin.register(Comparison)
class ComparisonAdmin(admin.ModelAdmin):
    list_display = ['short_id', 'status', 'created_at', 'completed_at',
                    'avg_position_diff', 'avg_angle_diff', 'total_aligned_pairs']
    list_filter = ['status', 'created_at']
    search_fields = ['id', 'error_message']
    readonly_fields = ['id', 'created_at', 'completed_at', 'avg_position_diff',
                       'avg_angle_diff', 'dtw_position_distance', 'dtw_angle_distance',
                       'total_aligned_pairs', 'error_message']

    fieldsets = (
        ('Basic Info', {
            'fields': ('id', 'status', 'created_at', 'completed_at')
        }),
        ('Videos', {
            'fields': ('video1', 'video2')
        }),
        ('Statistics', {
            'fields': ('avg_position_diff', 'avg_angle_diff',
                       'dtw_position_distance', 'dtw_angle_distance',
                       'total_aligned_pairs'),
            'classes': ('collapse',)
        }),
        ('Results', {
            'fields': ('result_json', 'result_csv', 'landmark_csv',
                       'joint_csv', 'summary_txt', 'comparison_video'),
            'classes': ('collapse',)
        }),
        ('Errors', {
            'fields': ('error_message',),
            'classes': ('collapse',)
        }),
    )

    def short_id(self, obj):
        return str(obj.id)[:8]

    short_id.short_description = 'ID'

    actions = ['delete_selected']

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.select_related()