from django.db import models
from django.utils import timezone
import uuid


class Comparison(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video1 = models.FileField(upload_to='videos/')
    video2 = models.FileField(upload_to='videos/')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Results
    result_json = models.FileField(upload_to='results/', null=True, blank=True)
    result_csv = models.FileField(upload_to='results/', null=True, blank=True)
    landmark_csv = models.FileField(upload_to='results/', null=True, blank=True)
    joint_csv = models.FileField(upload_to='results/', null=True, blank=True)
    summary_txt = models.FileField(upload_to='results/', null=True, blank=True)
    comparison_video = models.FileField(upload_to='results/', null=True, blank=True)

    # Statistics
    avg_position_diff = models.FloatField(null=True, blank=True)
    avg_angle_diff = models.FloatField(null=True, blank=True)
    dtw_position_distance = models.FloatField(null=True, blank=True)
    dtw_angle_distance = models.FloatField(null=True, blank=True)
    total_aligned_pairs = models.IntegerField(null=True, blank=True)

    error_message = models.TextField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Comparison {self.id} - {self.status}"