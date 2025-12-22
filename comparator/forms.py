from django import forms
from .models import Comparison


class ComparisonForm(forms.ModelForm):
    class Meta:
        model = Comparison
        fields = ['video1', 'video2']
        widgets = {
            'video1': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'video/*'
            }),
            'video2': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'video/*'
            }),
        }
        labels = {
            'video1': 'Reference Video',
            'video2': 'Comparison Video',
        }

    def clean_video1(self):
        video = self.cleaned_data.get('video1')
        if video:
            if video.size > 100 * 1024 * 1024:  # 100MB limit
                raise forms.ValidationError("Video file too large ( > 100MB )")
        return video

    def clean_video2(self):
        video = self.cleaned_data.get('video2')
        if video:
            if video.size > 100 * 1024 * 1024:  # 100MB limit
                raise forms.ValidationError("Video file too large ( > 100MB )")
        return video