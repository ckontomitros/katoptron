from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib import messages
from .models import Comparison
from .forms import ComparisonForm
from .tasks import process_comparison


def upload_view(request):
    """Handle video upload and initiate comparison"""
    if request.method == 'POST':
        form = ComparisonForm(request.POST, request.FILES)
        if form.is_valid():
            comparison = form.save()
            # Start background task
            process_comparison(str(comparison.id))
            messages.success(request, 'Videos uploaded successfully! Processing started.')
            return redirect('comparator:detail', comparison_id=comparison.id)
    else:
        form = ComparisonForm()

    return render(request, 'comparator/upload.html', {'form': form})


def comparison_detail(request, comparison_id):
    """Display comparison results"""
    comparison = get_object_or_404(Comparison, id=comparison_id)
    return render(request, 'comparator/results.html', {'comparison': comparison})


def comparison_status(request, comparison_id):
    """AJAX endpoint for checking comparison status"""
    comparison = get_object_or_404(Comparison, id=comparison_id)
    data = {
        'status': comparison.status,
        'created_at': comparison.created_at.isoformat(),
        'completed_at': comparison.completed_at.isoformat() if comparison.completed_at else None,
        'error_message': comparison.error_message,
    }

    if comparison.status == 'completed':
        data.update({
            'avg_position_diff': comparison.avg_position_diff,
            'avg_angle_diff': comparison.avg_angle_diff,
            'dtw_position_distance': comparison.dtw_position_distance,
            'dtw_angle_distance': comparison.dtw_angle_distance,
            'total_aligned_pairs': comparison.total_aligned_pairs,
        })

    return JsonResponse(data)


def comparison_list(request):
    """List all comparisons"""
    comparisons = Comparison.objects.all()
    return render(request, 'comparator/list.html', {'comparisons': comparisons})