from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib import messages
from .models import Comparison
from .forms import ComparisonForm
from .tasks import process_comparison
import json
import os

from .tennis_analyzer import TennisForehandAnalyzer


def upload_view(request):
    """Handle video upload and initiate comparison"""
    if request.method == 'POST':
        form = ComparisonForm(request.POST, request.FILES)
        if form.is_valid():
            comparison = form.save()
            # Start background task
            process_comparison(str(comparison.id))
            messages.success(request, 'Videos uploaded successfully! Analysis in progress...')
            return redirect('comparator:detail', comparison_id=comparison.id)
    else:
        form = ComparisonForm()

    return render(request, 'comparator/upload.html', {'form': form})


def comparison_detail(request, comparison_id):
    """Display comparison results with enhanced insights"""
    comparison = get_object_or_404(Comparison, id=comparison_id)

    context = {
        'comparison': comparison,
        'body_parts': None,
        'top_issues': None,
        'insights': None,
    }

    # If completed, load and parse additional data
    if comparison.status == 'completed' and comparison.landmark_csv:
        try:
            # Read landmark statistics
            landmark_path = comparison.landmark_csv.path
            if os.path.exists(landmark_path):
                import pandas as pd
                df = pd.read_csv(landmark_path)

                # Sort by average difference
                df_sorted = df.sort_values('avg_difference', ascending=False)

                # Categorize body parts
                upper_body = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                              'left_wrist', 'right_wrist', 'nose', 'left_eye', 'right_eye']
                lower_body = ['left_hip', 'right_hip', 'left_knee', 'right_knee',
                              'left_ankle', 'right_ankle', 'left_heel', 'right_heel']

                # Get top issues per category
                top_upper = df_sorted[df_sorted['landmark_name'].isin(upper_body)].head(3)
                top_lower = df_sorted[df_sorted['landmark_name'].isin(lower_body)].head(3)

                context['body_parts'] = {
                    'upper': [
                        {
                            'name': format_landmark_name(row['landmark_name']),
                            'difference': row['avg_difference'],
                            'priority': get_priority(row['avg_difference'])
                        }
                        for _, row in top_upper.iterrows()
                    ],
                    'lower': [
                        {
                            'name': format_landmark_name(row['landmark_name']),
                            'difference': row['avg_difference'],
                            'priority': get_priority(row['avg_difference'])
                        }
                        for _, row in top_lower.iterrows()
                    ]
                }

                # Overall insights
                context['insights'] = generate_insights(comparison, df)

        except Exception as e:
            print(f"Error loading landmark data: {e}")

    return render(request, 'comparator/results.html', context)


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


# Helper functions
def format_landmark_name(name):
    """Convert snake_case to Title Case"""
    return name.replace('_', ' ').title()


def get_priority(difference):
    """Determine priority level based on difference"""
    if difference > 0.15:
        return {'level': 'critical', 'label': 'Critical', 'class': 'danger', 'icon': 'circle-fill'}
    elif difference > 0.10:
        return {'level': 'high', 'label': 'High Priority', 'class': 'warning', 'icon': 'circle-fill'}
    elif difference > 0.05:
        return {'level': 'medium', 'label': 'Medium Priority', 'class': 'info', 'icon': 'circle-fill'}
    else:
        return {'level': 'good', 'label': 'Good Match', 'class': 'success', 'icon': 'circle-fill'}


def generate_insights(comparison, landmark_df):
    """Generate actionable insights from the data"""
    insights = {
        'overall_assessment': '',
        'key_strengths': [],
        'areas_for_improvement': [],
        'recommendations': []
    }

    # Overall assessment
    avg_diff = comparison.avg_position_diff
    avg_angle = comparison.avg_angle_diff

    if avg_diff < 0.05 and avg_angle < 10:
        insights['overall_assessment'] = "Outstanding! Your movement pattern is nearly identical to the reference."
    elif avg_diff < 0.10 and avg_angle < 20:
        insights['overall_assessment'] = "Great job! Your form is solid with only minor deviations."
    elif avg_diff < 0.20 and avg_angle < 30:
        insights['overall_assessment'] = "Good foundation, but there's room for improvement in several areas."
    else:
        insights[
            'overall_assessment'] = "Significant differences detected. Focus on the highlighted areas for best results."

    # Key strengths (lowest differences)
    best_landmarks = landmark_df.nsmallest(3, 'avg_difference')
    for _, row in best_landmarks.iterrows():
        insights['key_strengths'].append({
            'part': format_landmark_name(row['landmark_name']),
            'score': f"{(1 - row['avg_difference']) * 100:.0f}%"
        })

    # Areas for improvement (highest differences)
    worst_landmarks = landmark_df.nlargest(3, 'avg_difference')
    for _, row in worst_landmarks.iterrows():
        insights['areas_for_improvement'].append({
            'part': format_landmark_name(row['landmark_name']),
            'difference': f"{row['avg_difference']:.3f}"
        })

    # Generate recommendations
    if avg_angle > 25:
        insights['recommendations'].append({
            'icon': 'compass',
            'title': 'Focus on Joint Angles',
            'description': 'Your joint angles differ significantly. Pay special attention to elbow, knee, and shoulder bending.'
        })

    if avg_diff > 0.15:
        insights['recommendations'].append({
            'icon': 'rulers',
            'title': 'Improve Body Positioning',
            'description': 'Work on maintaining proper body position throughout the entire movement.'
        })

    # Check for asymmetry
    left_landmarks = landmark_df[landmark_df['landmark_name'].str.contains('left')]
    right_landmarks = landmark_df[landmark_df['landmark_name'].str.contains('right')]

    if len(left_landmarks) > 0 and len(right_landmarks) > 0:
        left_avg = left_landmarks['avg_difference'].mean()
        right_avg = right_landmarks['avg_difference'].mean()

        if abs(left_avg - right_avg) > 0.05:
            side = 'left' if left_avg > right_avg else 'right'
            insights['recommendations'].append({
                'icon': 'arrows-angle-expand',
                'title': 'Address Asymmetry',
                'description': f'Your {side} side shows more variation. Focus on symmetry and balance.'
            })

    insights['recommendations'].append({
        'icon': 'film',
        'title': 'Use Slow Motion',
        'description': 'Watch the comparison video at 0.5x speed to identify exact moments where form breaks down.'
    })

    insights['recommendations'].append({
        'icon': 'arrow-repeat',
        'title': 'Track Progress',
        'description': 'Re-record after practicing and compare again to measure improvement over time.'
    })

    return insights


def comparison_detail_with_llm(request, comparison_id):
    """Enhanced comparison view with LLM tennis analysis"""
    comparison = get_object_or_404(Comparison, id=comparison_id)

    context = {
        'comparison': comparison,
        'tennis_analysis': None,
    }

    if comparison.status == 'completed' and comparison.landmark_csv:
        try:
            # Load landmark data
            import pandas as pd
            df = pd.read_csv(comparison.landmark_csv.path)
            landmark_stats = df.set_index('landmark_name').to_dict('index')

            # Prepare comparison data
            comparison_data = {
                'avg_position_diff': comparison.avg_position_diff,
                'avg_angle_diff': comparison.avg_angle_diff,
                'total_aligned_pairs': comparison.total_aligned_pairs,
            }

            # Generate LLM analysis
            analyzer = TennisForehandAnalyzer()
            tennis_analysis = analyzer.generate_tennis_analysis(
                comparison_data,
                landmark_stats
            )

            context['tennis_analysis'] = tennis_analysis.get('analysis', {})

        except Exception as e:
            print(f"Error generating tennis analysis: {e}")

    return render(request, 'comparator/tennis_results.html', context)