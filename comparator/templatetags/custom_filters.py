from django import template

register = template.Library()


@register.filter
def calculate_similarity_score(position_diff):
    """
    Convert position difference to a similarity percentage score
    Lower difference = higher similarity
    """
    if position_diff is None:
        return {
            'percentage': 0,
            'label': 'No Data',
            'category': 'poor'
        }

    # Convert to float if needed
    diff = float(position_diff)

    # Calculate similarity (inverse of difference, scaled 0-100)
    # Assuming differences range from 0 to 0.5 (very different)
    # We'll map 0 -> 100%, 0.25 -> 0%
    similarity = max(0, min(100, 100 * (1 - (diff / 10))))

    # Categorize
    if similarity >= 90:
        category = 'excellent'
        label = 'Excellent Match'
    elif similarity >= 75:
        category = 'good'
        label = 'Good Match'
    elif similarity >= 50:
        category = 'fair'
        label = 'Needs Improvement'
    else:
        category = 'poor'
        label = 'Significant Differences'

    return {
        'percentage': int(similarity),
        'label': label,
        'category': category
    }


@register.filter
def multiply(value, arg):
    """Multiply value by arg"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0


@register.filter
def add(value, arg):
    """Add arg to value"""
    try:
        return float(value) + float(arg)
    except (ValueError, TypeError):
        return 0


@register.filter
def get_priority_class(difference):
    """Return CSS class based on difference magnitude"""
    try:
        diff = float(difference)
        if diff > 0.15:
            return 'danger'
        elif diff > 0.10:
            return 'warning'
        elif diff > 0.05:
            return 'info'
        else:
            return 'success'
    except (ValueError, TypeError):
        return 'secondary'


@register.filter
def get_priority_label(difference):
    """Return priority label based on difference"""
    try:
        diff = float(difference)
        if diff > 0.15:
            return 'Critical'
        elif diff > 0.10:
            return 'High Priority'
        elif diff > 0.05:
            return 'Medium Priority'
        else:
            return 'Good Match'
    except (ValueError, TypeError):
        return 'Unknown'


@register.filter
def format_metric(value, metric_type='position'):
    """Format metric values for display"""
    try:
        val = float(value)
        if metric_type == 'position':
            return f"{val:.3f}"
        elif metric_type == 'angle':
            return f"{val:.1f}°"
        else:
            return f"{val:.2f}"
    except (ValueError, TypeError):
        return "N/A"
@register.filter
def risk_icon(risk_level):
    """Return Bootstrap icon name based on risk level"""
    icons = {
        'high': 'exclamation-triangle-fill',
        'medium': 'exclamation-circle-fill',
        'low': 'info-circle-fill',
    }
    return icons.get(risk_level, 'info-circle-fill')