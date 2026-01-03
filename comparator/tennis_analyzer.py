# ============================================================================
# File: comparator/tennis_analyzer.py (OpenAI Version)
# ============================================================================
"""
Tennis-specific analysis using OpenAI GPT-4 for forehand technique feedback
"""
from django.shortcuts import render, get_object_or_404
from openai import OpenAI
import os
from typing import Dict, List
import json

from comparator.models import Comparison
from decouple import config

OPENAI_API_KEY = config('OPENAI_API_KEY', default='')

class TennisForehandAnalyzer:
    """
    Analyzes tennis forehand technique and provides personalized coaching feedback
    using OpenAI GPT-4
    """
    
    def __init__(self, api_key=None):
        """Initialize with OpenAI API key"""
        self.api_key = OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Key tennis forehand checkpoints
        self.tennis_landmarks = {
            'preparation': ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip'],
            'backswing': ['right_shoulder', 'right_elbow', 'left_shoulder'],
            'forward_swing': ['right_shoulder', 'right_elbow', 'right_wrist', 'right_hip', 'left_hip'],
            'contact': ['right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder'],
            'follow_through': ['right_shoulder', 'right_elbow', 'right_wrist', 'left_shoulder']
        }
    
    def generate_tennis_analysis(self, comparison_data: Dict, landmark_stats: Dict) -> Dict:
        """
        Generate comprehensive tennis forehand analysis using GPT-4
        
        Args:
            comparison_data: Dictionary with comparison metrics
            landmark_stats: Dictionary with per-landmark statistics
            
        Returns:
            Dictionary with LLM-generated insights and recommendations
        """
        
        # Prepare structured data for the LLM
        analysis_data = self._prepare_analysis_data(comparison_data, landmark_stats)
        
        # Generate analysis using GPT-4
        prompt = self._build_tennis_prompt(analysis_data)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # or "gpt-4o" for faster/cheaper
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert tennis coach with 20+ years of experience 
                        analyzing forehand technique. You specialize in biomechanics and have coached 
                        players from beginners to ATP/WTA professionals. You provide specific, 
                        actionable feedback focused on technique improvement."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},  # Force JSON response
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse the response
            analysis_text = response.choices[0].message.content
            parsed_analysis = self._parse_llm_response(analysis_text)
            
            return {
                'success': True,
                'analysis': parsed_analysis,
                'raw_response': analysis_text,
                'model': response.model,
                'tokens_used': response.usage.total_tokens
            }
            
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return {
                'success': False,
                'error': str(e),
                'analysis': self._generate_fallback_analysis(analysis_data)
            }
    
    def _prepare_analysis_data(self, comparison_data: Dict, landmark_stats: Dict) -> Dict:
        """Structure the data for LLM analysis"""
        
        # Identify worst performing areas
        worst_landmarks = sorted(
            landmark_stats.items(),
            key=lambda x: x[1].get('avg_difference', 0),
            reverse=True
        )[:5]
        
        # Identify best performing areas
        best_landmarks = sorted(
            landmark_stats.items(),
            key=lambda x: x[1].get('avg_difference', 0)
        )[:3]
        
        # Categorize issues by swing phase
        phase_issues = self._categorize_by_phase(landmark_stats)
        
        return {
            'overall_metrics': {
                'similarity_score': self._calculate_similarity(comparison_data.get('avg_position_diff', 0)),
                'avg_position_diff': comparison_data.get('avg_position_diff', 0),
                'avg_angle_diff': comparison_data.get('avg_angle_diff', 0),
                'total_frames': comparison_data.get('total_aligned_pairs', 0)
            },
            'problem_areas': [
                {
                    'body_part': self._format_landmark_name(lm[0]),
                    'difference': round(lm[1].get('avg_difference', 0), 3),
                    'severity': self._get_severity(lm[1].get('avg_difference', 0))
                }
                for lm in worst_landmarks
            ],
            'strengths': [
                {
                    'body_part': self._format_landmark_name(lm[0]),
                    'difference': round(lm[1].get('avg_difference', 0), 3)
                }
                for lm in best_landmarks
            ],
            'phase_analysis': phase_issues
        }
    
    def _categorize_by_phase(self, landmark_stats: Dict) -> Dict:
        """Categorize issues by tennis swing phase"""
        phases = {}
        
        for phase, landmarks in self.tennis_landmarks.items():
            phase_diffs = []
            for landmark in landmarks:
                if landmark in landmark_stats:
                    phase_diffs.append(landmark_stats[landmark].get('avg_difference', 0))
            
            if phase_diffs:
                phases[phase] = {
                    'avg_difference': sum(phase_diffs) / len(phase_diffs),
                    'landmarks': landmarks
                }
        
        return phases
    
    def _build_tennis_prompt(self, data: Dict) -> str:
        """Build the prompt for GPT-4"""
        
        prompt = f"""You are analyzing a tennis player's forehand technique compared to a professional reference.

ANALYSIS DATA:
- Overall Similarity Score: {data['overall_metrics']['similarity_score']}%
- Average Position Difference: {data['overall_metrics']['avg_position_diff']:.3f}
- Average Angle Difference: {data['overall_metrics']['avg_angle_diff']:.1f}°
- Frames Analyzed: {data['overall_metrics']['total_frames']}

TOP PROBLEM AREAS (ranked by difference from professional form):
{self._format_list(data['problem_areas'], lambda x: f"- {x['body_part']}: {x['difference']} ({x['severity']} priority)")}

PLAYER STRENGTHS:
{self._format_list(data['strengths'], lambda x: f"- {x['body_part']}: Excellent form (difference: {x['difference']})")}

SWING PHASE BREAKDOWN:
{self._format_phase_analysis(data['phase_analysis'])}

Please provide a comprehensive tennis coaching analysis in JSON format with these exact keys:

{{
  "overall_assessment": "2-3 sentences about the player's forehand quality and main takeaway",
  "skill_level": "beginner|intermediate|advanced",
  "critical_issues": [
    {{
      "issue": "specific biomechanical problem",
      "impact": "how this affects power, accuracy, spin, or consistency",
      "injury_risk": "low|medium|high"
    }}
  ],
  "technical_recommendations": [
    {{
      "phase": "preparation|backswing|forward_swing|contact|follow_through",
      "problem": "what's wrong",
      "correction": "how to fix it",
      "drill": "specific drill or exercise to practice"
    }}
  ],
  "strengths": ["list of 2-3 things they're doing well"],
  "practice_plan": {{
    "immediate": ["2-3 things to focus on this week"],
    "medium_term": ["2-3 goals for this month"],
    "progression": "how to progress from beginner to advanced practice"
  }},
  "coaching_notes": "any additional important tips or warnings"
}}

Focus on:
1. Biomechanics and technique (not equipment or mental game)
2. Specific, measurable corrections (e.g., "increase elbow bend to 90°")
3. Practical drills that can be done on any court
4. Professional tennis terminology
5. Safety and injury prevention

Be direct and actionable. Avoid generic advice."""

        return prompt
    
    def _format_list(self, items: List, formatter) -> str:
        """Format list items for prompt"""
        return '\n'.join([formatter(item) for item in items]) if items else "None"
    
    def _format_phase_analysis(self, phases: Dict) -> str:
        """Format phase analysis for prompt"""
        if not phases:
            return "No phase data available"
        
        lines = []
        for phase, data in sorted(phases.items(), key=lambda x: x[1]['avg_difference'], reverse=True):
            lines.append(f"- {phase.replace('_', ' ').title()}: avg difference {data['avg_difference']:.3f}")
        return '\n'.join(lines)
    
    def _parse_llm_response(self, response_text: str) -> Dict:
        """Parse the LLM JSON response"""
        try:
            # GPT-4 with response_format should return clean JSON
            parsed = json.loads(response_text)
            
            # Validate required keys
            required_keys = ['overall_assessment', 'skill_level', 'critical_issues', 
                           'technical_recommendations', 'strengths', 'practice_plan']
            
            for key in required_keys:
                if key not in parsed:
                    print(f"Warning: Missing key '{key}' in response")
                    parsed[key] = [] if key != 'overall_assessment' else 'Analysis incomplete'
            
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Response text: {response_text[:200]}...")
            
            # Fallback: try to extract JSON from markdown code blocks
            if '```json' in response_text:
                try:
                    json_str = response_text.split('```json')[1].split('```')[0].strip()
                    return json.loads(json_str)
                except:
                    pass
            
            # Return structured fallback
            return {
                'overall_assessment': 'Unable to parse detailed analysis. See raw response.',
                'skill_level': 'unknown',
                'critical_issues': [],
                'technical_recommendations': [],
                'strengths': [],
                'practice_plan': {},
                'coaching_notes': response_text
            }
    
    def _generate_fallback_analysis(self, data: Dict) -> Dict:
        """Generate basic analysis if LLM fails"""
        score = data['overall_metrics']['similarity_score']
        
        if score >= 85:
            assessment = "Excellent forehand technique! Your form closely matches professional standards with minimal differences in key positions."
        elif score >= 70:
            assessment = "Good forehand foundation. Your overall form is solid, but specific areas could be refined for better consistency and power."
        elif score >= 50:
            assessment = "Developing forehand with potential. Several key technical areas need attention to reach the next level."
        else:
            assessment = "Significant differences from professional form detected. Focus on building strong fundamentals with proper technique."
        
        # Determine skill level from score
        if score >= 85:
            skill_level = 'advanced'
        elif score >= 65:
            skill_level = 'intermediate'
        else:
            skill_level = 'beginner'
        
        return {
            'overall_assessment': assessment,
            'skill_level': skill_level,
            'critical_issues': [
                {
                    'issue': area['body_part'] + ' positioning',
                    'impact': 'Affects shot consistency and power generation',
                    'injury_risk': 'medium' if area['severity'] == 'high' else 'low'
                }
                for area in data['problem_areas'][:3]
            ],
            'technical_recommendations': [
                {
                    'phase': 'contact',
                    'problem': 'Body alignment and timing inconsistencies',
                    'correction': 'Focus on proper weight transfer from back to front foot while rotating hips',
                    'drill': 'Practice shadow swings and stationary ball feeds, focusing on consistent contact point'
                }
            ],
            'strengths': [s['body_part'] + ' positioning' for s in data['strengths']],
            'practice_plan': {
                'immediate': [
                    'Focus on grip and stance fundamentals',
                    'Practice unit turn and shoulder rotation',
                    'Work on consistent contact point'
                ],
                'medium_term': [
                    'Build muscle memory with repetition drills',
                    'Progress from stationary to moving ball feeds',
                    'Add variety drills (cross-court, down-the-line)'
                ],
                'progression': 'Start with stationary feeds to groove technique, then progress to hand-fed balls, ball machine, and finally live rally situations while maintaining form.'
            },
            'coaching_notes': 'Consider working with a certified coach for personalized feedback and correction. Video yourself regularly to track progress and identify areas for improvement.'
        }
    
    def _calculate_similarity(self, position_diff: float) -> int:
        """Convert position difference to similarity percentage"""
        return max(0, min(100, int(100 * (1 - (position_diff / 0.25)))))
    
    def _format_landmark_name(self, name: str) -> str:
        """Format landmark name for display"""
        return name.replace('_', ' ').title()
    
    def _get_severity(self, difference: float) -> str:
        """Determine severity level"""
        if difference > 0.15:
            return 'high'
        elif difference > 0.10:
            return 'medium'
        else:
            return 'low'


# ============================================================================
# Alternative: Use GPT-4o (Faster and Cheaper)
# ============================================================================
"""
GPT-4o is faster and cheaper than GPT-4 Turbo:
- GPT-4 Turbo: $10/M input, $30/M output
- GPT-4o: $5/M input, $15/M output (50% cheaper!)
- GPT-4o-mini: $0.15/M input, $0.60/M output (98% cheaper, but less capable)

To use GPT-4o instead:
Change model in the API call to: "gpt-4o"
"""


# ============================================================================
# Integration with Django Views
# ============================================================================
"""
Add to comparator/views.py:
"""



# ============================================================================
# Cost Comparison
# ============================================================================
"""
OpenAI Pricing (as of 2024):

Model               Input (per 1M)    Output (per 1M)   Cost per Analysis
----------------    --------------    ---------------   -----------------
GPT-4 Turbo         $10               $30               $0.055
GPT-4o              $5                $15               $0.028
GPT-4o-mini         $0.15             $0.60             $0.001

Typical analysis uses:
- Input: ~1,000 tokens (prompt + data)
- Output: ~1,500 tokens (detailed analysis)

Recommendation:
- Use GPT-4o for production (good balance of quality and cost)
- Use GPT-4o-mini for testing/development (very cheap)
- Use GPT-4 Turbo only if you need maximum quality
"""


# ============================================================================
# Usage Example
# ============================================================================
"""
# In Django shell or view:
from comparator.tennis_analyzer import TennisForehandAnalyzer

analyzer = TennisForehandAnalyzer()

comparison_data = {
    'avg_position_diff': 0.12,
    'avg_angle_diff': 18.5,
    'total_aligned_pairs': 247
}

landmark_stats = {
    'right_shoulder': {'avg_difference': 0.15},
    'right_elbow': {'avg_difference': 0.18},
    'right_wrist': {'avg_difference': 0.10},
    # ... more landmarks
}

result = analyzer.generate_tennis_analysis(comparison_data, landmark_stats)

if result['success']:
    analysis = result['analysis']
    print(f"Skill Level: {analysis['skill_level']}")
    print(f"Assessment: {analysis['overall_assessment']}")
    print(f"Tokens Used: {result['tokens_used']}")
    print(f"Critical Issues: {len(analysis['critical_issues'])}")
"""