"""
Tennis-specific analysis using LLM (Claude API) for forehand technique feedback
"""

import anthropic
import os
from typing import Dict, List
import json


class TennisForehandAnalyzer:
    """
    Analyzes tennis forehand technique and provides personalized coaching feedback
    using Claude AI
    """

    def __init__(self, api_key=None):
        """Initialize with Anthropic API key"""
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        self.client = anthropic.Anthropic(api_key=self.api_key)

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
        Generate comprehensive tennis forehand analysis using Claude

        Args:
            comparison_data: Dictionary with comparison metrics
            landmark_stats: Dictionary with per-landmark statistics

        Returns:
            Dictionary with LLM-generated insights and recommendations
        """

        # Prepare structured data for the LLM
        analysis_data = self._prepare_analysis_data(comparison_data, landmark_stats)

        # Generate analysis using Claude
        prompt = self._build_tennis_prompt(analysis_data)

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Parse the response
            analysis_text = response.content[0].text
            parsed_analysis = self._parse_llm_response(analysis_text)

            return {
                'success': True,
                'analysis': parsed_analysis,
                'raw_response': analysis_text
            }

        except Exception as e:
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
        """Build the prompt for Claude"""

        prompt = f"""You are an expert tennis coach analyzing a player's forehand technique compared to a professional reference.

ANALYSIS DATA:
- Overall Similarity Score: {data['overall_metrics']['similarity_score']}%
- Average Position Difference: {data['overall_metrics']['avg_position_diff']:.3f}
- Average Angle Difference: {data['overall_metrics']['avg_angle_diff']:.1f}°

TOP PROBLEM AREAS (ranked by difference from pro):
{self._format_list(data['problem_areas'], lambda x: f"- {x['body_part']}: {x['difference']} ({x['severity']} priority)")}

PLAYER STRENGTHS:
{self._format_list(data['strengths'], lambda x: f"- {x['body_part']}: Excellent form (difference: {x['difference']})")}

SWING PHASE BREAKDOWN:
{self._format_phase_analysis(data['phase_analysis'])}

Please provide a comprehensive tennis coaching analysis with:

1. **Overall Assessment** (2-3 sentences)
   - Skill level indication
   - Main takeaway about their forehand

2. **Critical Issues** (Top 2-3 most important fixes)
   - Specific biomechanical problems
   - How these affect shot quality (power, accuracy, spin, consistency)
   - What might cause injury risk

3. **Technical Recommendations** (3-5 specific drills/corrections)
   - Phase-by-phase breakdown (preparation, backswing, contact, follow-through)
   - Specific body position corrections
   - Common tennis drills to address each issue

4. **Strengths to Maintain** (2-3 points)
   - What they're doing well
   - Why this is important for tennis

5. **Practice Plan** (structured routine)
   - Immediate focus (this week)
   - Medium-term goals (this month)
   - Progression steps

Format your response as JSON with these keys:
{{
  "overall_assessment": "string",
  "skill_level": "beginner|intermediate|advanced",
  "critical_issues": [
    {{
      "issue": "string",
      "impact": "string",
      "injury_risk": "low|medium|high"
    }}
  ],
  "technical_recommendations": [
    {{
      "phase": "string",
      "problem": "string",
      "correction": "string",
      "drill": "string"
    }}
  ],
  "strengths": ["string"],
  "practice_plan": {{
    "immediate": ["string"],
    "medium_term": ["string"],
    "progression": "string"
  }},
  "coaching_notes": "string"
}}

Be specific, actionable, and use tennis-specific terminology. Focus on biomechanics and technique."""

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
            # Try to extract JSON from response
            # Handle cases where LLM adds markdown code blocks
            if '```json' in response_text:
                json_str = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                json_str = response_text.split('```')[1].split('```')[0].strip()
            else:
                json_str = response_text.strip()

            return json.loads(json_str)
        except:
            # If JSON parsing fails, return structured text
            return {
                'overall_assessment': response_text,
                'skill_level': 'unknown',
                'critical_issues': [],
                'technical_recommendations': [],
                'strengths': [],
                'practice_plan': {},
                'coaching_notes': 'Full analysis in overall_assessment'
            }

    def _generate_fallback_analysis(self, data: Dict) -> Dict:
        """Generate basic analysis if LLM fails"""
        score = data['overall_metrics']['similarity_score']

        if score >= 85:
            assessment = "Excellent forehand technique! Your form closely matches professional standards."
        elif score >= 70:
            assessment = "Good forehand foundation with room for refinement in specific areas."
        elif score >= 50:
            assessment = "Developing forehand with several key areas needing attention."
        else:
            assessment = "Significant differences from professional form. Focus on fundamentals."

        return {
            'overall_assessment': assessment,
            'skill_level': 'intermediate' if score >= 70 else 'beginner',
            'critical_issues': [
                {
                    'issue': area['body_part'],
                    'impact': 'Affects consistency and power',
                    'injury_risk': 'medium' if area['severity'] == 'high' else 'low'
                }
                for area in data['problem_areas'][:3]
            ],
            'technical_recommendations': [
                {
                    'phase': 'contact',
                    'problem': 'Body alignment issues',
                    'correction': 'Focus on proper weight transfer and rotation',
                    'drill': 'Shadow swings and ball toss drills'
                }
            ],
            'strengths': [s['body_part'] for s in data['strengths']],
            'practice_plan': {
                'immediate': ['Focus on stance and preparation'],
                'medium_term': ['Build consistent contact point'],
                'progression': 'Progress from stationary to moving ball drills'
            },
            'coaching_notes': 'Consider working with a coach for personalized feedback.'
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