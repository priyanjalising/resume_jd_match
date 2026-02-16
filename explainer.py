"""
Match Explainability - Generate "Why This Score?" Reports

BENEFIT: Recruiters understand rankings, build trust in system
IMPACT: Reduction in "why did this person score X?" questions
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class MatchExplanation:
    """Structured explanation of match result"""
    overall_score: float
    recommendation: str
    strengths: List[str]
    concerns: List[str]
    skill_details: Dict
    experience_details: Dict
    suggestions: List[str]
    
    def to_text(self) -> str:
        """Convert to human-readable text"""
        lines = []
        
        # Header
        lines.append("=" * 70)
        lines.append(f"MATCH ANALYSIS - {self.recommendation}")
        lines.append(f"Overall Score: {self.overall_score:.0%}")
        lines.append("=" * 70)
        
        # Strengths
        if self.strengths:
            lines.append("\n✅ STRENGTHS:")
            for strength in self.strengths:
                lines.append(f"  • {strength}")
        
        # Concerns
        if self.concerns:
            lines.append("\n⚠️  CONCERNS:")
            for concern in self.concerns:
                lines.append(f"  • {concern}")
        
        # Skill details
        if self.skill_details:
            lines.append("\n📊 SKILL ANALYSIS:")
            if self.skill_details.get('matched'):
                lines.append(f"  ✓ Matched: {', '.join(self.skill_details['matched'][:5])}")
            if self.skill_details.get('missing'):
                lines.append(f"  ✗ Missing: {', '.join(self.skill_details['missing'][:5])}")
        
        # Suggestions
        if self.suggestions:
            lines.append("\n💡 NEXT STEPS:")
            for suggestion in self.suggestions:
                lines.append(f"  • {suggestion}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


class MatchExplainer:
    """
    Generate explanations for match scores.
    
    USAGE:
        explainer = MatchExplainer()
        explanation = explainer.explain(match_result, job_desc, resume)
        print(explanation.to_text())
    """
    
    def explain(self,
                match_result: Dict,
                job_desc_text: str = None,
                resume_text: str = None) -> MatchExplanation:
        """
        Generate comprehensive explanation.
        
        Args:
            match_result: Result from improved_matcher
            job_desc_text: Optional JD text for context
            resume_text: Optional resume text for context
            
        Returns:
            MatchExplanation object
        """
        overall_score = match_result['final_score']
        section_scores = match_result['section_scores']
        
        # Identify strengths (scores >= 0.7)
        strengths = []
        for section, score in section_scores.items():
            if score >= 0.7:
                strengths.append(self._describe_strength(section, score))
        
        # Identify concerns (scores < 0.7)
        concerns = []
        for section, score in section_scores.items():
            if score < 0.7:
                concerns.append(self._describe_concern(section, score))
        
        # Skill details (if available)
        skill_details = {}
        if 'skill_analysis' in match_result:
            skill_analysis = match_result['skill_analysis']
            skill_details = {
                'matched': list(skill_analysis.get('matched_skills', [])),
                'missing': list(skill_analysis.get('missing_skills', []))
            }
        
        # Experience details (if available)
        experience_details = {}
        if 'experience_analysis' in match_result:
            exp = match_result['experience_analysis']
            experience_details = {
                'candidate_years': exp.get('candidate_years', 0),
                'required_years': exp.get('required', {}).get('min_years', 0),
                'meets_requirements': exp.get('meets_requirements', False)
            }
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            overall_score,
            concerns,
            skill_details,
            experience_details
        )
        
        return MatchExplanation(
            overall_score=overall_score,
            recommendation=match_result['recommendation'],
            strengths=strengths,
            concerns=concerns,
            skill_details=skill_details,
            experience_details=experience_details,
            suggestions=suggestions
        )
    
    def _describe_strength(self, section: str, score: float) -> str:
        """Convert section score to strength description"""
        if score >= 0.9:
            level = "Excellent"
        elif score >= 0.8:
            level = "Very strong"
        else:
            level = "Good"
        
        descriptions = {
            'skills': f"{level} technical skill match ({score:.0%})",
            'experience': f"{level} relevant experience ({score:.0%})",
            'education': f"{level} educational background ({score:.0%})",
            'location': f"{level} location match ({score:.0%})"
        }
        
        return descriptions.get(section, f"{level} {section} match ({score:.0%})")
    
    def _describe_concern(self, section: str, score: float) -> str:
        """Convert section score to concern description"""
        if score >= 0.5:
            level = "Moderate"
        elif score >= 0.3:
            level = "Significant"
        else:
            level = "Major"
        
        descriptions = {
            'skills': f"{level} skill gaps ({score:.0%} match)",
            'experience': f"{level} experience mismatch ({score:.0%})",
            'education': f"{level} education gap ({score:.0%})",
            'location': f"{level} location mismatch ({score:.0%})"
        }
        
        return descriptions.get(section, f"{level} {section} concern ({score:.0%})")
    
    def _generate_suggestions(self,
                             overall_score: float,
                             concerns: List[str],
                             skill_details: Dict,
                             experience_details: Dict) -> List[str]:
        """Generate actionable suggestions"""
        suggestions = []
        
        # Based on overall score
        if overall_score >= 0.8:
            suggestions.append("Schedule interview - strong candidate")
        elif overall_score >= 0.65:
            suggestions.append("Review resume in detail, likely interview")
        elif overall_score >= 0.5:
            suggestions.append("Consider if skill gaps are trainable")
        else:
            suggestions.append("Pass unless desperate for candidates")
        
        # Based on skill gaps
        if skill_details.get('missing'):
            missing_count = len(skill_details['missing'])
            if missing_count <= 2:
                suggestions.append(f"Ask about {missing_count} missing skills in interview")
            else:
                suggestions.append(f"Significant skill gaps ({missing_count} missing)")
        
        # Based on experience
        if experience_details.get('meets_requirements') == False:
            suggestions.append("Under-qualified based on years of experience")
        
        return suggestions


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Demonstrate explainer"""
    
    # Mock match result
    match_result = {
        'final_score': 0.78,
        'recommendation': 'Good Match - Recommend Interview',
        'section_scores': {
            'skills': 0.85,
            'experience': 0.82,
            'education': 0.75,
            'location': 0.50,
            'overview': 0.80
        },
        'skill_analysis': {
            'matched_skills': {'python', 'aws', 'docker', 'sql', 'git'},
            'missing_skills': {'kubernetes', 'graphql'}
        },
        'experience_analysis': {
            'candidate_years': 6,
            'required': {'min_years': 5},
            'meets_requirements': True
        }
    }
    
    # Generate explanation
    explainer = MatchExplainer()
    explanation = explainer.explain(match_result)
    
    # Print
    print(explanation.to_text())


if __name__ == "__main__":
    example_usage()