"""
Years of Experience Extraction and Matching

BENEFIT: Filter out under/over-qualified candidates
IMPACT: Reduction in mismatched interviews
"""

import re
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ExperienceRequirement:
    """Job description experience requirements"""
    min_years: Optional[float] = None
    max_years: Optional[float] = None
    required: bool = False
    
    def __str__(self):
        if self.min_years and self.max_years:
            return f"{self.min_years}-{self.max_years} years"
        elif self.min_years:
            return f"{self.min_years}+ years"
        else:
            return "Not specified"


class ExperienceExtractor:
    """
    Extract years of experience from job descriptions and resumes.
    
    APPROACHES:
    1. JD: Parse requirement statements ("5+ years")
    2. Resume: Calculate from date ranges or explicit statements
    """
    
    # Patterns for JD requirements
    REQUIREMENT_PATTERNS = [
        r'(\d+)\s*\+\s*years?',              # "5+ years"
        r'(\d+)\s*or more years?',            # "5 or more years"
        r'minimum\s+(\d+)\s*years?',          # "minimum 5 years"
        r'at least\s+(\d+)\s*years?',         # "at least 5 years"
        r'(\d+)-(\d+)\s*years?',              # "3-5 years"
        r'(\d+)\s*to\s*(\d+)\s*years?',       # "3 to 5 years"
    ]
    
    def extract_from_jd(self, jd_text: str) -> ExperienceRequirement:
        """
        Extract experience requirements from job description.
        
        Args:
            jd_text: Job description text
            
        Returns:
            ExperienceRequirement object
            
        Examples:
            "5+ years of Python" → min_years=5, max_years=None
            "3-5 years required" → min_years=3, max_years=5
            "Senior level" → min_years=None (can't determine)
        """
        text_lower = jd_text.lower()
        
        for pattern in self.REQUIREMENT_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                groups = match.groups()
                
                if len(groups) == 1:
                    # Single number ("5+ years")
                    min_years = float(groups[0])
                    return ExperienceRequirement(
                        min_years=min_years,
                        max_years=None,
                        required=True
                    )
                elif len(groups) == 2:
                    # Range ("3-5 years")
                    min_years = float(groups[0])
                    max_years = float(groups[1])
                    return ExperienceRequirement(
                        min_years=min_years,
                        max_years=max_years,
                        required=True
                    )
        
        # Check for seniority level as fallback
        if any(level in text_lower for level in ['senior', 'lead', 'principal']):
            return ExperienceRequirement(min_years=5, required=False)
        elif any(level in text_lower for level in ['mid-level', 'intermediate']):
            return ExperienceRequirement(min_years=3, required=False)
        elif any(level in text_lower for level in ['junior', 'entry']):
            return ExperienceRequirement(min_years=0, max_years=2, required=False)
        
        # No requirements found
        return ExperienceRequirement()
    
    def extract_from_resume(self, resume_text: str) -> float:
        """
        Calculate total years of experience from resume.
        
        APPROACHES (in order of priority):
        1. Explicit statement: "7 years of experience"
        2. Date range calculation: "2018-2023" → 5 years
        3. Graduation year estimation
        
        Args:
            resume_text: Resume text
            
        Returns:
            Total years of experience
        """
        # Approach 1: Explicit statement
        explicit_exp = self._extract_explicit_experience(resume_text)
        if explicit_exp:
            return explicit_exp
        
        # Approach 2: Calculate from date ranges
        calculated_exp = self._calculate_from_dates(resume_text)
        if calculated_exp:
            return calculated_exp
        
        # Approach 3: Estimate from graduation
        graduation_exp = self._estimate_from_graduation(resume_text)
        return graduation_exp
    
    def _extract_explicit_experience(self, text: str) -> Optional[float]:
        """
        Find explicit experience statements.
        
        Patterns:
        - "7 years of experience"
        - "7+ years in software engineering"
        """
        patterns = [
            r'(\d+)\+?\s*years?\s*of\s*experience',
            r'(\d+)\+?\s*years?\s*in',
            r'experience[:\s]+(\d+)\+?\s*years?'
        ]
        
        text_lower = text.lower()
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return float(match.group(1))
        
        return None
    
    def _calculate_from_dates(self, text: str) -> Optional[float]:
        """
        Calculate experience from date ranges.
        
        Patterns:
        - "2018-2023" → 5 years
        - "Jan 2020 - Dec 2023" → 4 years
        - "2020-Present" → current_year - 2020
        """
        # Pattern: YYYY-YYYY or YYYY-Present
        date_ranges = re.findall(
            r'(\d{4})\s*[-–]\s*(\d{4}|present|current)',
            text,
            re.IGNORECASE
        )
        
        if not date_ranges:
            return None
        
        total_years = 0
        current_year = datetime.now().year
        
        for start, end in date_ranges:
            start_year = int(start)
            
            if end.lower() in ['present', 'current']:
                end_year = current_year
            else:
                end_year = int(end)
            
            # Calculate duration
            duration = end_year - start_year
            
            # Sanity check (no more than 50 years)
            if 0 <= duration <= 50:
                total_years += duration
        
        return total_years if total_years > 0 else None
    
    def _estimate_from_graduation(self, text: str) -> float:
        """
        Estimate experience from graduation year.
        
        Heuristic: current_year - graduation_year - 1
        (Subtract 1 to account for year of graduation)
        """
        # Find graduation years
        graduation_patterns = [
            r'graduated[:\s]+(\d{4})',
            r'(?:BS|BA|MS|MA|PhD)[,\s]+(\d{4})',
            r'university[,\s]+(\d{4})'
        ]
        
        text_lower = text.lower()
        years = []
        
        for pattern in graduation_patterns:
            matches = re.findall(pattern, text_lower)
            years.extend(int(y) for y in matches if 1970 <= int(y) <= datetime.now().year)
        
        if years:
            graduation_year = max(years)  # Most recent degree
            experience = datetime.now().year - graduation_year - 1
            return max(0, experience)  # Can't be negative
        
        return 0.0  # Unknown


class ExperienceMatcher:
    """
    Match experience requirements with candidate experience.
    """
    
    def __init__(self):
        self.extractor = ExperienceExtractor()
    
    def match(self, jd_text: str, resume_text: str) -> Dict:
        """
        Match experience requirements.
        
        Args:
            jd_text: Job description
            resume_text: Resume
            
        Returns:
            {
                'required': ExperienceRequirement,
                'candidate': float,
                'score': float (0-1),
                'meets_requirements': bool,
                'gap': float (negative if under-qualified)
            }
        """
        # Extract requirements and experience
        required = self.extractor.extract_from_jd(jd_text)
        candidate_years = self.extractor.extract_from_resume(resume_text)
        
        # Calculate score
        score = self._calculate_score(required, candidate_years)
        
        # Determine if requirements met
        meets_requirements = False
        gap = 0.0
        
        if required.min_years:
            meets_requirements = candidate_years >= required.min_years
            gap = candidate_years - required.min_years
        else:
            meets_requirements = True  # No requirements specified
        
        return {
            'required': required,
            'candidate_years': candidate_years,
            'score': score,
            'meets_requirements': meets_requirements,
            'gap': gap
        }
    
    def _calculate_score(self, 
                        required: ExperienceRequirement,
                        candidate_years: float) -> float:
        """
        Score experience match (0-1).
        
        Scoring logic:
            candidate >= required → 1.0 (meets/exceeds)
            candidate >= 0.8 * required → 0.8 (close)
            candidate >= 0.5 * required → 0.5 (partial)
            candidate < 0.5 * required → 0.0-0.5 (too junior)
        
        If no requirements, return 0.5 (neutral)
        """
        if not required.min_years:
            return 0.5  # Neutral if no requirements
        
        # Check if meets minimum
        if candidate_years >= required.min_years:
            # Check if exceeds maximum (over-qualified)
            if required.max_years and candidate_years > required.max_years * 1.5:
                return 0.8  # Slightly penalize over-qualification
            return 1.0  # Perfect match
        
        # Under-qualified: score based on ratio
        ratio = candidate_years / required.min_years
        
        if ratio >= 0.8:
            return 0.8  # Close
        elif ratio >= 0.5:
            return 0.5  # Partial
        else:
            return ratio  # Proportional (0.0-0.5)
    
    def get_recommendation(self, match_result: Dict) -> str:
        """
        Generate recommendation based on experience match.
        
        Returns human-readable recommendation.
        """
        required = match_result['required']
        candidate = match_result['candidate_years']
        score = match_result['score']
        
        if score >= 0.9:
            return f"✅ Excellent experience match ({candidate} years, required: {required})"
        elif score >= 0.7:
            return f"✓ Good experience ({candidate} years, required: {required})"
        elif score >= 0.5:
            return f"⚠️ Borderline ({candidate} years, required: {required})"
        else:
            return f"❌ Insufficient experience ({candidate} years, required: {required})"


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Demonstrate experience matching"""
    
    jd = """
    Senior Backend Engineer
    
    Requirements:
    - 5+ years of backend development
    - Experience with microservices
    - Strong Python skills
    """
    
    resume = """
    John Doe
    Senior Software Engineer
    
    Experience:
    Software Engineer at Google (2018-2023) - 5 years
    - Built microservices with Python
    - Led team of 4 engineers
    
    Junior Developer at Startup (2016-2018) - 2 years
    - Full-stack development
    
    Education:
    BS Computer Science, Stanford University (2016)
    
    Total: 7 years of professional experience
    """
    
    matcher = ExperienceMatcher()
    result = matcher.match(jd, resume)
    
    print("Experience Match Analysis")
    print("=" * 50)
    print(f"Required: {result['required']}")
    print(f"Candidate: {result['candidate_years']} years")
    print(f"Score: {result['score']:.0%}")
    print(f"Meets Requirements: {result['meets_requirements']}")
    print(f"Gap: {result['gap']:+.1f} years")
    print()
    print(matcher.get_recommendation(result))


if __name__ == "__main__":
    example_usage()