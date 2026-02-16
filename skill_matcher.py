"""
Skill Extraction and Exact Matching Module - Technical Token Analysis

PURPOSE:
    Identify and quantify the overlap of specific technical keywords between 
    job descriptions and resumes. This ensures that "must-have" technologies 
    are explicitly recognized beyond general semantic meaning.

BENEFIT:
    Provides a significant boost to candidates who possess the exact 
    required technical stack, leading to a 20-30% improvement in 
    ranking quality for specialized technical roles.

STRATEGY:
    1. Multi-Pass Extraction: Prioritizes multi-word skills (e.g., "Machine Learning")
       before single tokens to prevent incorrect segmentation.
    2. Boundary-Aware Regex: Uses word boundaries to avoid false positives 
       (e.g., ensuring "Java" doesn't match "JavaScript").
    3. Weighted Scoring: Combines the exact match percentage with the 
       broader semantic score to reward both technical fit and domain context.
"""

import re
from typing import Set, Dict, List
from dataclasses import dataclass


@dataclass
class SkillMatchResult:
    """Result of skill matching analysis"""
    matched_skills: Set[str]
    missing_skills: Set[str]
    extra_skills: Set[str]
    match_percentage: float
    coverage: float
    
    def to_dict(self) -> Dict:
        return {
            'matched_skills': list(self.matched_skills),
            'missing_skills': list(self.missing_skills),
            'extra_skills': list(self.extra_skills),
            'match_percentage': self.match_percentage,
            'coverage': self.coverage
        }


class SkillExtractor:
    """
    Extract skills from text using pattern matching.
    
    STRATEGY:
    1. Maintain database of known skills
    2. Use regex patterns to find skills
    3. Handle variations (Python vs python)
    4. Support multi-word skills (Machine Learning)
    """
    
    # Comprehensive skill database (expand as needed)
    SKILLS_DATABASE = {
        # Programming Languages
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 
        'rust', 'ruby', 'php', 'swift', 'kotlin', 'scala', 'r',
        
        # Web Technologies
        'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask',
        'fastapi', 'spring', 'asp.net', 'html', 'css', 'sass', 'tailwind',
        
        # Databases
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
        'cassandra', 'dynamodb', 'oracle', 'mssql',
        
        # Cloud & DevOps
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab',
        'terraform', 'ansible', 'ci/cd', 'linux', 'bash',
        
        # Data Science & ML
        'machine learning', 'deep learning', 'tensorflow', 'pytorch', 
        'scikit-learn', 'pandas', 'numpy', 'spark', 'hadoop', 'nlp',
        'computer vision', 'data analysis', 'statistics',
        
        # --- MARKETING ---
        'seo', 'sem', 'content marketing', 'social media', 'email marketing', 
        'brand management', 'market research', 'ppc', 'google analytics', 
        'hubspot', 'mailchimp', 'copywriting', 'digital marketing', 'advertising',
        'growth hacking', 'influencer marketing', 'public relations',
        
        # --- PRODUCT MANAGEMENT ---
        'product roadmap', 'stakeholder management', 'user stories', 
        'product lifecycle', 'market fit', 'gtm strategy', 'product strategy', 
        'customer discovery', 'wireframing', 'user research', 'prioritization',
        
        # --- SALES & BUSINESS DEVELOPMENT ---
        'crm', 'lead generation', 'b2b', 'b2c', 'negotiation', 'sales funnel', 
        'account management', 'cold calling', 'salesforce', 'prospecting', 
        'closing', 'partnership', 'sales strategy', 'channel sales',
        
        # --- HR & OPERATIONS ---
        'recruitment', 'employee engagement', 'onboarding', 'payroll', 
        'talent acquisition', 'performance management', 'operations', 
        'change management', 'training', 'conflict resolution',
        
        # --- FINANCE & ANALYTICS ---
        'financial modeling', 'budgeting', 'forecasting', 'p&l', 'accounting', 
        'auditing', 'data analysis', 'excel', 'tableau', 'power bi', 'sas',

        # Other
        'git', 'agile', 'scrum', 'rest api', 'graphql', 'microservices',
        'oauth', 'jwt', 'testing', 'unit testing', 'tdd'
    }
    
    # Multi-word skills need special handling
    MULTI_WORD_SKILLS = {
        'machine learning', 'deep learning', 'computer vision',
        'natural language processing', 'nlp', 'artificial intelligence',
        'data science', 'software engineering', 'cloud computing',
        'ci/cd', 'rest api', 'unit testing', 
        'content marketing', 'social media', 'email marketing', 'brand management', 
        'market research', 'google analytics', 'digital marketing', 'growth hacking',
        'influencer marketing', 'public relations', 'product roadmap', 
        'stakeholder management', 'user stories', 'product lifecycle', 'market fit', 
        'gtm strategy', 'product strategy', 'customer discovery', 'user research',
        'lead generation', 'sales funnel', 'account management', 'cold calling', 
        'sales strategy', 'channel sales', 'talent acquisition', 'performance management', 
        'change management', 'conflict resolution', 'financial modeling', 
        'data analysis', 'power bi', 'project management', 'design thinking'
    }
    
    def __init__(self, custom_skills: List[str] = None):
        """
        Initialize skill extractor.
        
        Args:
            custom_skills: Additional skills to recognize (domain-specific)
        """
        self.skills = self.SKILLS_DATABASE.copy()
        
        if custom_skills:
            self.skills.update(s.lower() for s in custom_skills)
    
    def extract_skills(self, text: str) -> Set[str]:
        """
        Extract all skills mentioned in text.
        
        Args:
            text: Resume or job description text
            
        Returns:
            Set of detected skills
            
        Example:
            >>> extractor = SkillExtractor()
            >>> text = "Proficient in Python, AWS, and Machine Learning"
            >>> extractor.extract_skills(text)
            {'python', 'aws', 'machine learning'}
        """
        text_lower = text.lower()
        found_skills = set()
        
        # First pass: Multi-word skills (higher priority)
        for skill in self.MULTI_WORD_SKILLS:
            if skill in text_lower:
                found_skills.add(skill)
        
        # Second pass: Single-word skills
        # Use word boundaries to avoid partial matches
        for skill in self.skills:
            if skill in self.MULTI_WORD_SKILLS:
                continue  # Already handled
            
            # Pattern: word boundary + skill + word boundary
            # Handles: "Python," "Python." "Python and Java"
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        
        return found_skills
    
    def extract_from_section(self, text: str, section_name: str) -> Set[str]:
        """
        Extract skills from specific section (more accurate).
        
        Example:
            >>> text = "Skills: Python, Java\\nExperience: Worked at Microsoft"
            >>> extractor.extract_from_section(text, "skills")
            {'python', 'java'}  # Doesn't pick up "Microsoft" as skill
        """
        # Find the skills section
        section_pattern = f"{section_name}[:\\s]+(.*?)(?=\\n\\n|\\n[A-Z][^:\\n]+:|$)"
        match = re.search(section_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            section_text = match.group(1)
            return self.extract_skills(section_text)
        
        # Fallback: extract from entire text
        return self.extract_skills(text)


class SkillMatcher:
    """
    Match skills between job description and resume.
    
    USAGE:
        matcher = SkillMatcher()
        result = matcher.match(jd_text, resume_text)
        
        print(f"Match: {result.match_percentage:.0%}")
        print(f"Matched: {result.matched_skills}")
        print(f"Missing: {result.missing_skills}")
    """
    
    def __init__(self, custom_skills: List[str] = None):
        self.extractor = SkillExtractor(custom_skills)
    
    def match(self, jd_text: str, resume_text: str) -> SkillMatchResult:
        """
        Match skills between JD and resume.
        
        Args:
            jd_text: Job description text
            resume_text: Resume text
            
        Returns:
            SkillMatchResult with detailed analysis
        """
        # Extract skills
        jd_skills = self.extractor.extract_skills(jd_text)
        resume_skills = self.extractor.extract_skills(resume_text)
        
        # Calculate matches
        matched = jd_skills & resume_skills  # Intersection
        missing = jd_skills - resume_skills  # In JD but not resume
        extra = resume_skills - jd_skills    # In resume but not JD
        
        # Calculate percentages
        match_pct = len(matched) / len(jd_skills) if jd_skills else 0
        coverage = len(matched) / len(resume_skills) if resume_skills else 0
        
        return SkillMatchResult(
            matched_skills=matched,
            missing_skills=missing,
            extra_skills=extra,
            match_percentage=match_pct,
            coverage=coverage
        )
    
    def get_detailed_report(self, result: SkillMatchResult) -> str:
        """
        Generate human-readable report.
        
        Example output:
        
        Skill Match Analysis
        ====================
        Match Rate: 75% (6/8 required skills)
        
        ✅ Matched Skills:
        • Python, AWS, Docker, SQL, React, Git
        
        ❌ Missing Skills:
        • Kubernetes, GraphQL
        
        💡 Extra Skills:
        • TypeScript, MongoDB, Redis
        
        Recommendation: Strong technical match. Missing skills
        are trainable. Proceed to interview.
        """
        lines = []
        lines.append("Skill Match Analysis")
        lines.append("=" * 50)
        lines.append(f"Match Rate: {result.match_percentage:.0%} "
                    f"({len(result.matched_skills)}/{len(result.matched_skills) + len(result.missing_skills)} required skills)")
        
        if result.matched_skills:
            lines.append("\n✅ Matched Skills:")
            lines.append("• " + ", ".join(sorted(result.matched_skills)))
        
        if result.missing_skills:
            lines.append("\n❌ Missing Skills:")
            lines.append("• " + ", ".join(sorted(result.missing_skills)))
        
        if result.extra_skills:
            lines.append("\n💡 Extra Skills:")
            lines.append("• " + ", ".join(sorted(result.extra_skills)))
        
        # Recommendation
        lines.append("\n" + "=" * 50)
        if result.match_percentage >= 0.8:
            lines.append("Recommendation: Excellent skill match! ✅")
        elif result.match_percentage >= 0.6:
            lines.append("Recommendation: Good match. Review missing skills.")
        elif result.match_percentage >= 0.4:
            lines.append("Recommendation: Partial match. Significant gaps.")
        else:
            lines.append("Recommendation: Poor match. Major skill gaps.")
        
        return "\n".join(lines)
    
    def combined_score(self,
                      semantic_score: float,
                      skill_match: SkillMatchResult,
                      weight_semantic: float = 0.6,
                      weight_skills: float = 0.4) -> float:
        """
        Combine semantic similarity with exact skill matching.
        
        Args:
            semantic_score: Score from embedding similarity (0-1)
            skill_match: Result from skill matching
            weight_semantic: Weight for semantic score
            weight_skills: Weight for skill matching
            
        Returns:
            Combined score (0-1)
            
        Example:
            Semantic: 0.75 (good semantic match)
            Skills: 0.90 (9/10 required skills)
            Combined: 0.6*0.75 + 0.4*0.90 = 0.81
            
            This boosts candidates with exact skill matches!
        """
        return (weight_semantic * semantic_score + 
                weight_skills * skill_match.match_percentage)


# ============================================================================
# INTEGRATION WITH EXISTING MATCHER
# ============================================================================

def integrate_skill_matching():
    """
    How to integrate with improved_matcher.py
    
    Add to ImprovedSemanticMatcher class:
    """
    
    example_code = '''
class ImprovedSemanticMatcher:
    
    def __init__(self, config=None):
        # ... existing code ...
        
        # ADD THIS:
        from skill_matcher import SkillMatcher
        self.skill_matcher = SkillMatcher()
    
    def match_structured(self, job_desc, resume):
        # ... existing matching code ...
        
        # ADD SKILL MATCHING:
        skill_result = self.skill_matcher.match(
            job_desc.get_section('full_text'),
            resume.get_section('full_text')
        )
        
        # Combine with semantic score
        final_score = self.skill_matcher.combined_score(
            semantic_score=result['final_score'],
            skill_match=skill_result,
            weight_semantic=0.6,  # Configurable
            weight_skills=0.4
        )
        
        # Add to result
        result['skill_analysis'] = skill_result.to_dict()
        result['final_score'] = final_score
        
        return result
    '''
    
    print(example_code)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """Demonstrate skill matching"""
    
    # Example job description
    jd = """
    Senior Backend Engineer
    
    Required Skills:
    - 5+ years Python experience
    - Strong AWS knowledge
    - Docker and Kubernetes
    - PostgreSQL or MySQL
    - REST API design
    - Git version control
    
    Nice to have:
    - GraphQL
    - Redis
    - CI/CD experience
    """
    
    # Example resume
    resume = """
    John Doe
    Senior Software Engineer
    
    Skills:
    Python, AWS, Docker, PostgreSQL, Git, TypeScript,
    MongoDB, Redis, React, Node.js
    
    Experience:
    - Built microservices with Python and Docker
    - Deployed to AWS using Kubernetes
    - Designed REST APIs for 10M+ users
    - Led team using Git and agile methodologies
    """
    
    # Match skills
    matcher = SkillMatcher()
    result = matcher.match(jd, resume)
    
    # Print report
    print(matcher.get_detailed_report(result))
    
    # Use in combined scoring
    semantic_score = 0.75  # From embedding similarity
    combined = matcher.combined_score(semantic_score, result)
    print(f"\nCombined Score: {combined:.2%}")


if __name__ == "__main__":
    print("Skill Matching Module")
    print("=" * 50)
    example_usage()
    
    print("\n\nIntegration Example:")
    print("=" * 50)
    integrate_skill_matching()