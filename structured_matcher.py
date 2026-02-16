"""
Structured Matching Orchestrator - Detailed Multi-Resume Scoring

PURPOSE:
    Provides a high-level interface for processing batches of resumes 
    against a single job description. It generates granular, section-by-section 
    comparisons rather than just a single "black box" score.

BENEFIT:
    Allows recruiters to see the "why" behind a ranking. For example, 
    identifying a candidate who has a perfect skill match but weak 
    industry experience.

STRATEGY:
    1. Batch Processing: Iterates through a list of raw resume texts.
    2. Object-Oriented Parsing: Converts raw text into StructuredResume 
       and StructuredJobDescription objects via the Parsers module.
    3. Granular Scoring: Leverages the SemanticMatcher to calculate 
       similarities for Skills, Experience, and Education independently.
    4. Rank & Sort: Aggregates results into a sorted list based on 
       the overall match percentage.
"""

import re
import numpy as np
from typing import List, Dict, Optional
from parsers import StructuredJobDescription, StructuredResume
from matcher import SemanticMatcher  # ✅ FIX: Changed from StructuredSemanticMatcher


class EnhancedResumeMatchingSystem:
    """
    Enhanced system with structured section-level analysis
    """
    
    def __init__(self, method='structured_semantic'):
        """
        Initialize enhanced matching system
        
        Args:
            method: 'structured_semantic' (default) or 'semantic' (original)
        """
        self.method = method
        
        
        self.matcher = SemanticMatcher()  # ✅ FIX: Changed from StructuredSemanticMatcher
        
    
    def score_resumes_detailed(self, job_description: str, 
                               resumes: List[str]) -> List[Dict]:
        """
        Score resumes with detailed section-level breakdown
        
        Returns list of detailed results for each resume
        """
        # Parse job description
        job_desc = StructuredJobDescription(job_description)
        
        # Process each resume
        results = []
        for idx, resume_text in enumerate(resumes):
            resume = StructuredResume(resume_text)
            
            # Get detailed match
            match_result = self.matcher.match_structured(job_desc, resume)
            section_scores = match_result.get("section_scores", {})

            # result = {
        #     'resume_id': idx,
        #     'overall_score': round(match_result.get('final_score', 0.0), 4),
        #     'skills_score': round(section_scores.get('skills', 0.0), 4),
        #     'experience_score': round(section_scores.get('experience', 0.0), 4),
        #     'education_score': round(section_scores.get('education', 0.0), 4),
        #     'location_score': round(section_scores.get('location', 0.0), 4),
        #     'weighted_score': round(match_result.get('weighted_score', 0.0), 4),
        #     'recommendation': match_result.get('recommendation', "N/A"),
        #     'rank': None
        # }
            match_result = self.matcher.match_structured(job_desc, resume)

            result = match_result.copy()

            result['resume_id'] = idx
            result['rank'] = None

            results.append(result)
        
        # Sort by overall score and assign ranks
        results_sorted = sorted(
                results, 
                key=lambda x: x['overall_score'], 
                reverse=True)
        
        for rank, result in enumerate(results_sorted, 1):
            result['rank'] = rank
        
        return results_sorted
    
    def get_top_k_detailed(self, job_description: str, resumes: List[str], 
                          k: int = 5) -> List[Dict]:
        """Get top K resumes with detailed breakdown"""
        results = self.score_resumes_detailed(job_description, resumes)
        return results[:k]


def demo_structured_matching():
    """Demonstrate the enhanced structured matching"""
    
    print("="*80)
    print("ENHANCED STRUCTURED RESUME MATCHING - DEMO")
    print("="*80)
    print()
    
    # Job description with clear sections
    job_desc = """
    Senior Machine Learning Engineer
    
    About the Role:
    We're seeking an experienced ML engineer to join our AI team and lead
    the development of production ML systems.
    
    Required Skills:
    - 5+ years of Python programming experience
    - Expert knowledge of TensorFlow or PyTorch
    - Strong experience with cloud platforms (AWS, GCP, or Azure)
    - Experience with MLOps and production deployment
    - Proficiency in SQL and data manipulation
    
    Responsibilities:
    - Design and implement machine learning models for production
    - Deploy and monitor ML models in cloud environments
    - Collaborate with cross-functional teams
    - Mentor junior engineers
    - Optimize model performance and scalability
    
    Qualifications:
    - Master's degree in Computer Science or related field
    - 5+ years of professional experience in software engineering
    - 3+ years of hands-on ML experience
    - Experience with NLP or Computer Vision is a plus
    """
    
    # Sample resumes
    resumes = [
        """
        Jane Doe
        Senior ML Engineer | 7 Years Experience
        
        Summary:
        Accomplished Machine Learning Engineer with 7+ years building production
        ML systems. Deep expertise in Python, TensorFlow, and cloud deployment.
        
        Skills:
        Python, TensorFlow, PyTorch, AWS, GCP, Docker, Kubernetes, SQL, PostgreSQL,
        MLOps, CI/CD, NLP, Computer Vision, REST APIs
        
        Experience:
        Senior ML Engineer at TechCorp (2021-Present)
        - Designed and deployed 15+ ML models to production on AWS
        - Implemented MLOps pipelines reducing deployment time by 60%
        - Led team of 3 junior engineers, mentoring on best practices
        - Built NLP models for text classification and sentiment analysis
        
        ML Engineer at DataSystems (2019-2021)
        - Built recommendation systems serving 10M+ users
        - Optimized model inference reducing latency by 40%
        
        Education:
        MS Computer Science - Stanford University (2019)
        BS Computer Science - UC Berkeley (2017)
        """,
        
        """
        John Smith
        Junior Data Analyst | 2 Years Experience
        
        Profile:
        Enthusiastic data analyst looking to transition into machine learning.
        Strong SQL skills and basic Python knowledge from online courses.
        
        Skills:
        SQL, Excel, Tableau, Python (basic), pandas, scikit-learn (learning)
        
        Experience:
        Data Analyst at BusinessCorp (2023-Present)
        - Created dashboards and reports using Tableau
        - Performed data analysis on business metrics
        - Some exposure to predictive modeling
        
        Education:
        BS Statistics - State University (2022)
        Certificate in Data Science (Online, 2023)
        Currently taking Machine Learning course on Coursera
        """,
        
        """
        Sarah Johnson
        ML Engineer | 5 Years Experience
        
        About:
        Machine learning engineer with 5 years of experience building and
        deploying ML models. Strong cloud deployment expertise.
        
        Technical Skills:
        Python, PyTorch, scikit-learn, AWS, Azure, Docker, Git, SQL, MLflow
        
        Professional Experience:
        ML Engineer at AI Solutions (2020-Present)
        - Developed fraud detection models using PyTorch
        - Deployed models to Azure cloud platform
        - Implemented CI/CD pipelines for ML workflows
        - Collaborated with product teams on ML features
        
        Software Engineer at DataFlow (2019-2020)
        - Built data pipelines with Python and Spark
        - Developed REST APIs for model serving
        
        Education & Certifications:
        MS Data Science - Georgia Tech (2019)
        BS Mathematics - MIT (2017)
        AWS Certified Machine Learning Specialty
        """
    ]
    
    # Initialize enhanced system
    print("Initializing Enhanced Structured Matcher...")
    system = EnhancedResumeMatchingSystem(method='structured_semantic')
    print("✓ System initialized\n")
    
    # Get detailed results
    print("Processing resumes with section-level analysis...")
    results = system.score_resumes_detailed(job_desc, resumes)
    print("✓ Analysis complete\n")
    
    # Display results
    print("DETAILED MATCHING RESULTS")
    print("="*80)
    
    for result in results:
        resume_id = result['resume_id']
        name = resumes[resume_id].split('\n')[1].strip()
        
        print(f"\nRank #{result['rank']}: {name}")
        print("-" * 80)
        print(f"Overall Score:      {result['overall_score']:.4f} ({result['overall_score']:.1%})")
        print(f"Recommendation:     {result['recommendation']}")
        print()
        print("Section Breakdown:")
        print(f"  • Skills Match:       {result['skills_score']:.4f} ({result['skills_score']:.1%})")
        print(f"  • Experience Match:   {result['experience_score']:.4f} ({result['experience_score']:.1%})")
        print(f"  • Education Match:    {result['education_score']:.4f} ({result['education_score']:.1%})")
        print(f"  • Weighted Score:     {result['weighted_score']:.4f} ({result['weighted_score']:.1%})")
    
    print()
    print("="*80)
    print("KEY INSIGHT: Section-level scores show WHERE each candidate is strong/weak")
    print("="*80)


if __name__ == "__main__":
    demo_structured_matching()