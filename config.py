"""
Centralized Configuration System for Resume Matching

PURPOSE:
    Single source of truth for all system parameters. No more hunting
    for hard-coded values scattered throughout the codebase.

QUICK START:
    # Use defaults
    config = Config()
    
    # Customize
    config.weights.skills = 0.40
    engine = ImprovedMatchingEngine(config=config)

KEY BENEFIT:
    Change one number here instead of searching through 10 files.
"""

from dataclasses import dataclass
from typing import Dict
import os


@dataclass
class ModelConfig:
    """Embedding model configuration"""
    name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"  # or "cuda" for GPU
    batch_size: int = 32


"""
IMPORTANT: All weights must sum to 1.0 (100%)

CUSTOMIZE BY ROLE TYPE:
- Technical role: skills=0.40, experience=0.35
- Management role: experience=0.40, skills=0.30
- Entry-level: education=0.30, skills=0.30
- Remote role: location=0.05 (matters less)
"""
@dataclass
class ScoringWeights:
    """Section weights for scoring"""
    skills: float = 0.35
    experience: float = 0.30
    education: float = 0.15
    overview: float = 0.10
    location: float = 0.10
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = sum([self.skills, self.experience, self.education, 
                     self.overview, self.location])
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'skills': self.skills,
            'experience': self.experience,
            'education': self.education,
            'overview': self.overview,
            'location': self.location
        }

"""
IMPORTANT: All weights must sum to 1.0 (100%)

CUSTOMIZE BY ROLE TYPE:
- Technical role: skills=0.40, experience=0.35
- Management role: experience=0.40, skills=0.30
- Entry-level: education=0.30, skills=0.30
- Remote role: location=0.05 (matters less)
"""
@dataclass
class ThresholdConfig:
    """Score thresholds for recommendations"""
    strong_match: float = 0.8
    good_match: float = 0.65
    potential_match: float = 0.5
    weak_match: float = 0.35
    
    def get_recommendation(self, score: float) -> str:
        """Get recommendation label for score"""
        if score >= self.strong_match:
            return "Strong Match - Highly Recommend"
        elif score >= self.good_match:
            return "Good Match - Recommend Interview"
        elif score >= self.potential_match:
            return "Potential Match - Review Carefully"
        elif score >= self.weak_match:
            return "Weak Match - Consider If Desperate"
        else:
            return "Poor Match - Pass"


@dataclass
class PathConfig:
    """File paths configuration"""
    data_dir: str = "data"
    resume_dir: str = "data/resume"
    job_desc_file: str = "data/job_description.txt"
    output_dir: str = "output"
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class LLMConfig:
    """LLM parser configuration"""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 3
    timeout: int = 30


@dataclass
class ParserConfig:
    """Parser quality thresholds"""
    min_section_length: int = 30
    required_sections: list = None
    
    def __post_init__(self):
        if self.required_sections is None:
            self.required_sections = ["skills", "experience", "education"]


class Config:
    """Main configuration container"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.weights = ScoringWeights()
        self.thresholds = ThresholdConfig()
        self.paths = PathConfig()
        self.llm = LLMConfig()
        self.parser = ParserConfig()
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Config':
        """Load configuration from YAML/JSON file"""
        # TODO: Implement file-based config loading
        raise NotImplementedError("File-based config not yet implemented")


# Global default config instance
DEFAULT_CONFIG = Config()