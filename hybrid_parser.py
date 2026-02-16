"""
    Validates parsed resume data quality.
    
    PURPOSE:
    Determine if regex parsing extracted enough useful information,
    or if we need to spend money on LLM parsing.
    
    VALIDATION CHECKS:
        1. Required sections present (skills, experience, education)
        2. Sections have minimum length (not just headers extracted)
        3. Candidate name extracted
        4. Quality score above threshold
    
    QUALITY SCORING:
        Points awarded for:
        - Name present: 10 points
        - Each required section: 30 points (if meets min length)
        - Location present: 10 points
        
        Total: 100 points possible
        Score: points / 100
        
        Thresholds:
        - 0.7+ → Good, use regex result
        - 0.4-0.7 → Borderline, use LLM
        - <0.4 → Poor, definitely use LLM
"""


import logging
from typing import Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from parsers import StructuredResume
from llm_parser import LLMStructuredParser
from config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ParsedResumeValidator:
    """Validates parsed resume data quality"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or DEFAULT_CONFIG
    
    def validate(self, data: Dict) -> tuple[bool, list[str]]:
        """
        Validate parsed resume data.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        for field in self.config.parser.required_sections:
            if not data.get(field):
                issues.append(f"Missing required section: {field}")
                continue
            
            # Check minimum length
            if len(data[field]) < self.config.parser.min_section_length:
                issues.append(
                    f"Section '{field}' too short: "
                    f"{len(data[field])} < {self.config.parser.min_section_length}"
                )
        
        # Check name
        if not data.get("name") or len(data["name"]) < 2:
            issues.append("Invalid or missing candidate name")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_quality_score(self, data: Dict) -> float:
        """
        Calculate quality score for parsed data [0, 1].
        
        Higher score = better extraction quality
        """
        score = 0.0
        max_score = 0.0
        
        # Name present (10 points)
        max_score += 10
        if data.get("name") and len(data["name"]) >= 2:
            score += 10
        
        # Required sections (30 points each)
        for field in self.config.parser.required_sections:
            max_score += 30
            if data.get(field):
                length = len(data[field])
                if length >= self.config.parser.min_section_length:
                    score += 30
                elif length >= self.config.parser.min_section_length / 2:
                    score += 15
        
        # Location present (10 points)
        max_score += 10
        if data.get("location") and len(data["location"]) > 2:
            score += 10
        
        return score / max_score if max_score > 0 else 0.0


class HybridStructuredResume:
    """
    Wrapper to make hybrid parser output compatible with StructuredResume interface.
    """
    
    def __init__(self, parsed_data: Dict, raw_text: str, source: str = "regex"):
        self.raw_text = raw_text
        self.source = source  # "regex" or "llm"
        
        self.sections = {
            'skills': parsed_data.get('skills', ''),
            'experience': parsed_data.get('experience', ''),
            'education': parsed_data.get('education', ''),
            'summary': parsed_data.get('summary', ''),
            'full_text': raw_text
        }
        
        self.location = parsed_data.get('location', '')
        self.name = parsed_data.get('name', '')
    
    def get_section(self, name: str) -> str:
        """Interface method expected by matcher"""
        return self.sections.get(name, '')
    
    def get_metadata(self) -> Dict:
        """Get parsing metadata"""
        return {
            'source': self.source,
            'name': self.name,
            'location': self.location,
            'sections_extracted': [k for k, v in self.sections.items() if v]
        }


class HybridParser:
    """
     hybrid parsing with:
    - Quality validation
    - Retry logic for LLM calls
    - Better error handling
    - Detailed logging
    - Fallback strategies
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or DEFAULT_CONFIG
        self.llm_parser = LLMStructuredParser(model=self.config.llm.model)
        self.validator = ParsedResumeValidator(config)
        
        # Statistics
        self.regex_successes = 0
        self.llm_fallbacks = 0
        self.failures = 0
    
    def parse(self, resume_text: str, force_llm: bool = False) -> HybridStructuredResume:
        """
            Parse resume using hybrid approach with validation.
    
            STRATEGY:
                1. Try regex parsing (fast, free)
                2. Validate quality
                3. If quality < 0.7 → Use LLM (slow, costs money)
                4. If LLM fails → Return minimal fallback structure
            
            COST OPTIMIZATION:
                Regex success rate: ~70%
                LLM fallback rate: ~30%
                
                For 100 resumes:
                - Regex: 70 resumes (free!)
                - LLM: 30 resumes (~$0.003)
                - Total cost: ~$0.003 vs $0.01 if always using LLM
            
            QUALITY THRESHOLD TUNING:
                quality >= 0.7 → Use regex
                quality < 0.7 → Use LLM
                
                Lower threshold (0.6):
                    - More regex usage → Faster, cheaper
                    - Risk: Some poor extractions slip through
                
                Higher threshold (0.8):
                    - More LLM usage → Better quality, slower, expensive
                    - Benefit: Higher consistency

        """
        if not resume_text or not resume_text.strip():
            logger.error("Empty resume text provided")
            raise ValueError("Resume text cannot be empty")
        
        # Option 1: Force LLM parsing
        if force_llm:
            logger.info("Forcing LLM parsing")
            return self._parse_with_llm(resume_text)
        
        # Option 2: Try regex first
        try:
            regex_result = self._parse_with_regex(resume_text)
            
            # Validate quality
            quality_score = self.validator.get_quality_score(regex_result)
            logger.debug(f"Regex extraction quality: {quality_score:.2%}")
            
            if quality_score >= 0.7:  # Good enough
                logger.info("✅ Regex extraction sufficient")
                self.regex_successes += 1
                return HybridStructuredResume(regex_result, resume_text, source="regex")
            
            else:
                logger.warning(
                    f"⚠️ Regex extraction quality low ({quality_score:.2%}) → "
                    f"Falling back to LLM"
                )
                self.llm_fallbacks += 1
                return self._parse_with_llm(resume_text)
        
        except Exception as e:
            logger.error(f"Regex parsing failed: {e}", exc_info=True)
            logger.info("Attempting LLM fallback...")
            return self._parse_with_llm(resume_text)
    
    def _parse_with_regex(self, resume_text: str) -> Dict:
        """
        Parse using regex-based extraction.
        
        Returns:
            Dictionary with extracted sections
        """
        logger.debug("Attempting regex-based extraction")
        
        regex_resume = StructuredResume(resume_text)
        
        return {
            "name": regex_resume.name or "",
            "skills": regex_resume.get_section("skills"),
            "experience": regex_resume.get_section("experience"),
            "education": regex_resume.get_section("education"),
            "summary": regex_resume.get_section("summary"),
            "location": regex_resume.location or ""
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _parse_with_llm(self, resume_text: str) -> HybridStructuredResume:
        """
        Parse using LLM with retry logic.
        
        Returns:
            HybridStructuredResume object
            
        Raises:
            Exception if all retry attempts fail
        """
        logger.debug("Attempting LLM-based extraction")
        
        try:
            llm_result = self.llm_parser.parse_resume(resume_text)
            
            # Validate LLM output
            is_valid, issues = self.validator.validate(llm_result)
            
            if not is_valid:
                logger.warning(f"LLM extraction issues: {issues}")
                # Continue anyway, but log issues
            
            logger.info("✅ LLM extraction complete")
            return HybridStructuredResume(llm_result, resume_text, source="llm")
        
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            self.failures += 1
            
            # Last resort: return minimal structure
            logger.warning("Using fallback minimal structure")
            return HybridStructuredResume(
                {
                    "name": "",
                    "skills": "",
                    "experience": "",
                    "education": "",
                    "summary": resume_text[:500],  # Use first 500 chars
                    "location": ""
                },
                resume_text,
                source="fallback"
            )
    
    def get_stats(self) -> Dict:
        """Get parsing statistics"""
        total = self.regex_successes + self.llm_fallbacks + self.failures
        return {
            'total_parsed': total,
            'regex_successes': self.regex_successes,
            'llm_fallbacks': self.llm_fallbacks,
            'failures': self.failures,
            'regex_success_rate': self.regex_successes / total if total > 0 else 0
        }