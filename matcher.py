"""
Semantic Matching Engine - Compares Meaning, Not Just Keywords

HOW IT WORKS:
1. Converts text to vectors (numbers that represent meaning)
2. Similar meanings → Similar vectors
3. Compares vectors using cosine similarity
4. Combines scores with configured weights

"""

import re
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from config import Config, DEFAULT_CONFIG
from parsers import StructuredJobDescription, StructuredResume
from skill_matcher import SkillMatcher
from experience_matcher import ExperienceMatcher
from explainer import MatchExplainer



# Configure logger
logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for text embeddings to avoid recomputation"""
    
    def __init__(self, maxsize: int = 1000):
        self._cache: Dict[str, np.ndarray] = {}
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Retrieve from cache if available.

            WHY: Encoding is expensive (~100ms). Cache saves huge time.
            
            Example impact (100 resumes):
                Without cache: Job description encoded 100 times = 10s wasted
                With cache: Job description encoded 1 time, 99 hits = 9.9s saved
        """
        result = self._cache.get(text)
        if result is not None:
            self.hits += 1
        else:
            self.misses += 1
        return result
    
    def set(self, text: str, embedding: np.ndarray):
        """Store embedding in cache"""
        if len(self._cache) >= self.maxsize:
            # Remove oldest entry (simple FIFO)
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = embedding
    
    def clear(self):
        """Clear cache"""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'size': len(self._cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


class SemanticMatcher:
    """
    Enhanced semantic matcher with:
    - Embedding caching
    - Proper logging
    - Error handling
    - Batch processing
    - Configurable weights
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize matcher with configuration.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or DEFAULT_CONFIG
        self.skill_matcher = SkillMatcher()
        self.experience_matcher = ExperienceMatcher()
        self.explainer = MatchExplainer()
        
        # Load model
        logger.info(f"Loading embedding model: {self.config.model.name}")
        try:
            self.model = SentenceTransformer(
                self.config.model.name,
                device=self.config.model.device
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Initialize cache
        self.cache = EmbeddingCache()
        
        # Store weights
        self.weights = self.config.weights.to_dict()
        
        # Statistics
        self.matches_processed = 0
    
    # ----------------------------------------------------------
    # TEXT PREPROCESSING
    # ----------------------------------------------------------
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def preprocess_text(text: str) -> str:
        """
        Clean text before embedding.
    
        WHY: Consistency + Better embeddings
        - "Python" vs "python" → Same result after preprocessing
        - Remove noise that doesn't add meaning
        
        Example: "Senior ML Engineer!!!" → "senior ml engineer"
        """
        if not text:
            return ""
        
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-z0-9\s\.\,\-\+\#]', ' ', text)
        return text.strip()
    
    # ----------------------------------------------------------
    # EMBEDDING WITH CACHING
    # ----------------------------------------------------------
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text with caching.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        # Check cache first
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        
        # Encode and cache
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            self.cache.set(text, embedding)
            return embedding
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            # Return zero vector as fallback
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Batch encode multiple texts efficiently.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings
        """
        # Check which texts are cached
        embeddings = []
        texts_to_encode = []
        indices_to_encode = []
        
        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                embeddings.append((i, cached))
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        # Encode uncached texts in batch
        if texts_to_encode:
            try:
                new_embeddings = self.model.encode(
                    texts_to_encode,
                    batch_size=self.config.model.batch_size,
                    show_progress_bar=False
                )
                
                # Cache new embeddings
                for text, emb in zip(texts_to_encode, new_embeddings):
                    self.cache.set(text, emb)
                
                # Add to results
                for idx, emb in zip(indices_to_encode, new_embeddings):
                    embeddings.append((idx, emb))
            
            except Exception as e:
                logger.error(f"Batch encoding failed: {e}")
                # Return zero vectors
                dim = self.model.get_sentence_embedding_dimension()
                for idx in indices_to_encode:
                    embeddings.append((idx, np.zeros(dim)))
        
        # Sort by original indices and extract vectors
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
    
    # ----------------------------------------------------------
    # SIMILARITY COMPUTATION
    # ----------------------------------------------------------
    
    @staticmethod
    def normalized_cosine(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute normalized cosine similarity [0, 1].
        
        Args:
            vec1, vec2: Embedding vectors
            
        Returns:
            Similarity score in range [0, 1]
        """
        try:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float((similarity + 1) / 2)
        
        except Exception as e:
            logger.warning(f"Similarity computation failed: {e}")
            return 0.0
    
    # ----------------------------------------------------------
    # LOCATION SCORING
    # ----------------------------------------------------------
    
    def location_score(self, 
                      job_loc: Optional[str], 
                      resume_loc: Optional[str]) -> float:
        """
        Compute location match score with detailed logging.
        
        Args:
            job_loc: Job location requirement
            resume_loc: Candidate location
            
        Returns:
            Location match score [0, 1]
        """
        # No location requirement
        if not job_loc:
            logger.debug("No job location requirement → neutral score")
            return 0.5
        
        # No candidate location
        if not resume_loc:
            logger.debug("No candidate location → zero score")
            return 0.0
        
        job_loc = job_loc.lower()
        resume_loc = resume_loc.lower()
        
        # Remote position
        if "remote" in job_loc:
            logger.debug("Remote position → full score")
            return 1.0
        
        # Exact match
        if job_loc in resume_loc:
            logger.debug(f"Exact location match: {job_loc}")
            return 1.0
        
        # Partial match (city or state)
        for part in job_loc.split(","):
            if part.strip() in resume_loc:
                logger.debug(f"Partial location match: {part}")
                return 0.75
        
        logger.debug(f"No location match: {job_loc} vs {resume_loc}")
        return 0.0
    
    # ----------------------------------------------------------
    # MAIN MATCHING FUNCTION
    # ----------------------------------------------------------
    
    def match_structured(self,
                        job_desc: StructuredJobDescription,
                        resume: StructuredResume) -> Dict:
        """
        Perform structured semantic matching with detailed breakdown.
        
        Args:
            job_desc: Parsed job description
            resume: Parsed resume
            
        Returns:
            Dictionary with scores and recommendation
        """
        try:
            logger.debug(f"Matching resume: {resume.name}")
            
            # Extract and preprocess sections
            job_sections = self._extract_job_sections(job_desc)
            resume_sections = self._extract_resume_sections(resume)
            
            # Compute section similarities (SEMANTIC SCORES)
            section_scores = self._compute_section_scores(
                job_sections, 
                resume_sections
            )
            
            # Add location score
            loc_score = self.location_score(job_desc.location, resume.location)
            section_scores['location'] = loc_score
            
            # Compute weighted semantic score
            weighted_score = self._compute_weighted_score(section_scores)
            
            # Get overall semantic score
            overall_score = section_scores.get('full_text', 0.0)
            
            # ⚠️ FIX: Calculate base semantic score BEFORE using it
            # Original formula: 70% weighted + 30% overall
            base_semantic_score = (weighted_score * 0.7 + overall_score * 0.3)
            
            # Add skill matching
            skill_result = self.skill_matcher.match(
                job_desc.get_section('full_text'),
                resume.get_section('full_text')
            )
            
            # Add experience matching
            exp_result = self.experience_matcher.match(
                job_desc.get_section('full_text'),
                resume.get_section('full_text')
            )
            
            # ✅ FIX: Now combine scores properly
            # Blend: 50% semantic + 30% skills + 20% experience
            final_score = (
                0.5 * base_semantic_score +
                0.3 * skill_result.match_percentage +
                0.2 * exp_result['score']
            )
            
            # Get recommendation
            recommendation = self.config.thresholds.get_recommendation(final_score)
            
            # Update statistics
            self.matches_processed += 1
            
            logger.debug(f"Match complete: {final_score:.2%} - {recommendation}")
            
            return {
                "section_scores": section_scores,
                "weighted_score": weighted_score,
                "overall_score": overall_score,
                "final_score": final_score,
                "recommendation": recommendation,
                "skill_analysis": skill_result.to_dict(),
                "experience_analysis": exp_result
            }
        
        except Exception as e:
            logger.error(f"Matching failed for {resume.name}: {e}", exc_info=True)
            # Return zero scores on failure
            return self._get_failed_result()
    
    # ----------------------------------------------------------
    # HELPER METHODS
    # ----------------------------------------------------------
    
    def _extract_job_sections(self, job_desc: StructuredJobDescription) -> Dict[str, str]:
        """Extract and preprocess job description sections"""
        return {
            'skills': self.preprocess_text(job_desc.get_section('skills')),
            'experience': self.preprocess_text(job_desc.get_section('responsibilities')),
            'education': self.preprocess_text(job_desc.get_section('qualifications')),
            'overview': self.preprocess_text(job_desc.get_section('overview')),
            'full_text': self.preprocess_text(job_desc.get_section('full_text'))
        }
    
    def _extract_resume_sections(self, resume: StructuredResume) -> Dict[str, str]:
        """Extract and preprocess resume sections"""

        return {
            'skills': self.preprocess_text(resume.get_section('skills')),
            'experience': self.preprocess_text(resume.get_section('experience')),
            'education': self.preprocess_text(resume.get_section('education')),
            'overview': self.preprocess_text(resume.get_section('summary')),
            'full_text': self.preprocess_text(resume.get_section('full_text'))
        }

        
    
    def extract_skill_set(self, text: str) -> set:
        """
        Extract skill tokens from text.
        Very simple tokenizer for now.
        """
        if not text:
            return set()

        tokens = re.findall(r'\b[a-zA-Z\+\#\.]{2,}\b', text.lower())
        return set(tokens)

    def compute_skill_overlap(self, jd_skills_text: str, resume_skills_text: str) -> float:
        """
        Compute explicit skill coverage score.
        Returns value between 0 and 1.
        """
        jd_skills = self.extract_skill_set(jd_skills_text)
        resume_skills = self.extract_skill_set(resume_skills_text)

        if not jd_skills:
            return 0.5  # neutral

        overlap = len(jd_skills & resume_skills)
        return overlap / len(jd_skills)

    def _compute_section_scores(self, 
                                job_sections: Dict[str, str],
                                resume_sections: Dict[str, str]) -> Dict[str, float]:
        """Compute similarity for each section"""
        
        # Prepare texts for batch encoding
        texts = []
        section_keys = []
        
        for key in job_sections:
            if job_sections[key] and resume_sections[key]:
                texts.append(job_sections[key])
                texts.append(resume_sections[key])
                section_keys.append(key)
        
        if not texts:
            logger.warning("No valid sections to compare")
            return {}
        
        # Batch encode all texts
        embeddings = self.encode_batch(texts)
        
        # Compute similarities
        section_scores = {}
        for i, key in enumerate(section_keys):
            job_vec = embeddings[i * 2]
            resume_vec = embeddings[i * 2 + 1]
            section_scores[key] = self.normalized_cosine(job_vec, resume_vec)
        
        return section_scores
    
    def _compute_weighted_score(self, section_scores: Dict[str, float]) -> float:
        """Compute weighted average of section scores"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for section, weight in self.weights.items():
            if section in section_scores:
                weighted_sum += section_scores[section] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _get_failed_result(self) -> Dict:
        """Return default result for failed matches"""
        return {
            "section_scores": {},
            "weighted_score": 0.0,
            "overall_score": 0.0,
            "final_score": 0.0,
            "recommendation": "Error - Could Not Process"
        }
    
    # ----------------------------------------------------------
    # UTILITY METHODS
    # ----------------------------------------------------------
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        stats = self.cache.get_stats()
        stats['matches_processed'] = self.matches_processed
        return stats
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.cache.clear()
        logger.info("Cache cleared")