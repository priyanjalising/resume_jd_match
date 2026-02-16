"""
 Resume Matching Engine with:
- Proper logging setup
- Progress tracking
- Better error handling
- CLI interface
- Statistics reporting
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

from config import Config, DEFAULT_CONFIG
from utils import load_job_description, load_resumes_from_directory
from matcher import SemanticMatcher
from hybrid_parser import HybridParser
from parsers import StructuredJobDescription
from results_save import save_results_to_csv


# ----------------------------------------------------------
# LOGGING SETUP
# ----------------------------------------------------------

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure logging with console and optional file output.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging output
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy libraries
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# RESULTS CONTAINER
# ----------------------------------------------------------

@dataclass
class MatchingResults:
    """Container for matching results with metadata"""
    results: List[Dict]
    candidate_names: List[str]
    job_description_path: str
    total_resumes: int
    processing_time: float
    config: Config
    
    def get_top_k(self, k: int) -> List[Dict]:
        """Get top K candidates"""
        return self.results[:min(k, len(self.results))]
    
    def get_by_rank(self, rank: int) -> Optional[Dict]:
        """Get candidate by rank"""
        for result in self.results:
            if result['rank'] == rank:
                return result
        return None
    
    def get_by_name(self, name: str) -> Optional[Dict]:
        """Get candidate by name"""
        for result in self.results:
            idx = result['resume_id']
            if self.candidate_names[idx] == name:
                return result
        return None


# ----------------------------------------------------------
# MAIN ENGINE
# ----------------------------------------------------------

class MatchingEngine:
    """
     resume matching engine with:
    - Progress tracking
    - Statistics collection
    - Error handling
    - Configurable behavior
    """
    
    def __init__(self, config: Optional[Config] = None, verbose: bool = True):
        """
        Initialize matching engine.
        
        Args:
            config: Configuration object
            verbose: Show progress bars and detailed output
        """
        self.config = config or DEFAULT_CONFIG
        self.verbose = verbose
        
        # Initialize components
        logger.info("Initializing matching engine...")
        
        self.parser = HybridParser(config=self.config)
        logger.info("✅ Hybrid parser initialized")
        
        self.matcher = SemanticMatcher(config=self.config)
        logger.info("✅ Semantic matcher initialized")
    
    def match_resumes(self,
                     job_description_path: str,
                     resumes_directory: str) -> MatchingResults:
        """
        Run complete matching pipeline.
        
        Args:
            job_description_path: Path to job description file
            resumes_directory: Directory containing resume files
            
        Returns:
            MatchingResults object with detailed results
        """
        import time
        start_time = time.time()
        
        logger.info("="*80)
        logger.info("STARTING RESUME MATCHING PIPELINE")
        logger.info("="*80)
        
        # Step 1: Load job description
        logger.info(f"Loading job description from: {job_description_path}")
        try:
            jd_text = load_job_description(job_description_path)
            job_desc = StructuredJobDescription(jd_text)
            logger.info(f"✅ Job description loaded ({len(jd_text)} characters)")
        except Exception as e:
            logger.error(f"Failed to load job description: {e}")
            raise
        
        # Step 2: Load resumes
        logger.info(f"Loading resumes from: {resumes_directory}")
        try:
            resume_texts, candidate_names = load_resumes_from_directory(resumes_directory)
            logger.info(f"✅ Loaded {len(resume_texts)} resumes")
        except Exception as e:
            logger.error(f"Failed to load resumes: {e}")
            raise
        
        # Step 3: Parse and match resumes
        logger.info("Starting resume matching...")
        results = []
        
        # Use progress bar if verbose
        iterator = enumerate(resume_texts)
        if self.verbose:
            iterator = tqdm(
                iterator,
                total=len(resume_texts),
                desc="Matching resumes",
                unit="resume"
            )
        
        for idx, resume_text in iterator:
            try:
                # Parse resume
                resume = self.parser.parse(resume_text)
                
                # Match against job description
                match_result = self.matcher.match_structured(job_desc, resume)
                
                # Build result
                result = self._build_result(idx, match_result)
                results.append(result)
                
                if self.verbose and not isinstance(iterator, enumerate):
                    # Update progress bar description
                    iterator.set_postfix({
                        'current': candidate_names[idx][:20],
                        'score': f"{result['overall_score']:.2%}"
                    })
            
            except Exception as e:
                logger.error(
                    f"Failed to process resume {idx} ({candidate_names[idx]}): {e}",
                    exc_info=True
                )
                # Add failed result
                results.append(self._build_failed_result(idx))
        
        # Step 4: Rank results
        logger.info("Ranking candidates...")
        results_sorted = sorted(
            results,
            key=lambda x: x['overall_score'],
            reverse=True
        )
        
        for rank, result in enumerate(results_sorted, 1):
            result['rank'] = rank
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        logger.info("="*80)
        logger.info(f"✅ MATCHING COMPLETE in {processing_time:.2f}s")
        logger.info(f"   Processed: {len(results)} resumes")
        logger.info(f"   Average: {processing_time/len(results):.2f}s per resume")
        logger.info("="*80)
        
        # Print statistics
        self._print_statistics(results_sorted, candidate_names)
        
        return MatchingResults(
            results=results_sorted,
            candidate_names=candidate_names,
            job_description_path=job_description_path,
            total_resumes=len(results),
            processing_time=processing_time,
            config=self.config
        )

    def _build_result(self, idx: int, match_result: Dict) -> Dict:
        """Build a complete result dictionary by harvesting nested analysis."""
        
        # 1. Extract nested dictionaries from the matcher's output
        section_scores = match_result.get("section_scores", {})
        skill_analysis = match_result.get("skill_analysis", {})
        exp_analysis = match_result.get("experience_analysis", {})
        
        # 2. Map the data to the keys expected by results_save.py
        return {
            'resume_id': idx,
            'rank': None, # Set later by ranker
            'overall_score': round(match_result.get('final_score', 0.0), 4),
            'recommendation': match_result.get('recommendation', "N/A"),
            
            # Section Scores (Semantic)
            'section_scores': section_scores, # results_save.py expects this dictionary
            'weighted_score': round(match_result.get('weighted_score', 0.0), 4),
            
            # Skill Analysis
            'skill_analysis': skill_analysis, # results_save.py extracts matched/missing from here
            
            # Experience Analysis
            'experience_analysis': exp_analysis # results_save.py extracts years/gap from here
        }
    
    def _build_failed_result(self, idx: int) -> Dict:
        """Build result for failed processing"""
        return {
            'resume_id': idx,
            'overall_score': 0.0,
            'skills_score': 0.0,
            'experience_score': 0.0,
            'education_score': 0.0,
            'location_score': 0.0,
            'weighted_score': 0.0,
            'recommendation': "Error - Processing Failed",
            'rank': None
        }
    
    def _print_statistics(self, results: List[Dict], names: List[str]):
        """Print matching statistics"""
        logger.info("\n" + "="*80)
        logger.info("MATCHING STATISTICS")
        logger.info("="*80)
        
        # Score distribution
        scores = [r['overall_score'] for r in results]
        logger.info(f"\nScore Distribution:")
        logger.info(f"  Mean:    {sum(scores)/len(scores):.2%}")
        logger.info(f"  Median:  {sorted(scores)[len(scores)//2]:.2%}")
        logger.info(f"  Min:     {min(scores):.2%}")
        logger.info(f"  Max:     {max(scores):.2%}")
        
        # Recommendation breakdown
        recommendations = {}
        for result in results:
            rec = result['recommendation']
            recommendations[rec] = recommendations.get(rec, 0) + 1
        
        logger.info(f"\nRecommendation Breakdown:")
        for rec, count in sorted(recommendations.items(), key=lambda x: -x[1]):
            logger.info(f"  {rec}: {count}")
        
        # Parser statistics
        parser_stats = self.parser.get_stats()
        logger.info(f"\nParser Performance:")
        logger.info(f"  Regex success rate: {parser_stats['regex_success_rate']:.1%}")
        logger.info(f"  LLM fallbacks: {parser_stats['llm_fallbacks']}")
        logger.info(f"  Failures: {parser_stats['failures']}")
        
        # Cache statistics
        cache_stats = self.matcher.get_cache_stats()
        logger.info(f"\nCache Performance:")
        logger.info(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
        logger.info(f"  Cache size: {cache_stats['size']}")
        
        # Top 5 candidates
        logger.info(f"\nTop 5 Candidates:")
        for i, result in enumerate(results[:5], 1):
            idx = result['resume_id']
            name = names[idx]
            score = result['overall_score']
            rec = result['recommendation']
            logger.info(f"  {i}. {name}: {score:.2%} - {rec}")
        
        logger.info("="*80 + "\n")
    
    def print_detailed_results(self, matching_results: MatchingResults):
        """Print detailed results for each candidate"""
        print("\n" + "="*80)
        print("DETAILED MATCHING RESULTS")
        print("="*80)
        
        for result in matching_results.results:
            idx = result['resume_id']
            name = matching_results.candidate_names[idx]
            
            print(f"\n{'='*80}")
            print(f"Rank #{result['rank']}: {name}")
            print('='*80)
            print(f"Overall Score:        {result['overall_score']:.0%}")
            print(f"Recommendation:       {result['recommendation']}")
            print()
            print("Section Breakdown:")
            print(f"  • Skills Match:       {result['skills_score']:.0%} {self._get_badge(result['skills_score'])}")
            print(f"  • Experience Match:   {result['experience_score']:.0%} {self._get_badge(result['experience_score'])}")
            print(f"  • Education Match:    {result['education_score']:.0%} {self._get_badge(result['education_score'])}")
            print(f"  • Location Match:     {result['location_score']:.0%}")
            print(f"  • Weighted Score:     {result['weighted_score']:.0%}")
    
    @staticmethod
    def _get_badge(score: float) -> str:
        """Get emoji badge for score"""
        if score >= 0.8:
            return "✅ Excellent"
        elif score >= 0.65:
            return "✓ Good"
        else:
            return "⚠️ Needs improvement"


# ----------------------------------------------------------
# CLI INTERFACE
# ----------------------------------------------------------

def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Resume Matching Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--jd',
        type=str,
        default='data/job_description.txt',
        help='Path to job description file'
    )
    
    parser.add_argument(
        '--resumes',
        type=str,
        default='data/resume',
        help='Directory containing resume files'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output/matching_results.csv',
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Optional log file path'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Disable progress bars'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Print detailed results for each candidate'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    
    # Initialize engine
    engine = MatchingEngine(verbose=not args.quiet)
    
    # Run matching
    try:
        results = engine.match_resumes(args.jd, args.resumes)
        
        # Print detailed results if requested
        if args.detailed:
            engine.print_detailed_results(results)
        
        # Save to CSV
        save_results_to_csv(
            results.results,
            results.candidate_names,
            output_path=args.output
        )
        
        logger.info(f"\n✅ Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()