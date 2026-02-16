"""
Execution Wrapper - Orchestration and Automated Logging

PURPOSE:
    Provides a high-level entry point for the matching engine, ensuring 
    that every session is reproducible, logged, and easy to execute 
    via a Command Line Interface (CLI).

BENEFIT:
    Automates the "boring stuff"—creating output directories, generating 
    timestamped logs, and formatting summaries—so recruiters and 
    developers can focus on match quality.

STRATEGY:
    1. CLI Argument Parsing: Allows runtime overrides for file paths and
       debug levels without modifying the core logic.
    2. Dynamic Logging: Routes system output to both the console (for real-time
       feedback) and a persistent log file (for audit trails).
    3. Error Resilience: Wraps the entire pipeline in high-level try/except 
       blocks to provide clean, informative error messages during failures.

Usage:
    python run_matching.py
    python run_matching.py --debug
    python run_matching.py --custom-config
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from engine import MatchingEngine, setup_logging
from config import Config
from results_save import save_results_to_csv


def create_output_directory():
    """Ensure output directory exists"""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def generate_log_filename(prefix="matching"):
    """
    Creates a unique, timestamped filename for the session log.
    Example: output/matching_20260216_115531.log
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"output/{prefix}_{timestamp}.log"


def run_matching_with_logging(
    jd_path="data/job_description.txt",
    resumes_dir="data/resume",
    output_csv="output/matching_results.csv",
    log_level="INFO",
    custom_config=None
):
    """
    The main orchestration function. Coordinates the Engine, 
    Logging, and Result Exporter.
    """
    # Create output directory
    create_output_directory()
    
    # Generate log filename
    log_file = generate_log_filename()
    
    # Setup logging (both console and file)
    setup_logging(level=log_level, log_file=log_file)
    
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("RESUME MATCHING SESSION STARTED")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Job Description: {jd_path}")
    logger.info(f"Resumes Directory: {resumes_dir}")
    logger.info(f"Output CSV: {output_csv}")
    logger.info(f"Log Level: {log_level}")
    logger.info("="*80)
    
    try:
        # Initialize engine
        config = custom_config or Config()
        engine = MatchingEngine(config=config, verbose=True)
        
        # Run matching
        results = engine.match_resumes(jd_path, resumes_dir)
        
        # Save results
        save_results_to_csv(
            results.results,
            results.candidate_names,
            output_path=output_csv
        )
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("SESSION SUMMARY")
        logger.info("="*80)
        logger.info(f"✅ Processed: {results.total_resumes} resumes")
        logger.info(f"✅ Time: {results.processing_time:.2f}s")
        logger.info(f"✅ Results saved to: {output_csv}")
        logger.info(f"✅ Log saved to: {log_file}")
        
        # Top 3 candidates
        logger.info("\nTop 3 Candidates:")
        for i, result in enumerate(results.results[:3], 1):
            idx = result['resume_id']
            name = results.candidate_names[idx]
            score = result['overall_score']
            logger.info(f"  {i}. {name}: {score:.2%}")
        
        logger.info("="*80)
        
        print(f"\n✅ Complete! Log file: {log_file}")
        
        return results
    
    except Exception as e:
        logger.error(f"\n❌ MATCHING FAILED: {e}", exc_info=True)
        print(f"\n❌ Error occurred. Check log file: {log_file}")
        raise


def main():
    """
    Standard entry point using argparse to handle CLI interactions.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run resume matching with automatic logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                # Basic run
                python run_matching.py
                
                # Debug mode
                python run_matching.py --debug
                
                # Custom paths
                python run_matching.py --jd my_job.txt --resumes my_resumes/
                
                # Custom config
                python run_matching.py --custom-config
                        """
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
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--custom-config',
        action='store_true',
        help='Use custom configuration (modify in script)'
    )
    
    args = parser.parse_args()
    
    # Determine log level
    log_level = "DEBUG" if args.debug else "INFO"
    
    # Custom config if requested
    custom_config = None
    if args.custom_config:
        from config import Config, ScoringWeights
        
        custom_config = Config()
        
        # Example customization - modify as needed
        custom_config.weights = ScoringWeights(
            skills=0.40,        # Prioritize skills
            experience=0.35,
            education=0.10,
            overview=0.10,
            location=0.05
        )
        
        print("📝 Using custom configuration:")
        print(f"   Skills weight: {custom_config.weights.skills}")
        print(f"   Experience weight: {custom_config.weights.experience}")
    
    # Run matching
    results = run_matching_with_logging(
        jd_path=args.jd,
        resumes_dir=args.resumes,
        output_csv=args.output,
        log_level=log_level,
        custom_config=custom_config
    )
    
    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)