"""
Enhanced CSV Export - Save Complete Match Results

- All section scores (skills, experience, education, overview, location)
- Skill analysis (matched skills, missing skills, match percentage)
- Experience analysis (years, requirements, gap)
- Multiple export formats (detailed CSV, summary CSV, Excel)
"""

import os
import csv
import json
from typing import List, Dict
import pandas as pd


def save_results_to_csv(results, candidate_names, output_path="output/matching_results.csv"):
    """
    Save DETAILED resume matching results to CSV file.
    
    NOW INCLUDES:
    - All section scores
    - Skill match details
    - Experience analysis
    - Full breakdown
    
    Creates output directory if it does not exist.
    """
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        # Write EXPANDED header with all new fields
        writer.writerow([
            # Basic Info
            "Rank",
            "Candidate Name",
            
            # Main Scores
            "Final Score",
            "Recommendation",
            
            # Section Scores (Semantic Similarity)
            "Skills Semantic Score",
            "Experience Semantic Score",
            "Education Semantic Score",
            "Overview Semantic Score",
            "Location Score",
            
            # Aggregated Scores
            "Weighted Semantic Score",
            "Overall Semantic Score",
            
            # Skill Analysis
            "Skill Match %",
            "Matched Skills",
            "Missing Skills",
            "Extra Skills",
            
            # Experience Analysis
            "Candidate Years",
            "Required Years",
            "Meets Experience Req",
            "Experience Gap",
            "Experience Score"
        ])
        
        # Write rows with all data
        for result in results:
            idx = result["resume_id"]
            
            # Get skill analysis (if available)
            skill_analysis = result.get("skill_analysis", {})
            matched_skills = skill_analysis.get("matched_skills", [])
            missing_skills = skill_analysis.get("missing_skills", [])
            extra_skills = skill_analysis.get("extra_skills", [])
            skill_match_pct = skill_analysis.get("match_percentage", 0.0)
            
            # Get experience analysis (if available)
            exp_analysis = result.get("experience_analysis", {})
            candidate_years = exp_analysis.get("candidate_years", 0.0)
            
            # Handle required years (might be ExperienceRequirement object or dict)
            required = exp_analysis.get("required", {})
            if hasattr(required, 'min_years'):
                required_years = required.min_years or 0.0
            elif isinstance(required, dict):
                required_years = required.get("min_years", 0.0)
            else:
                required_years = 0.0
            
            meets_req = exp_analysis.get("meets_requirements", "Unknown")
            exp_gap = exp_analysis.get("gap", 0.0)
            exp_score = exp_analysis.get("score", 0.0)
            
            # Get section scores
            section_scores = result.get("section_scores", {})
            
            # Write row
            writer.writerow([
                # Basic Info
                result["rank"],
                candidate_names[idx],
                
                # Main Scores
                f"{result['overall_score']:.2%}",
                result["recommendation"],
                
                # Section Scores
                f"{section_scores.get('skills', 0.0):.2%}",
                f"{section_scores.get('experience', 0.0):.2%}",
                f"{section_scores.get('education', 0.0):.2%}",
                f"{section_scores.get('overview', 0.0):.2%}",
                f"{section_scores.get('location', 0.0):.2%}",
                
                # Aggregated Scores
                f"{result.get('weighted_score', 0.0):.2%}",
                f"{result.get('overall_score', 0.0):.2%}",
                
                # Skill Analysis
                f"{skill_match_pct:.2%}",
                ", ".join(matched_skills) if matched_skills else "None",
                ", ".join(missing_skills) if missing_skills else "None",
                ", ".join(extra_skills) if extra_skills else "None",
                
                # Experience Analysis
                f"{candidate_years:.1f}",
                f"{required_years:.1f}",
                meets_req,
                f"{exp_gap:+.1f}",
                f"{exp_score:.2%}"
            ])
    
    print(f"\n✅ Detailed results saved to: {output_path}")


def save_summary_csv(results, candidate_names, output_path="output/matching_summary.csv"):
    """
    Save SUMMARY version (fewer columns for quick review).
    
    Best for: Quick scanning, top candidates only
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        # Simplified header
        writer.writerow([
            "Rank",
            "Name",
            "Score",
            "Recommendation",
            "Skills Match",
            "Experience",
            "Missing Skills"
        ])
        
        # Write rows
        for result in results:
            idx = result["resume_id"]
            
            skill_analysis = result.get("skill_analysis", {})
            exp_analysis = result.get("experience_analysis", {})
            
            missing_skills = skill_analysis.get("missing_skills", [])
            candidate_years = exp_analysis.get("candidate_years", 0.0)
            
            writer.writerow([
                result["rank"],
                candidate_names[idx],
                f"{result['overall_score']:.0%}",
                result["recommendation"],
                f"{skill_analysis.get('match_percentage', 0.0):.0%}",
                f"{candidate_years:.0f} yrs",
                ", ".join(list(missing_skills)[:3]) if missing_skills else "✓ All"
            ])
    
    print(f"✅ Summary saved to: {output_path}")


def save_to_excel(results, candidate_names, output_path="output/matching_results.xlsx"):
    """
    Save to Excel with multiple sheets and formatting.
    
    SHEETS:
    1. Summary - Top-level overview
    2. Detailed - All scores and analysis
    3. Skills - Skill match breakdown
    4. Experience - Experience analysis
    """
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare data for pandas
        data_detailed = []
        data_summary = []
        data_skills = []
        data_experience = []
        
        for result in results:
            idx = result["resume_id"]
            name = candidate_names[idx]
            
            skill_analysis = result.get("skill_analysis", {})
            exp_analysis = result.get("experience_analysis", {})
            section_scores = result.get("section_scores", {})
            
            # Detailed sheet data
            data_detailed.append({
                'Rank': result['rank'],
                'Name': name,
                'Overall Score': result['overall_score'],
                'Recommendation': result['recommendation'],
                'Skills Semantic': section_scores.get('skills', 0.0),
                'Experience Semantic': section_scores.get('experience', 0.0),
                'Education Semantic': section_scores.get('education', 0.0),
                'Location': section_scores.get('location', 0.0),
                'Skill Match %': skill_analysis.get('match_percentage', 0.0),
                'Candidate Years': exp_analysis.get('candidate_years', 0.0),
                'Experience Score': exp_analysis.get('score', 0.0)
            })
            
            # Summary sheet data
            data_summary.append({
                'Rank': result['rank'],
                'Name': name,
                'Score': result['overall_score'],
                'Recommendation': result['recommendation']
            })
            
            # Skills sheet data
            matched = skill_analysis.get('matched_skills', [])
            missing = skill_analysis.get('missing_skills', [])
            data_skills.append({
                'Rank': result['rank'],
                'Name': name,
                'Match %': skill_analysis.get('match_percentage', 0.0),
                'Matched': ', '.join(matched) if matched else 'None',
                'Missing': ', '.join(missing) if missing else 'None'
            })
            
            # Experience sheet data
            required = exp_analysis.get('required', {})
            if hasattr(required, 'min_years'):
                required_years = required.min_years or 0.0
            elif isinstance(required, dict):
                required_years = required.get('min_years', 0.0)
            else:
                required_years = 0.0
            
            data_experience.append({
                'Rank': result['rank'],
                'Name': name,
                'Candidate Years': exp_analysis.get('candidate_years', 0.0),
                'Required Years': required_years,
                'Meets Requirement': exp_analysis.get('meets_requirements', 'Unknown'),
                'Gap': exp_analysis.get('gap', 0.0),
                'Score': exp_analysis.get('score', 0.0)
            })
        
        # Create Excel file with multiple sheets
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            pd.DataFrame(data_summary).to_excel(
                writer, sheet_name='Summary', index=False
            )
            
            # Detailed sheet
            pd.DataFrame(data_detailed).to_excel(
                writer, sheet_name='Detailed', index=False
            )
            
            # Skills sheet
            pd.DataFrame(data_skills).to_excel(
                writer, sheet_name='Skills', index=False
            )
            
            # Experience sheet
            pd.DataFrame(data_experience).to_excel(
                writer, sheet_name='Experience', index=False
            )
        
        print(f"✅ Excel file saved to: {output_path}")
        
    except ImportError:
        print("⚠️ Excel export requires 'openpyxl'. Install with: pip install openpyxl")
        print("   Falling back to CSV export...")
        save_results_to_csv(results, candidate_names, output_path.replace('.xlsx', '.csv'))


def save_json_export(results, candidate_names, output_path="output/matching_results.json"):
    """
    Save as JSON for programmatic access.
    
    Best for: API integration, further processing
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare JSON-serializable data
    json_data = []
    
    for result in results:
        idx = result["resume_id"]
        
        # Deep copy and add candidate name
        result_copy = result.copy()
        result_copy['candidate_name'] = candidate_names[idx]
        
        # Convert sets to lists for JSON serialization
        if 'skill_analysis' in result_copy:
            skill_analysis = result_copy['skill_analysis']
            for key in ['matched_skills', 'missing_skills', 'extra_skills']:
                if key in skill_analysis and isinstance(skill_analysis[key], set):
                    skill_analysis[key] = list(skill_analysis[key])
        
        json_data.append(result_copy)
    
    # Save JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON export saved to: {output_path}")


def save_all_formats(results, candidate_names, base_path="output/matching_results"):
    """
    Save in ALL formats for maximum flexibility.
    
    Creates:
    - matching_results.csv (detailed)
    - matching_summary.csv (quick view)
    - matching_results.xlsx (Excel with multiple sheets)
    - matching_results.json (programmatic access)
    """
    
    print("\n" + "="*60)
    print("EXPORTING RESULTS IN MULTIPLE FORMATS")
    print("="*60)
    
    # Remove extension if present
    base_path = base_path.replace('.csv', '').replace('.xlsx', '').replace('.json', '')
    
    # Save all formats
    save_results_to_csv(results, candidate_names, f"{base_path}.csv")
    save_summary_csv(results, candidate_names, f"{base_path}_summary.csv")
    save_to_excel(results, candidate_names, f"{base_path}.xlsx")
    save_json_export(results, candidate_names, f"{base_path}.json")
    
    print("\n" + "="*60)
    print("✅ ALL FORMATS EXPORTED SUCCESSFULLY")
    print("="*60)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """
    Example of how to use the improved export functions.
    """
    
    # Mock results (replace with your actual results)
    results = [{
        'rank': 1,
        'resume_id': 0,
        'overall_score': 0.85,
        'recommendation': 'Strong Match',
        'section_scores': {
            'skills': 0.90,
            'experience': 0.85,
            'education': 0.75,
            'location': 1.0
        },
        'weighted_score': 0.87,
        'overall_score': 0.84,
        'skill_analysis': {
            'match_percentage': 0.90,
            'matched_skills': ['python', 'aws', 'docker'],
            'missing_skills': ['kubernetes'],
            'extra_skills': ['rust', 'golang']
        },
        'experience_analysis': {
            'candidate_years': 7.0,
            'required': {'min_years': 5.0},
            'meets_requirements': True,
            'gap': 2.0,
            'score': 1.0
        }
    }]
    
    candidate_names = ["John Doe"]
    
    # Option 1: Detailed CSV only
    save_results_to_csv(results, candidate_names)
    
    # Option 2: Summary CSV only
    save_summary_csv(results, candidate_names)
    
    # Option 3: Excel with multiple sheets
    save_to_excel(results, candidate_names)
    
    # Option 4: JSON export
    save_json_export(results, candidate_names)
    
    # Option 5: ALL formats at once!
    save_all_formats(results, candidate_names)


if __name__ == "__main__":
    print("Results Export Module - Enhanced Version")
    print("="*60)
    print("\nAvailable functions:")
    print("1. save_results_to_csv() - Detailed CSV with all fields")
    print("2. save_summary_csv() - Quick summary view")
    print("3. save_to_excel() - Multi-sheet Excel file")
    print("4. save_json_export() - JSON for APIs")
    print("5. save_all_formats() - Export everything!")
    print("\nRun example_usage() to see demonstrations.")