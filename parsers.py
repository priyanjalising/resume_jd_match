"""
Document Section Parsers - Regex-Based Extraction Engine

PURPOSE:
    Break down monolithic text blobs from Resumes and Job Descriptions 
    into structured components. This enables "apples-to-apples" 
    semantic comparison between specific sections.

BENEFIT:
    Allows the Scoring Engine to weight specific features (e.g., Skills 
    vs. Education) differently based on the role requirements.

STRATEGY:
    1. Heuristic Header Detection: Uses common industry keywords (e.g., 
       "Work History", "Core Competencies") to identify section starts.
    2. Lookahead Capture: Extracts content following a header until 
       the next header or a significant break is detected.
    3. Fallback Logic: Implements basic extraction (like taking the 
       first paragraph) if explicit headers are missing.
"""

import re
from typing import Dict, Optional


# ==========================================================
# STRUCTURED JOB DESCRIPTION PARSER
# ==========================================================

class StructuredJobDescription:
    """
    Parses raw Job Description text into structured sections.

    Extracted Sections:
        - skills
        - responsibilities
        - qualifications
        - overview
        - full_text

    Also extracts location (if specified).
    """

    def __init__(self, raw_text: str):
        """
        Initialize parser.

        Args:
            raw_text (str): Entire job description text
        """

        self.raw_text = raw_text

        # Parse structured sections
        self.sections = self._parse_sections(raw_text)

        # Extract job location if present
        self.location = self._extract_location(raw_text)

    # ------------------------------------------------------
    # SECTION PARSING LOGIC
    # ------------------------------------------------------

    def _parse_sections(self, text: str) -> Dict[str, str]:
        """
        Extract structured sections from JD using regex patterns.

        Uses common section header keywords.
        """

        # Default section placeholders
        sections = {
            'skills': '',
            'responsibilities': '',
            'qualifications': '',
            'overview': '',
            'full_text': text  # Keep entire text for full semantic comparison
        }

        # Regex patterns for section detection
        # Strategy:
        # - Match section headers
        # - Capture content until next section or end of text
        patterns = {
            'skills': r'(?:required skills|skills|technical skills|requirements)[:\s]*\n(.*?)(?=\n\n|\n[A-Z][^:\n]+:|$)',

            'responsibilities': r'(?:responsibilities|duties|role|what you\'ll do)[:\s]*\n(.*?)(?=\n\n|\n[A-Z][^:\n]+:|$)',

            'qualifications': r'(?:qualifications|education|experience required)[:\s]*\n(.*?)(?=\n\n|\n[A-Z][^:\n]+:|$)',
        }

        for section, pattern in patterns.items():

            # IGNORECASE → case-insensitive
            # DOTALL → '.' matches newline
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

            if match:
                # Extract captured group (actual content)
                sections[section] = match.group(1).strip()

        # --------------------------------------------------
        # OVERVIEW LOGIC
        # --------------------------------------------------

        # If no explicit overview section exists,
        # we assume the first paragraph is a summary.
        paragraphs = text.split('\n\n')
        sections['overview'] = paragraphs[0] if paragraphs else text[:500]

        return sections

    # ------------------------------------------------------
    # LOCATION EXTRACTION
    # ------------------------------------------------------

    def _extract_location(self, text: str) -> Optional[str]:
        """
        Extract location from job description.

        Looks for:
            Location: City, State

        Returns lowercase string or None.
        """

        match = re.search(
            r'location[:\s]+([A-Za-z,\s]+)',
            text,
            re.IGNORECASE
        )

        return match.group(1).strip().lower() if match else None

    # ------------------------------------------------------
    # PUBLIC ACCESSOR
    # ------------------------------------------------------

    def get_section(self, name: str) -> str:
        """
        Safely retrieve a parsed section.
        Returns empty string if not found.
        """
        return self.sections.get(name, '')


# ==========================================================
# STRUCTURED RESUME PARSER
# ==========================================================

class StructuredResume:
    """
    Parses raw Resume text into structured sections.

    Extracted Sections:
        - skills
        - experience
        - education
        - summary
        - full_text

    Also extracts candidate location.
    """

    def __init__(self, raw_text: str):
        """
        Initialize resume parser.

        Args:
            raw_text (str): Entire resume text
        """

        self.raw_text = raw_text

        # Extract structured sections
        self.sections = self._parse_sections(raw_text)

        # Extract location if available
        self.location = self._extract_location(raw_text)

        # Extract candidate name
        self.name = self._extract_name(raw_text)

    # ------------------------------------------------------
    # SECTION PARSING
    # ------------------------------------------------------

    def _parse_sections(self, text: str) -> Dict[str, str]:
        """
        Extract resume sections using common header keywords.
        """

        sections = {
            'skills': '',
            'experience': '',
            'education': '',
            'summary': '',
            'full_text': text  # Used for full semantic comparison
        }

        patterns = {
            'skills': r'(?:skills|technical skills|core competencies|skills implemented)[:\s]*\n(.*?)(?=\n\n|\n[A-Z][^:\n]+:|$)',

            'experience': r'(?:experience|work experience|professional experience|employment|employment history|work history)[:\s]*\n(.*?)(?=\n\n|\n[A-Z][^:\n]+:|$)',

            'education': r'(?:education|academic background|qualifications)[:\s]*\n(.*?)(?=\n\n|\n[A-Z][^:\n]+:|$)',

            'summary': r'(?:summary|profile|objective|about)[:\s]*\n(.*?)(?=\n\n|\n[A-Z][^:\n]+:|$)',
        }

        for section, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

            if match:
                sections[section] = match.group(1).strip()

        return sections

    # ------------------------------------------------------
    # LOCATION EXTRACTION
    # ------------------------------------------------------

    def _extract_location(self, text: str) -> Optional[str]:
        """
        Extract candidate location.

        Matches phrases like:
            Location:
            Based in:
            City:
        """

        match = re.search(
            r'(?:location|based in|city)[:\s]+([A-Za-z,\s]+)',
            text,
            re.IGNORECASE
        )

        return match.group(1).strip().lower() if match else None

    # ------------------------------------------------------
    # PUBLIC ACCESSOR
    # ------------------------------------------------------

    def get_section(self, name: str) -> str:
        """
        Safely retrieve parsed section.
        Returns empty string if missing.
        """
        return self.sections.get(name, '')
    
    def _extract_name(self, text: str):
        """
        Extract candidate name from top of resume.
        Heuristic:
        - First few lines
        - Likely 2–3 capitalized words
        """

        lines = text.split('\n')[:5]

        for line in lines:
            line = line.strip()

            if "@" in line:
                continue

            if re.search(r'\d{3,}', line):
                continue

            if re.match(r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2}$', line):
                return line

        return ""

