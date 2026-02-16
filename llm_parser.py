"""
LLM-Based Structured Parser - Intelligence for Unstructured Resumes

PURPOSE:
    Provides a high-accuracy fallback when standard regex parsing fails. 
    It leverages Large Language Models to "understand" resume and JD 
    layouts that are non-standard, messy, or highly complex.

BENEFIT:
    Increases the extraction success rate for "noisy" or unique resumes,
    ensuring critical matching data (skills, years) is not lost.

STRATEGY:
    1. Receive raw text from HybridParser.
    2. Use OpenAI's JSON Schema enforcement to guarantee structured output.
    3. Map unstructured blobs into standardized sections for the Semantic Matcher.
"""

import json
from typing import Dict
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


class LLMStructuredParser:

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    # ======================================================
    # RESUME PARSER
    # ======================================================

    def parse_resume(self, raw_text: str) -> Dict:
        """
        Extract structured resume information.

        Returns:
            {
                "name": str,
                "skills": str,
                "experience": str,
                "education": str,
                "summary": str,
                "location": str
            }
        """

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "resume_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "skills": {"type": "string"},
                            "experience": {"type": "string"},
                            "education": {"type": "string"},
                            "summary": {"type": "string"},
                            "location": {"type": "string"}
                        },
                        "required": [
                            "name",
                            "skills",
                            "experience",
                            "education",
                            "summary",
                            "location"
                        ]
                    }
                }
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured resume information. "
                        "Return ONLY valid JSON matching the schema. "
                        "If any field is missing, return an empty string."
                    )
                },
                {
                    "role": "user",
                    "content": raw_text
                }
            ]
        )

        return json.loads(response.choices[0].message.content)

    # ======================================================
    # JOB DESCRIPTION PARSER
    # ======================================================

    def parse_job_description(self, raw_text: str) -> Dict:
        """
        Extract structured job description information.

        Returns:
            {
                "skills": str,
                "responsibilities": str,
                "qualifications": str,
                "overview": str,
                "location": str
            }
        """

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "jd_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "skills": {"type": "string"},
                            "responsibilities": {"type": "string"},
                            "qualifications": {"type": "string"},
                            "overview": {"type": "string"},
                            "location": {"type": "string"}
                        },
                        "required": [
                            "skills",
                            "responsibilities",
                            "qualifications",
                            "overview",
                            "location"
                        ]
                    }
                }
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured job description information. "
                        "Return ONLY valid JSON matching the schema. "
                        "If any field is missing, return an empty string."
                    )
                },
                {
                    "role": "user",
                    "content": raw_text
                }
            ]
        )

        return json.loads(response.choices[0].message.content)
