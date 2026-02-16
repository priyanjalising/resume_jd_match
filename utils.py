import os
import fitz
from docx import Document


def print_separator(title: str = ""):
    """
    Prints a visual separator line in console output.

    If a title is provided:
        - Prints a full-width banner with centered title
    If no title:
        - Prints a simple dashed line

    Useful for formatting CLI output cleanly.
    """

    if title:
        print(f"\n{'=' * 80}")
        print(f"{title:^80}")   # Center-align title within 80 characters
        print('=' * 80)
    else:
        print('-' * 80)


def load_job_description(filepath: str) -> str:
    """
    Reads the job description text from a file.

    Parameters:
        filepath (str): Path to the job description .txt file

    Returns:
        str: Full text of the job description

    Raises:
        FileNotFoundError: If the file does not exist

    Why explicit error?
        So failures are clear and fail fast rather than silent.
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Job description file not found at: {filepath}"
        )

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_pdf(filepath: str) -> str:
    """
    Extract text from PDF file.
    """
    try:
        doc = fitz.open(filepath)
        text = ""

        for page in doc:
            text += page.get_text()

        doc.close()
        return text.strip()

    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {filepath}\n{e}")

def extract_text_from_docx(filepath: str) -> str:
    """
    Extract text from Word (.docx) file.
    """
    try:
        doc = Document(filepath)
        text = []

        for paragraph in doc.paragraphs:
            text.append(paragraph.text)

        return "\n".join(text).strip()

    except Exception as e:
        raise ValueError(f"Failed to extract text from DOCX: {filepath}\n{e}")

def load_resume(filepath: str) -> str:
    """
    Reads a single resume text file.

    Parameters:
        filepath (str): Path to a resume .txt file

    Returns:
        str: Full text of the resume

    Raises:
        FileNotFoundError: If file does not exist
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Resume file not found at: {filepath}"
        )

    _, ext = os.path.splitext(filepath)

    ext = ext.lower()

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()

    elif ext == ".pdf":
        return extract_text_from_pdf(filepath)
    
    elif ext == ".docx":
        return extract_text_from_docx(filepath)

    else:
        raise ValueError(f"Unsupported file type: {ext}")
#   with open(filepath, "r", encoding="utf-8") as f:
#         return f.read()


def load_resumes_from_directory(directory: str):
    """
    Loads all `.txt` resume files from a given directory.

    Parameters:
        directory (str): Path to directory containing resume .txt files

    Returns:
        tuple:
            resumes (List[str]) → List of resume text content
            names (List[str]) → List of candidate names derived from file names

    Behavior:
        - Only processes `.txt` files
        - Sorts files alphabetically (ensures consistent ranking order)
        - Raises error if directory doesn't exist
        - Raises error if no resumes found

    Why sorting?
        Ensures deterministic behavior across runs.
    """

    if not os.path.exists(directory):
        raise FileNotFoundError(
            f"Resume directory not found at: {directory}"
        )

    resumes = []
    names = []

    # Iterate over files in deterministic (sorted) order
    for filename in sorted(os.listdir(directory)):

        if filename.lower().endswith((".txt", ".pdf", ".docx")):

            filepath = os.path.join(directory, filename)

            try:
                text = load_resume(filepath)

                if text:
                    resumes.append(text)
                    names.append(os.path.splitext(filename)[0])
                else:
                    print(f"⚠️ Empty resume skipped: {filename}")

            except Exception as e:
                print(f"⚠️ Skipping {filename}: {e}")

    if not resumes:
        raise ValueError(f"No valid resumes found in: {directory}")

    return resumes, names

    # for filename in sorted(os.listdir(directory)):

    #     # Only process text files
    #     if filename.endswith('.txt'):
    #         filepath = os.path.join(directory, filename)

    #         # Read resume content
    #         resumes.append(load_resume(filepath))

    #         # Extract candidate name (filename without extension)
    #         names.append(os.path.splitext(filename)[0])

    # # Fail fast if directory contains no usable resumes
    # if not resumes:
    #     raise ValueError(f"No .txt files found in: {directory}")

    # return resumes, names
