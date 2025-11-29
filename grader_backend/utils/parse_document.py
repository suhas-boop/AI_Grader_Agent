# grader_backend/utils/parse_document.py

import io

try:
    import fitz  # PyMuPDF
except ImportError:  # optional
    fitz = None

try:
    import docx  # python-docx
except ImportError:  # optional
    docx = None


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extract text from a PDF given its raw bytes."""
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF (fitz) is not installed. "
            "Install it with `pip install pymupdf` to enable PDF parsing."
        )

    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return "\n".join(pages).strip()


def extract_text_from_docx_bytes(data: bytes) -> str:
    """Extract text from a DOCX given its raw bytes."""
    if docx is None:
        raise RuntimeError(
            "python-docx is not installed. "
            "Install it with `pip install python-docx` to enable DOCX parsing."
        )

    bio = io.BytesIO(data)
    document = docx.Document(bio)
    paragraphs = [para.text for para in document.paragraphs]
    return "\n".join(paragraphs).strip()


def extract_text_from_file_bytes(data: bytes, filename: str) -> str:
    """Dispatch function: choose parser based on file extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf_bytes(data)
    if lower.endswith(".docx"):
        return extract_text_from_docx_bytes(data)
    raise RuntimeError(f"Unsupported file type: {filename}")
def extract_text_from_choice(choice: dict) -> str:
    """
    Robustly extract a text string from a choice object returned by NIM/OpenAI.
    Handles both:
      - message["content"] as a string
      - message["content"] as a list of parts with 'text' fields
    """
    message = choice.get("message", {}) or {}
    content = message.get("content")

    # If content is a list (new-style content parts), join any text fields
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if "text" in part and isinstance(part["text"], str):
                    parts.append(part["text"])
                elif part.get("type") in ("output_text", "text"):
                    t = part.get("text")
                    if isinstance(t, str):
                        parts.append(t)
        content = "".join(parts).strip() if parts else None

    if not isinstance(content, str) or not content.strip():
        snippet = json.dumps(choice, default=str)[:500]
        raise ValueError(
            f"LLM response has no usable 'content'. First choice snippet: {snippet}"
        )

    return content.strip()

