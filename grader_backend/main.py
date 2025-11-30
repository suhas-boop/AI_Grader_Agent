# at top:
from fastapi import FastAPI, UploadFile, File, HTTPException
from grader_backend.utils.parse_document import extract_text_from_file_bytes, extract_text_from_choice
from grader_backend.utils.nim_client import chat_completion,embedding
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import logging
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 🔓 allow all origins in dev
    allow_credentials=False,      # no cookies/auth → keep this False
    allow_methods=["*"],
    allow_headers=["*"],
)
class Exemplar(BaseModel):
    text: str = Field(..., description="Plain text of the exemplar submission")
    grade: Optional[str] = Field(None, description="Optional grade label given to the exemplar")
    comments: Optional[str] = Field(None, description="Optional instructor comments on the exemplar")

class RubricCriterionLevel(BaseModel):
    label: str
    descriptor: str

class RubricCriterion(BaseModel):
    id: str
    name: str
    description: str
    weight: float = Field(..., ge=0, le=1)
    levels: List[RubricCriterionLevel]

class Rubric(BaseModel):
    title: str
    criteria: List[RubricCriterion]
    overall_notes: Optional[str]

class RubricGenerateRequest(BaseModel):
    objective: str
    exemplars: List[Exemplar] = []

class RubricGenerateResponse(BaseModel):
    rubric: Rubric
    raw_model_output: Optional[dict] = None
class GradeCriterionResult(BaseModel):
    criterion_id: str
    level_label: str
    score: float
    explanation: str

class GradeSubmissionRequest(BaseModel):
    objective: str
    rubric: Rubric
    submission_text: str

class GradeSubmissionResponse(BaseModel):
    results: List[GradeCriterionResult]
    overall_score: float
    overall_comment: str
    raw_model_output: Optional[dict] = None


# --- Prompt builder helper ---

def build_rubric_prompt(objective: str, exemplars: List[Exemplar]) -> List[dict]:
    exemplar_block = ""
    if exemplars:
        parts = []
        for idx, e in enumerate(exemplars, start=1):
            parts.append(
                f"Exemplar {idx}:\n"
                f"Grade: {e.grade or 'N/A'}\n"
                f"Comments: {e.comments or 'N/A'}\n"
                f"Text:\n{e.text}\n"
                "------"
            )
        exemplar_block = "\n\nHere are some graded exemplars:\n" + "\n\n".join(parts)

    system_msg = {
        "role": "system",
        "content": (
            "You are an expert instructor and assessment designer. "
            "Given an assignment objective and graded exemplars, design a clear, analytic grading rubric.\n\n"
            "The rubric MUST:\n"
            "- Be analytic (separate criteria like clarity, correctness, structure, citations, etc.).\n"
            "- Provide 3-5 performance levels with descriptive labels.\n"
            "- Assign each criterion a weight between 0 and 1. Weights must sum to 1.\n"
            "Return ONLY a JSON object with fields: title, criteria (array), overall_notes.\n"
            "Each criterion must have: id, name, description, weight, levels (array of {label, descriptor})."
        )
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Assignment objective:\n{objective}\n\n"
            f"{exemplar_block}\n\n"
            "Return the rubric JSON now."
        )
    }
    return [system_msg, user_msg]

def build_grading_prompt(req: GradeSubmissionRequest) -> List[dict]:
    rubric_json = req.rubric.model_dump()

    system_msg = {
        "role": "system",
        "content": (
            "You are a fair, consistent grader. "
            "You must strictly apply the provided analytic rubric to the student submission.\n\n"
            "Your job:\n"
            "- Read the assignment objective.\n"
            "- Read the rubric.\n"
            "- Read the student submission.\n"
            "- For EACH criterion in the rubric, assign exactly one performance level and a numeric score.\n"
            "- Provide a short explanation grounded in quotes or paraphrases from the submission.\n\n"
            "Return ONLY a JSON object with fields:\n"
            "{\n"
            "  \"criterion_results\": [\n"
            "    {\n"
            "      \"criterion_id\": \"string\",  // must match rubric.id\n"
            "      \"level_label\": \"string\",   // one of the rubric's level labels\n"
            "      \"score\": number,             // 0-10, where 10 is best\n"
            "      \"explanation\": \"string\"\n"
            "    }, ...\n"
            "  ],\n"
            "  \"overall_score\": number,         // weighted average 0-10 using rubric weights\n"
            "  \"overall_comment\": \"string\"     // 2-4 sentence summary\n"
            "}\n"
            "Score each criterion on a 0-10 scale where 10 is best, then compute overall_score as the "
            "weighted average of criterion scores using the rubric weights."
        ),
    }

    user_msg = {
        "role": "user",
        "content": (
            f"Assignment objective:\n{req.objective}\n\n"
            f"Rubric (JSON):\n{json.dumps(rubric_json, indent=2)}\n\n"
            "Student submission:\n"
            f"{req.submission_text}\n\n"
            "Return ONLY the JSON grading result."
        ),
    }

    return [system_msg, user_msg]
# inside your FastAPI app definition:
@app.post("/parse-document")
async def parse_document_endpoint(file: UploadFile = File(...)):
    filename = file.filename
    data = await file.read()
    try:
        text = extract_text_from_file_bytes(data, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return { "text": text }

@app.post("/rubric/generate", response_model=RubricGenerateResponse)
async def generate_rubric_endpoint(req: RubricGenerateRequest):
    messages = build_rubric_prompt(req.objective, req.exemplars)

    model_id = os.getenv("NIM_CHAT_MODEL", "qwen/qwen3-next-80b-a3b-instruct")

    try:
        
        response = chat_completion(
            model_id=model_id,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    
    try:
        choice = response["choices"][0]
        message = choice.get("message", {}) or {}
        content = message.get("content")

        
        if isinstance(content, list):
            parts = []
            for part in content:
                # handle both simple strings and dict parts
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    # try common keys used for text parts
                    if "text" in part:
                        parts.append(part["text"])
                    elif part.get("type") in ("output_text", "text"):
                        parts.append(part.get("text", ""))
            content = "".join(parts).strip() if parts else None

        if not isinstance(content, str) or not content.strip():
            # Log enough to debug, then fail cleanly
            snippet = json.dumps(choice, default=str)[:500]
            raise ValueError(f"LLM response has no usable 'content'. First choice: {snippet}")

        # Now parse JSON from the text content
        rubric_dict = json.loads(content)
        rubric = Rubric(**rubric_dict)

    except Exception as e:
        # This will show up in the FastAPI error body
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse rubric JSON from model output: {e}",
        )

    return RubricGenerateResponse(rubric=rubric, raw_model_output=response)
@app.post("/grade", response_model=GradeSubmissionResponse)
async def grade_submission_endpoint(req: GradeSubmissionRequest):
    # 1) Build prompt
    messages = build_grading_prompt(req)

    # 2) Choose model
    model_id = os.getenv("NIM_CHAT_MODEL", "qwen/qwen3-next-80b-a3b-instruct")
    if not model_id:
        raise HTTPException(status_code=500, detail="NIM_CHAT_MODEL env var not set")

    # 3) Call NIM / OpenAI-compatible endpoint
    try:
        response = chat_completion(
            messages=messages,
            model_id=model_id,
            temperature=0.2,
            max_tokens=2048,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # 4) Pull out the first choice and raw text
    try:
        if "choices" not in response or not response["choices"]:
            raise ValueError(f"No 'choices' in model response: {response}")

        choice = response["choices"][0]
        text = extract_text_from_choice(choice)

        # Log for debugging
        logger.info("=== RAW LLM GRADE TEXT START ===")
        logger.info(text)
        logger.info("=== RAW LLM GRADE TEXT END ===")

        # 5) Strip code fences like ```json ... ```
        if text.startswith("```"):
            parts = text.split("```")
            for part in parts:
                p = part.strip()
                # ```json\n{...}
                if p.lower().startswith("json"):
                    text = p[4:].lstrip()
                    break

        # 6) Try direct JSON parse first
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to extract the largest {...} block
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and last > first:
                candidate = text[first : last + 1]
                parsed = json.loads(candidate)
            else:
                # Re-raise the original error with context
                raise

        # 7) Validate into Pydantic models
        criterion_results: List[GradeCriterionResult] = []
        for item in parsed.get("criterion_results", []):
            normalized = {
                # LLM may return "criterion_id" OR "id"
                "criterion_id": item.get("criterion_id") or item.get("id") or "",
                "level_label": item.get("level_label") or "",
                "score": float(item.get("score", 0.0)),
                # LLM may call this "explanation" or "comment"
                "explanation": item.get("explanation") or item.get("comment") or "",
            }
            criterion_results.append(GradeCriterionResult(**normalized))

        overall_score = float(parsed.get("overall_score", 0.0))
        overall_comment = str(parsed.get("overall_comment", ""))


    except Exception as e:
        # Log the full traceback on the server
        logger.exception("Failed to parse grading JSON from model output")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse grading JSON from model output: {e}",
        )

    # 8) Return structured response
    return GradeSubmissionResponse(
        results=criterion_results,
        overall_score=overall_score,
        overall_comment=overall_comment,
        raw_model_output=response,
    )
