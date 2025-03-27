from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from models.model import match_resume_to_job
import pdfplumber
import docx

router = APIRouter()

def extract_text_from_pdf(file):
    """Extract text from a PDF resume."""
    with pdfplumber.open(file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

def extract_text_from_docx(file):
    """Extract text from a DOCX resume."""
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

@router.post("/upload_resume")
async def upload_resume(
    file: UploadFile = File(...), 
    job_description: str = Form(...)
):
    """Upload a resume file, extract text, and match it with a job description."""
    
    # Ensure the file type is supported
    if file.filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file.file)
    elif file.filename.endswith(".docx"):
        resume_text = extract_text_from_docx(file.file)
    else:
        return {"error": "Unsupported file format. Please upload a PDF or DOCX file."}

    # Compute similarity score
    score = match_resume_to_job(resume_text, job_description)
    return {"match_score": round(score * 100, 2)}
