import os
from threading import Thread



try:
    import google.generativeai as genai
except ImportError:
    genai = None

import ocr_service

# Placeholder for API Key - in production use os.getenv('GEMINI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if GEMINI_API_KEY and genai:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"GenAI Config Error: {e}")
        genai = None

def perform_ocr(image_path):
    """
    Extract text using the shared OCR service.
    """
    return ocr_service.extract_text_robust(image_path)

def generate_ai_feedback(text, assignment_description=""):
    """
    Generate feedback using Google Gemini or Mock.
    """
    if not text or len(text) < 10:
        return "Text too short for analysis.", "N/A"

    if GEMINI_API_KEY and genai:
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            prompt = f"Analyze the following student submission and provide constructive feedback and a grade (0-100). Assignment context: {assignment_description}. \n\nSubmission:\n{text}"
            
            response = model.generate_content(prompt)
            feedback = response.text
             # basic parsing for grade (imperfect)
            grade = "Pending" 
            if "Grade:" in feedback:
                grade = feedback.split("Grade:")[1].split()[0]
            
            return feedback, grade
        except Exception as e:
            print(f"AI Service Error (falling back to mock): {str(e)}")
            return mock_ai_feedback(text)

    else:
        # MOCK IMPLEMENTATION
        return mock_ai_feedback(text)

def mock_ai_feedback(text):
    """
    Mock feedback generator for demo purposes.
    """
    word_count = len(text.split())
    feedback = f"""
    AI Analysis
    
    - Content Analysis: The submission contains approximately {word_count} words.
    - Strengths: Good effort in writing. The concepts seem improved.
    - Areas for Improvement: Consider expanding on the key arguments. Check for grammar in the second paragraph.
    """
    return feedback, "85/100"
