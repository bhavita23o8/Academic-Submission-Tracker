import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def normalize_text(text):
    """
    Normalize text: lowercase, remove special characters.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def calculate_similarity(text1, text2):
    """
    Calculate variable length text similarity using TF-IDF and Cosine Similarity.
    Returns percentage (0-100).
    """
    if not text1 or not text2:
        return 0.0
    
    try:
        vectorizer = TfidfVectorizer().fit_transform([text1, text2])
        vectors = vectorizer.toarray()
        similarity = cosine_similarity(vectors)
        return float(similarity[0][1] * 100)
    except Exception as e:
        print(f"Similarity Calculation Error: {e}")
        return 0.0

def evaluate_submission(student_text, model_answer, previous_submissions=[]):
    """
    Deterministic evaluation of a student submission.
    
    Args:
        student_text (str): The student's answer.
        model_answer (str): The expected correct answer.
        previous_submissions (list): List of strings from other students (for plagiarism).
        
    Returns:
        dict: {
            "content_similarity": float,
            "plagiarism_score": float,
            "ai_grade": float,
            "feedback": str
        }
    """
    
    # 1. Normalize
    cleaned_student = normalize_text(student_text)
    cleaned_model = normalize_text(model_answer)
    cleaned_previous = [normalize_text(txt) for txt in previous_submissions if txt]
    
    if not cleaned_student:
        return {
            "content_similarity": 0.0,
            "plagiarism_score": 0.0,
            "ai_grade": 0.0,
            "feedback": "Submission appears empty or contains no valid text."
        }

    # 2. Content Relevance
    content_similarity = calculate_similarity(cleaned_student, cleaned_model)
    
    # 3. Plagiarism Detection
    plagiarism_score = 0.0
    if cleaned_previous:
        # Check against each previous submission and find max similarity
        # Ideally, we would vectorize all at once for efficiency, but loop is fine for <1000 docs
        
        # Optimization: Vectorize (student + all_previous) at once
        try:
            all_documents = [cleaned_student] + cleaned_previous
            tfidf_matrix = TfidfVectorizer().fit_transform(all_documents)
            
            # Compute cosine similarity between student (index 0) and others (indices 1..)
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            if len(cosine_similarities) > 0:
                plagiarism_score = float(np.max(cosine_similarities) * 100)
        except Exception as e:
            print(f"Plagiarism Check Error: {e}")
            plagiarism_score = 0.0

    # 4. Generate AI Grade
    # Formula: 0.7 * content + 0.3 * (100 - plagiarism)
    # Ensure non-negative
    raw_grade = (0.7 * content_similarity) + (0.3 * (100 - plagiarism_score))
    ai_grade = round(max(0, min(100, raw_grade)), 2)
    
    # 5. Generate Feedback
    feedback = ""
    if content_similarity >= 80:
        feedback = "Answer demonstrates strong conceptual understanding."
    elif content_similarity >= 55:
        feedback = "Answer is partially correct with scope for improvement."
    else:
        feedback = "Answer lacks sufficient relevance to the expected answer."
        
    if plagiarism_score > 30:
        feedback += f" Warning: High similarity ({round(plagiarism_score, 1)}%) with other submissions detected."

    return {
        "content_similarity": round(content_similarity, 2),
        "plagiarism_score": round(plagiarism_score, 2),
        "ai_grade": ai_grade,
        "feedback": feedback
    }
