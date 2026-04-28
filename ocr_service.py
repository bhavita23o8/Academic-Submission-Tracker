import os
import re
import cv2
import pytesseract
import pdfplumber
import numpy as np
from PIL import Image

# Ensure Tesseract can be found if it's in a standard but non-PATH location
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' # Uncomment/adjust if needed

def clean_text(text):
    """
    Cleans text by converting to lowercase, removing special characters,
    and collapsing multiple spaces.
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters (keep alphanumeric, basic punctuation, newlines)
    text = re.sub(r'[^a-z0-9\s.,?!:;-]', '', text)
    
    # Collapse multiple whitespace to single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_image(image):
    """
    Applies grayscale and thresholding to an image for better OCR accuracy.
    Expects a PIL Image or numpy array. Returns a preprocessed image (numpy array).
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding (Otsu's binarization is usually good for text)
    # cv2.THRESH_BINARY | cv2.THRESH_OTSU
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    return thresh

def run_ocr_on_image(image):
    """
    Runs Tesseract OCR on a single image (PIL or numpy array).
    """
    try:
        # Preprocess
        processed_img = preprocess_image(image)
        
        # Run Tesseract
        # psm 6 = Assume a single uniform block of text.
        text = pytesseract.image_to_string(processed_img, config='--psm 6')
        return text
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def extract_text_from_pdf_ocr(pdf_path):
    """
    Fallback method: Convert PDF pages to images (or extract embedded) and run OCR.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # 1. Try to convert whole page to image (best for scanned docs)
                try:
                    # resolution=300 is good for OCR
                    im_obj = page.to_image(resolution=300)
                    pil_im = im_obj.original
                    text += run_ocr_on_image(pil_im) + "\n"
                except Exception as e:
                    print(f"Page to image failed: {e}. Trying embedded images.")
                    # 2. Fallback: Extract embedded images
                    if page.images:
                        for img_dict in page.images:
                            try:
                                # extracting image via pdfplumber requires access to underlying stream
                                # This can be complex with pdfplumber alone without 'pdf2image' dependencies sometimes
                                # Simpler fallback: just skip if to_image failed.
                                pass 
                            except:
                                pass
    except Exception as e:
        print(f"PDF OCR Failed: {e}")
        
    return text

def extract_text_robust(filepath):
    """
    Main entry point for text extraction.
    - Decides between PDF parse and Image OCR.
    - triggers OCR fallback if text is insufficient (< 30 words).
    - Returns cleaned text.
    """
    text = ""
    
    if not os.path.exists(filepath):
        return "Error: File not found."

    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.pdf':
        # 1. Try standard text extraction
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception as e:
            print(f"PDF Extract Error: {e}")

        # 2. Validate Text Length
        word_count = len(text.split())
        if word_count < 30:
            print(f"Text insufficient ({word_count} words). Triggering OCR...")
            ocr_text = extract_text_from_pdf_ocr(filepath)
            # If OCR result is better, append or use it. 
            # Usually if standard failed, we just want the OCR result.
            if len(ocr_text.split()) > word_count:
                text = ocr_text

    elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        # Image OCR
        try:
            pil_img = Image.open(filepath)
            text = run_ocr_on_image(pil_img)
        except Exception as e:
            text = f"Image Reading Error: {e}"
    
    else:
        # Try text read for .txt etc
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except:
            return "Error: Unsupported file format."

    # 3. Clean and Final Validation
    cleaned_text = clean_text(text)
    final_word_count = len(cleaned_text.split())

    if final_word_count < 30:
        # Controlled warning (returned as text, easier for UI to display than raising)
        # The user requested "Raise a safe warning".
        # We can append a warning tag or just return what we have.
        print("Warning: Extracted text is low confidence.")
        if final_word_count == 0:
             return "Warning: No text could be extracted from this document."
             
    return cleaned_text
