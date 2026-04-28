import sys
import time

print("Starting import debugging...", flush=True)

def try_import(module_name):
    print(f"Importing {module_name}...", end="", flush=True)
    start = time.time()
    try:
        __import__(module_name)
        print(f" Done ({time.time() - start:.2f}s)", flush=True)
    except Exception as e:
        print(f" Failed: {e}", flush=True)

try_import("cv2")
try_import("numpy")
try_import("PIL")
try_import("pdfplumber")
try_import("pytesseract")
try_import("sklearn.feature_extraction.text")
try_import("sklearn.metrics.pairwise")
try_import("ocr_service")
try_import("evaluation_service")
try_import("ai_service")
try_import("models")
try_import("app") # This is likely where it hangs if it's app initialization logic

print("Import debugging finished.", flush=True)
