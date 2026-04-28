# 🎓 A Novel Smart Academic Workflow System for Submission Tracking and Performance Analytics 

A comprehensive **Assignment Submission & Grading System** enhanced with **Artificial Intelligence**. This project moves beyond simple text matching by using **Deep Learning (Sentence-BERT)** to detect semantic plagiarism (paraphrasing).

## 🚀 Key Features

### 🧠 1. Advanced Plagiarism Detection
- **Technology**: Uses **Sentence-Transformers (SBERT)** to create semantic vector embeddings.
- **Benefit**: Detects **paraphrased content**, rewritten sentences, and synonymous phrasing that simple text-matchers miss.
- **Visualization**: Comparison heatmaps and similarity scores.

### 📊 2. Visual Analytics Dashboard
- **Faculty Dashboard**: Interactive charts (Chart.js) showing:
  - Plagiarism Risk Distribution (Safe vs. Critical).
  - Workload Status (Pending vs. Reviewed).
- **Student Dashboard**: Progress line chart tracking academic integrity over time.

### 📄 3. Official PDF Reports
- **Certification**: Students can download an official **Academic Submission Report**.
- **Content**: Includes submission metadata, plagiarism score, AI estimated grade, and constructive feedback.
- **Format**: Professional PDF certificate generated on-the-fly.

### 🤖 4. AI Grading & Feedback
- **Automated Feedback**: Provides instant constructive criticism on submissions.
- **Grading**: Estimates a grade (0-100) based on content quality.

---

## 🛠️ Tech Stack
- **Backend**: Python, Flask, SQLAlchemy
- **AI/ML**: Sentence-Transformers (BERT), PyTorch, Scikit-Learn, TF-IDF, Pytesseract 
- **Frontend**: HTML5, CSS3 (Glassmorphism), Chart.js
- **PDF Generation**: ReportLab
- **Database**: SQLite

---

## ⚙️ Setup & Installation

1.  **Clone/Download the repository**.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: This installs `sentence-transformers`, which may take a moment)*

3.  **Run the Application**:
    ```bash
    python app.py
    ```

4.  **Access the App**:
    Open your browser to: `http://127.0.0.1:5000`

---

## 🔐 Default Login Credentials
The system comes pre-seeded with these accounts:

| Role | Username | Password |
| :--- | :--- | :--- |
| 👨‍🏫 **Faculty** | `faculty` | `password` |
| 👨‍🎓 **Student** | `student1` | `password` |
| 👨‍🎓 **Student** | `student2` | `password` |

---

## 📂 Project Structure
- `app.py`: Main Flask application.
- `ml_engine.py`: Singleton class handling the Deep Learning model.
- `ai_service.py`: Helper for OCR and AI content analysis.
- `templates/`: HTML frontends for Dashboards.
- `tests/`: Validation scripts for ML and PDF generation.
