from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import os
from dotenv import load_dotenv

load_dotenv()

# Optional Dependencies with Graceful Degradation
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

import ocr_service
import evaluation_service

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from datetime import datetime
    import matplotlib
    matplotlib.use('Agg') # Set backend to non-interactive
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import io
import base64
import difflib
from auth import auth  # Import the auth Blueprint
from extensions import db, bcrypt
from extensions import db, bcrypt
from models import User, Submission, Assignment, Activity, Class, TeacherClassMapping
from ai_service import perform_ocr
# from ml_engine import ml_engine
ml_engine = None
import re

from utils.security import encrypt_data, decrypt_data
from utils.auth_helpers import get_teacher_mappings, can_assign, can_review_activity, can_view_assignment
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = '6a292badcfc45b6e934d07210dbd2d07'

app.config['UPLOAD_FOLDER'] = 'uploads/original'
app.config['TEXT_FOLDER'] = 'uploads/text'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///submissions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEXT_FOLDER'], exist_ok=True)

db.init_app(app)
bcrypt.init_app(app)

with app.app_context():
    db.create_all()

# Register the auth Blueprint
app.register_blueprint(auth)

# --------- TEXT EXTRACTION (OCR + PDF + TXT) ----------
def extract_text_from_file(filepath):
    text = ""
    temp_filepath = filepath + ".temp"
    try:
        # Decrypt file to temporary location
        with open(filepath, 'rb') as f:
            encrypted_data = f.read()
        
        try:
            decrypted_data = decrypt_data(encrypted_data)
        except Exception:
            # Fallback for unencrypted legacy files
            decrypted_data = encrypted_data

        with open(temp_filepath, 'wb') as f:
            f.write(decrypted_data)

        # Use the Robust OCR Service
        text = ocr_service.extract_text_robust(temp_filepath)

    except Exception as e:
        text = f"Error extracting text: {str(e)}"
    finally:
        # Clean up temp decrypted file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            
    return text.strip()

# --------- AI DETECTION HELPER ----------
def detect_ai_content(text):
    """
    Detect potential AI-generated content using multiple heuristics
    Returns a score from 0-1 (1 = likely AI-generated)
    """
    if not text or len(text.strip()) < 50:
        return 0.0

    score = 0.0
    factors = 0

    # 1. Sentence Length Variation (AI has more uniform sentences)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) > 3:
        lengths = [len(s.split()) for s in sentences]
        if lengths and np:
            variation = np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
            if variation < 0.3:  # Low variation suggests AI
                score += 0.4
        factors += 1

    # 2. Lexical Diversity (AI uses more varied vocabulary)
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) > 20:
        unique_words = len(set(words))
        diversity = unique_words / len(words)
        if diversity > 0.7:  # High diversity might indicate AI
            score += 0.3
        factors += 1

    # 3. Generic Academic Phrases (AI often uses generic transitions)
    generic_phrases = ['in conclusion', 'furthermore', 'moreover', 'additionally',
                      'it is important to note', 'research shows', 'studies indicate']
    generic_count = sum(1 for phrase in generic_phrases if phrase in text.lower())
    if generic_count > 2:
        score += 0.3
    factors += 1

    # 4. Word repetition patterns (AI tends to repeat certain words less naturally)
    word_freq = {}
    for word in words:
        if len(word) > 3:  # Only consider meaningful words
            word_freq[word] = word_freq.get(word, 0) + 1

    # Check for over-repetition of certain words
    max_freq = max(word_freq.values()) if word_freq else 0
    if max_freq > len(words) * 0.05:  # If any word appears more than 5% of total words
        score += 0.2
    factors += 1

    return min(score / factors if factors > 0 else 0.0, 1.0)

# --------- PLAGIARISM CHECKER ----------
def calculate_similarity(new_text, existing_texts):
    if not existing_texts or not TfidfVectorizer:
        return []
    all_texts = existing_texts + [new_text]
    try:
        vectorizer = TfidfVectorizer().fit_transform(all_texts)
        similarity_matrix = cosine_similarity(vectorizer[-1], vectorizer[:-1])
        return similarity_matrix.flatten()
    except Exception:
        return []

def generate_heatmap_text(extracted_text, existing_texts):
    if not extracted_text or not existing_texts:
        return "", 0, 0

    similarity_scores = []
    for text in existing_texts:
        # Use ML Engine for semantic similarity
        if ml_engine:
            ratio = ml_engine.calculate_similarity(extracted_text, text)
        else:
            ratio = 0 # Fallback
        similarity_scores.append(ratio)

    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0

    # AI Detection
    ai_score = detect_ai_content(extracted_text)

    # 🎨 Create enhanced heatmap visualization with AI detection
    heatmap_base64 = ""
    if plt:
        try:
            # Use non-interactive backend to avoid GUI issues
            plt.switch_backend('Agg')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2))
        
            # Plagiarism heatmap
            ax1.imshow([[avg_similarity]], cmap='RdYlGn_r', vmin=0, vmax=1)
            ax1.set_title(f"Plagiarism: {avg_similarity*100:.1f}%")
            ax1.axis('off')
        
            # AI detection heatmap
            ax2.imshow([[ai_score]], cmap='Purples', vmin=0, vmax=1)
            ax2.set_title(f"AI Content: {ai_score*100:.1f}%")
            ax2.axis('off')
        
            plt.tight_layout()
        
            # Convert the plot to base64 so it can be shown on webpage
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            heatmap_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
        except Exception:
            heatmap_base64 = "" # Fail silently for heatmap

    return heatmap_base64, avg_similarity, ai_score

    return heatmap_base64, avg_similarity, ai_score

@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    
    if request.method == 'POST':
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']

        user = User.query.filter_by(username=session['user']).first()

        if not user or not bcrypt.check_password_hash(user.password_hash, current_password):
            flash('Incorrect current password', 'danger')
        elif new_password != confirm_password:
            flash('New passwords do not match', 'danger')
        elif len(new_password) < 8:
             flash('Password must be at least 8 characters long', 'danger')
        else:
            user.password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')
            db.session.commit()
            flash('Password changed successfully!', 'success')
            return redirect(url_for('index'))
            
    return render_template('change_password.html')

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    if session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif session.get('role') == 'faculty':
        return redirect(url_for('faculty_dashboard'))
    else:
        return redirect(url_for('student_dashboard'))
    

# faculty_dashboard 

@app.route('/faculty_dashboard')
def faculty_dashboard():
    if 'user' not in session or session.get('role') != 'faculty':
        return redirect(url_for('auth.login'))
    
    current_user = User.query.filter_by(username=session['user']).first()
    if not current_user:
        session.clear()
        return redirect(url_for('auth.login'))
        
    search_query = request.args.get('q', '')
    assignment_id = request.args.get('assignment_id')
    class_id_filter = request.args.get('class_id')
    
    # Get all mappings for this teacher
    mappings = get_teacher_mappings(current_user.id)
    
    # Filter submissions/assignments based on derived access
    # Logic: 
    # 1. Subject Teachers see assignments they created for their class/subject.
    # 2. Class Teachers see all activities for their class.
    
    # Collect IDs for filtering
    subject_class_ids = [m.class_id for m in mappings if m.role_type == 'subject_teacher']
    class_teacher_class_ids = [m.class_id for m in mappings if m.role_type == 'class_teacher']
    
    # Base query for submissions: 
    # Show submissions for assignments created by THIS teacher (Subject Teacher role)
    # OR show submissions for ANY assignment in classes where they are Class Teacher?
    # User said: "Class Teachers have holistic access to their specific class."
    # So Class Teachers should see EVERYTHING for that class.
    
    # 1. Submissions for assignments created by me (Subject Teacher)
    # 2. Submissions for assignments in classes where I am Class Teacher
    
    submissions_query = Submission.query.join(Assignment).filter(
        db.or_(
            Assignment.subject_teacher_id == current_user.id,
            Assignment.class_id.in_(class_teacher_class_ids)
        )
    )

    # Filter by assignment if provided
    current_assignment = None
    if assignment_id:
        assign = Assignment.query.get(assignment_id)
        # Check access
        if assign and (assign.subject_teacher_id == current_user.id or assign.class_id in class_teacher_class_ids):
            submissions_query = submissions_query.filter(Submission.assignment_id == assignment_id)
            current_assignment = assign
    
    # Filter by search string
    if search_query:
        # User ID is now Int/FK, but student dashboard might display names.
        # We need to join User to search by username
        submissions_query = submissions_query.join(User, Submission.student_id == User.id).filter(User.username.ilike(f'%{search_query}%'))
        
    submissions = submissions_query.filter(Submission.is_deleted == False).all()
        
    # Assignments: Created by me ONLY (to avoid clutter from other teachers in same class)
    assignments = Assignment.query.filter(
        Assignment.subject_teacher_id == current_user.id
    ).order_by(Assignment.created_at.desc()).all()
    
    # --- Analytics & Stats ---
    total_submissions = len(submissions)
    pending_count = sum(1 for s in submissions if s.status == 'pending')
    reviewed_count = total_submissions - pending_count
    
    # Plagiarism Risk Distribution
    low_risk = sum(1 for s in submissions if s.similarity < 0.3)
    medium_risk = sum(1 for s in submissions if 0.3 <= s.similarity < 0.6)
    high_risk = sum(1 for s in submissions if s.similarity >= 0.6)
    
    risk_data = [low_risk, medium_risk, high_risk]
    status_data = [pending_count, reviewed_count]

    # --- Activity Review (Class Teacher Only) ---
    activities = []
    if class_teacher_class_ids:
        activities = Activity.query.filter(
            Activity.class_id.in_(class_teacher_class_ids),
            Activity.status == 'pending'
        ).all()
    
    return render_template('index.html', 
                           submissions=submissions, 
                           assignments=assignments,
                           activities=activities,
                           current_assignment=current_assignment,
                           risk_data=risk_data,
                           status_data=status_data,
                           search_query=search_query,
                           mappings=mappings, # Pass all mappings to view
                           now=datetime.now())


@app.route('/admin_dashboard')
def admin_dashboard():
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    
    users = User.query.all()
    student_count = User.query.filter_by(role='student').count()
    faculty_count = User.query.filter_by(role='faculty').count()
    submission_count = Submission.query.count()
    
    classes = Class.query.all()
    
    return render_template('admin_dashboard.html', 
                           users=users, 
                           student_count=student_count, 
                           faculty_count=faculty_count, 
                           submission_count=submission_count,
                           classes=classes)

@app.route('/admin/create_class', methods=['POST'])
def admin_create_class():
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
        
    class_name = request.form.get('class_name')
    if class_name:
        if Class.query.filter_by(name=class_name).first():
             flash('Class already exists!', 'danger')
        else:
            new_class = Class(name=class_name)
            db.session.add(new_class)
            db.session.commit()
            flash(f'Class {class_name} created!', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/create_user', methods=['POST'])
def admin_create_user():
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    
    username = request.form['username']
    password = request.form['password']
    role = request.form['role']
    roll_number = request.form.get('roll_number')
    
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        flash('Username already exists!', 'danger')
        return redirect(url_for('admin_dashboard'))

    # Check roll number uniqueness if provided
    if roll_number:
        existing_roll = User.query.filter_by(roll_number=roll_number).first()
        if existing_roll:
             flash('Roll Number already exists!', 'danger')
             return redirect(url_for('admin_dashboard'))
        
    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, password_hash=hashed_pw, role=role, roll_number=roll_number)
    
    # Handle optional class assignment for students
    if role == 'student':
        class_id = request.form.get('student_class')
        if class_id:
            new_user.class_id = int(class_id)
            
    db.session.add(new_user)
    db.session.flush() # Flush to get ID
    
    # Handle Teacher Mappings
    # (Removed to prevent duplication. Use 'Assign Teacher' section instead.)

    db.session.commit()
    
    flash(f'User {username} created successfully!', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/assign_teacher', methods=['POST'])
def admin_assign_teacher():
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
        
    teacher_id = request.form.get('teacher_id')
    class_id = request.form.get('class_id')
    role_type = request.form.get('role_type') # subject_teacher, class_teacher
    subject_name = request.form.get('subject_name') # Optional
    
    if not (teacher_id and class_id and role_type and subject_name):
        flash("All fields including Subject Name are required.", "danger")
        return redirect(url_for('admin_dashboard'))
        
    # Constraint: One Class Teacher per Class
    if role_type == 'class_teacher':
        existing_ct = TeacherClassMapping.query.filter_by(class_id=int(class_id), role_type='class_teacher').first()
        if existing_ct:
            flash("This class already has a Class Teacher assigned!", "danger")
            return redirect(url_for('admin_dashboard'))

    # Create Mapping
    new_mapping = TeacherClassMapping(
        teacher_id=int(teacher_id),
        class_id=int(class_id),
        role_type=role_type,
        subject_name=subject_name
    )
    db.session.add(new_mapping)
    db.session.commit()
    flash("Teacher assigned successfully!", "success")
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    if 'user' not in session or session.get('role') != 'admin':
        return redirect(url_for('auth.login'))
    
    user_to_delete = User.query.get_or_404(user_id)
    
    # Prevent Admin from deleting themselves
    if user_to_delete.username == session['user']:
        flash("You cannot delete your own admin account!", "danger")
        return redirect(url_for('admin_dashboard'))
        
    try:
        # Delete associated data if necessary
        db.session.delete(user_to_delete)
        db.session.commit()
        flash(f"User {user_to_delete.username} deleted successfully.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting user: {str(e)}", "danger")
        
    return redirect(url_for('admin_dashboard'))

@app.route('/create_assignment', methods=['GET', 'POST'])
def create_assignment():
    if 'user' not in session or session.get('role') != 'faculty':
        return redirect(url_for('auth.login'))
    
    current_user = User.query.filter_by(username=session['user']).first()
    if not current_user:
        session.clear()
        return redirect(url_for('auth.login'))
    
    # Get mappings where they are subject teacher
    mappings = get_teacher_mappings(current_user.id)
    mappings = get_teacher_mappings(current_user.id)
    # Allow both Subject Teachers and Class Teachers to create assignments
    valid_mappings = [m for m in mappings if m.role_type in ['subject_teacher', 'class_teacher']]
    
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        deadline_str = request.form['deadline']
        
        # From form, we need class_id AND subject_name (since a teacher might teach multiple subjects in same class?)
        # Or we select from "Mapping" which is unique per class/subject.
        # Let's assume UI sends `mapping_id` to simplify selection.
        mapping_id = request.form.get('mapping_id')
        
        # Validation
        try:
            deadline = datetime.strptime(deadline_str, '%Y-%m-%dT%H:%M')
            mapping = TeacherClassMapping.query.get(int(mapping_id))
        except (ValueError, TypeError):
            flash('Invalid Input', 'danger')
            flash('Invalid Input', 'danger')
            return render_template('assignment_create.html', mappings=valid_mappings)

        # Check Permission
        if not mapping or mapping.teacher_id != current_user.id:
             flash('Unauthorized: Invalid class/subject selection.', 'danger')
             return redirect(url_for('faculty_dashboard'))

        file = request.files.get('file')
        filename = None
        if file and file.filename != '':
            filename = file.filename
            filepath = os.path.join('uploads/assignments', filename)
            
            # Encrypt assignment file
            file_data = file.read()
            encrypted_data = encrypt_data(file_data)
            with open(filepath, 'wb') as f:
                f.write(encrypted_data)

        new_assign = Assignment(
            title=title, 
            description=description, 
            deadline=deadline, 
            faculty_username=session['user'], # Keep for legacy
            subject_teacher_id=current_user.id,
            subject_name=mapping.subject_name if mapping.subject_name else "Class Activity", # Fallback for Class Teachers
            class_id=mapping.class_id,
            filename=filename
        )
        db.session.add(new_assign)
        db.session.commit()
        flash('Assignment created successfully!', 'success')
        return redirect(url_for('faculty_dashboard'))
    
    return render_template('assignment_create.html', mappings=valid_mappings)

@app.route('/review_submission/<int:submission_id>', methods=['GET', 'POST'])
def review_submission(submission_id):
    if 'user' not in session or session.get('role') != 'faculty':
        return redirect(url_for('auth.login'))

    submission = Submission.query.get_or_404(submission_id)

    if request.method == 'POST':
        status = request.form['status']
        review = request.form.get('review', '')

        submission.status = status
        submission.review = review
        db.session.commit()

        flash("Submission reviewed successfully!", "success")
        return redirect(url_for('faculty_dashboard'))

    return render_template('review_submission.html', submission=submission)


# student_dashboard 

@app.route('/student_dashboard')
def student_dashboard():
    if 'user' not in session or session.get('role') != 'student':
        return redirect(url_for('auth.login'))
        
    current_user = User.query.filter_by(username=session['user']).first()
    if not current_user:
        session.clear()
        return redirect(url_for('auth.login'))
    
    submissions = Submission.query.filter_by(user_id=session['user'], is_deleted=False).all()
    
    # Filter assignments by Student's Class ID
    if current_user.class_id:
        pending_assignments = Assignment.query.filter(
            Assignment.deadline > datetime.now(),
            Assignment.class_id == current_user.class_id
        ).all()
    else:
        pending_assignments = [] # No class assigned, no assignments seen
        
    # Fetch Student Activities
    activities = Activity.query.filter_by(student_id=current_user.id).order_by(Activity.date.desc()).all()
        
    return render_template('student_dashboard.html', 
                           submissions=submissions, 
                           assignments=pending_assignments, 
                           activities=activities,
                           now=datetime.now())


@app.route('/activities', methods=['GET', 'POST'])
def activities():
    if 'user' not in session or session.get('role') != 'student':
        return redirect(url_for('auth.login'))
    
    current_user = User.query.filter_by(username=session['user']).first()
    if not current_user:
        session.clear()
        return redirect(url_for('auth.login'))
    
    if request.method == 'POST':
        title = request.form['title']
        type = request.form['type']
        date_str = request.form['date']
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        description = request.form['description']
        
        # Simple proof handling - in real app upload file
        proof_file = None
        if 'proof' in request.files:
            file = request.files['proof']
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                # Ensure unique filename to prevent overwrites (timestamp)
                filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                proof_file = filename
        
        if not current_user.class_id:
             flash('You must be assigned to a class to log activities.', 'danger')
             return redirect(url_for('activities'))
             
        new_activity = Activity(
            student_id=current_user.id, # New FK
            class_id=current_user.class_id, # New FK
            title=title, 
            type=type, 
            date=date, 
            description=description, 
            proof_filename=proof_file,
            student_username=session['user'] # Legacy fallback
        )
        db.session.add(new_activity)
        db.session.commit()
        flash('Activity logged!', 'success')
        
    user_activities = Activity.query.filter_by(student_id=current_user.id).all()
    return render_template('activities.html', activities=user_activities)

@app.route('/review_activity/<int:activity_id>', methods=['POST'])
def review_activity(activity_id):
    if 'user' not in session or session.get('role') != 'faculty':
        return redirect(url_for('auth.login'))
        
    activity = Activity.query.get_or_404(activity_id)
    
    # Check if current user is the Class Teacher for this activity's class
    # We need to query TeacherClassMapping
    is_class_teacher = TeacherClassMapping.query.filter_by(
        teacher_id=User.query.filter_by(username=session['user']).first().id,
        class_id=activity.class_id,
        role_type='class_teacher'
    ).first()
    
    if not is_class_teacher:
        flash("You are not authorized to review activities for this class.", "danger")
        return redirect(url_for('faculty_dashboard'))
        
    action = request.form.get('action') # 'approve' or 'reject'
    
    if action == 'approve':
        activity.status = 'approved'
        flash("Activity approved successfully.", "success")
    elif action == 'reject':
        activity.status = 'rejected'
        flash("Activity rejected.", "warning")
        
    db.session.commit()
    return redirect(url_for('faculty_dashboard'))

@app.route('/download_proof/<int:activity_id>')
def download_proof(activity_id):
    if 'user' not in session:
        return redirect(url_for('auth.login'))
        
    activity = Activity.query.get_or_404(activity_id)
    
    # Check authorization (Student owner or Class Teacher)
    current_user_id = User.query.filter_by(username=session['user']).first().id
    
    is_owner = (activity.student_id == current_user_id)
    is_teacher = False
    
    if session.get('role') == 'faculty':
         is_teacher = TeacherClassMapping.query.filter_by(
            teacher_id=current_user_id,
            class_id=activity.class_id,
            role_type='class_teacher'
        ).first() is not None
        
    if not (is_owner or is_teacher):
        flash("Unauthorized access to proof.", "danger")
        return redirect(url_for('index'))

    if not activity.proof_filename:
        flash("No proof file attached.", "warning")
        return redirect(url_for('index'))
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], activity.proof_filename)
    if not os.path.exists(filepath):
        flash("File not found on server. This activity may have been created before file storage was enabled.", "danger")
        return redirect(url_for('index'))
        
    try:
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        flash(f"Error downloading proof: {str(e)}", "danger")
        return redirect(url_for('index'))
    

    # upload  

@app.route('/upload', methods=['GET', 'POST'])
@app.route('/upload/<int:assignment_id>', methods=['GET', 'POST'])
def upload(assignment_id=None):
    if 'user' not in session:
        return redirect(url_for('auth.login'))

    current_user = User.query.filter_by(username=session['user']).first()
    if not current_user:
        session.clear()
        return redirect(url_for('auth.login'))
    
    if request.method == 'POST':
        if not assignment_id:
             assignment_id = request.form.get('assignment_id')

        file = request.files.get('file')
        if not file or file.filename == '':
            flash("No file selected!", "danger")
            return redirect(url_for('upload', assignment_id=assignment_id))

        # Check file extension - now includes .txt
        allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.txt'}
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            flash("Invalid file type. Only PDF, image files, and text files are allowed!", "danger")
            return redirect(url_for('upload', assignment_id=assignment_id))

        # Encrypt and Save File
        file_data = file.read()
        encrypted_data = encrypt_data(file_data)
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        with open(filepath, 'wb') as f:
            f.write(encrypted_data)

        extracted_text = extract_text_from_file(filepath)
        
        # If text extraction failed or empty, try AI/OCR Service
        if not extracted_text.strip() or extracted_text.startswith("Error"):
             extracted_text = perform_ocr(filepath)

        # Get Assignment Description for Context (Model Answer)
        assignment_context = ""
        is_late = False
        if assignment_id:
            assign = Assignment.query.get(assignment_id)
            if assign:
                assignment_context = assign.description or ""
                if datetime.now() > assign.deadline:
                    is_late = True
                    flash("Note: Your submission is past the deadline and marked as Late.", "warning")
        
        # --- DETERMINISTIC EVALUATION ---
        # Fetch previous submissions for plagiarism check
        # We check against ALL submissions to be safe, or filtered by assignment if desired. 
        # Global check is better for detecting cross-assignment copying.
        existing_texts = [s.text for s in Submission.query.all() if s.text]

        # Use Evaluation Service
        eval_result = evaluation_service.evaluate_submission(
            student_text=extracted_text, 
            model_answer=assignment_context, 
            previous_submissions=existing_texts
        )

        # Unpack results
        # Use the service's similarity/plagiarism score
        # Note: existing code used 'similarity' for heatmap. We can keep heatmap generation if needed
        # but usage of 'similarity' field in DB should be consistent. 
        # The DB 'similarity' field likely refers to plagiarism % in this context.
        similarity = eval_result['plagiarism_score'] / 100.0 # DB likely expects float 0-1 or we store as is? 
        # Looking at previous code: `similarity, ai_score = generate_heatmap_text...` 
        # generate_heatmap_text returns similarity as 0-1 float usually in these apps.
        # evaluation_service returns 0-100. Let's convert to 0-1 for DB if that's what it expects.
        # Checking models.py: similarity = db.Column(db.Float)
        # Checking report generation: `score_percent = sim_score * 100`. So DB stores 0.0-1.0.
        
        db_similarity = eval_result['plagiarism_score'] / 100.0
        feedback = eval_result['feedback']
        grade = str(eval_result['ai_grade'])

        # Generate Heatmap and get AI Probability
        heatmap_img, _, ai_prob = generate_heatmap_text(extracted_text, existing_texts)

        # Save submission to DB only after successful processing
        new_entry = Submission(
            filename=file.filename, 
            text=extracted_text, 
            similarity=db_similarity, 
            student_id=current_user.id,
            user_id=session['user'],
            assignment_id=assignment_id,
            ai_feedback=feedback,
            ai_grade=grade,
            ocr_extracted=True,
            late_status=is_late
        )
        db.session.add(new_entry)
        db.session.commit()

        return render_template('result.html', similarity=db_similarity, ai_score=ai_prob, heatmap_img=heatmap_img, feedback=feedback, grade=grade)
    return render_template('upload.html', assignment_id=assignment_id)

@app.route('/delete_submission/<int:submission_id>', methods=['POST'])
def delete_submission(submission_id):
    if 'user' not in session:
        return redirect(url_for('auth.login'))
        
    submission = Submission.query.get_or_404(submission_id)
    
    # Check Ownership
    if submission.user_id != session['user']:
        flash("You are not authorized to delete this submission.", "danger")
        return redirect(url_for('student_dashboard'))
        
    # Check Deadline
    if submission.assignment.deadline and datetime.now() > submission.assignment.deadline:
         flash("Cannot delete submission: Deadline has passed.", "danger")
         return redirect(url_for('student_dashboard'))
         
    # Check Review Status
    if submission.status == 'reviewed':
        flash("Cannot delete submission: Faculty has already reviewed it.", "danger")
        return redirect(url_for('student_dashboard'))
        
    # Soft Delete
    try:
        submission.is_deleted = True
        db.session.commit()
        flash("Submission deleted successfully.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting submission: {str(e)}", "danger")
        
    return redirect(url_for('student_dashboard'))


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO

@app.route('/download_report/<int:submission_id>')
def download_report(submission_id):
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    
    try:
        submission = Submission.query.get_or_404(submission_id)
        
        # Create PDF in memory
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], alignment=1, fontSize=24, spaceAfter=20)
        story.append(Paragraph("Academic Submission Report", title_style))
        story.append(Spacer(1, 12))

        # Metadata
        meta_data = [
            ["Student Username:", submission.user_id],
            ["Filename:", submission.filename],
            ["Submission Date:", submission.timestamp.strftime('%Y-%m-%d %H:%M')],
            ["Status:", submission.status.title()]
        ]
        t = Table(meta_data, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTSIZE', (0,0), (-1,-1), 12),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 20))

        # Plagiarism Score
        # Handle if similarity is None (legacy data)
        sim_score = submission.similarity if submission.similarity is not None else 0.0
        score_percent = sim_score * 100
        color = colors.green if score_percent < 40 else (colors.orange if score_percent < 70 else colors.red)
        
        story.append(Paragraph(f"Plagiarism Similarity Score: <font color='{color}'>{score_percent:.1f}%</font>", 
                               ParagraphStyle('Score', parent=styles['Heading2'], fontSize=18)))
        story.append(Spacer(1, 12))

        # AI Grade & Feedback
        if submission.ai_grade:
            story.append(Paragraph(f"AI Estimated Grade: {submission.ai_grade}", styles['Heading3']))
        
        if submission.ai_feedback:
            import html
            story.append(Spacer(1, 12))
            story.append(Paragraph("AI Feedback:", styles['Heading3']))
            # Clean and Escape for ReportLab
            safe_feedback = html.escape(submission.ai_feedback)
            formatted_feedback = safe_feedback.replace('**', '').replace('* ', '&bull; ').replace('\n', '<br/>')
            
            try:
                story.append(Paragraph(formatted_feedback, styles['BodyText']))
            except Exception:
                story.append(Paragraph(safe_feedback, styles['BodyText']))

        doc.build(story)
        buffer.seek(0)
        
        return send_file(buffer, as_attachment=True, download_name=f"report_{submission.id}.pdf", mimetype='application/pdf')

    except Exception as e:
        import traceback
        error_msg = f"Error generating report:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg) # Log to terminal
        return Response(error_msg, mimetype='text/plain', headers={"Content-Disposition": "attachment;filename=error_log.txt"})
    
@app.route('/download_assignment/<int:assignment_id>')
def download_assignment(assignment_id):
    if 'user' not in session:
        return redirect(url_for('auth.login'))
        
    assignment = Assignment.query.get_or_404(assignment_id)
    if not assignment.filename:
        flash("No file attached to this assignment.", "warning")
        return redirect(url_for('student_dashboard'))
    
    # Decrypt for download
    filepath = os.path.join('uploads/assignments', assignment.filename)
    temp_filepath = filepath + ".temp_dl"
    
    try:
        with open(filepath, 'rb') as f:
            encrypted_data = f.read()
            
        try:
            decrypted_data = decrypt_data(encrypted_data)
        except Exception:
            # Fallback
            decrypted_data = encrypted_data
            
        with open(temp_filepath, 'wb') as f:
            f.write(decrypted_data)
            
        return send_file(temp_filepath, as_attachment=True, download_name=assignment.filename)
    except Exception as e:
        flash(f"Error downloading file: {str(e)}", "danger")
        return redirect(url_for('student_dashboard'))

@app.route('/view_submission_file/<int:submission_id>')
def view_submission_file(submission_id):
    if 'user' not in session or session.get('role') != 'faculty':
        return "Unauthorized", 403
        
    submission = Submission.query.get_or_404(submission_id)
    
    # Decrypt for viewing
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], submission.filename)
    if not os.path.exists(filepath):
        return "File not found", 404

    # We need to serve the decrypted content. 
    # Since send_file usually takes a path or file-like object, we use BytesIO.
    try:
        with open(filepath, 'rb') as f:
            encrypted_data = f.read()
            
        try:
            decrypted_data = decrypt_data(encrypted_data)
        except Exception:
            # Fallback
            decrypted_data = encrypted_data
            
        return send_file(
            io.BytesIO(decrypted_data),
            as_attachment=False,
            download_name=submission.filename,
            mimetype='application/pdf' if submission.filename.endswith('.pdf') else 'text/plain' 
            # Note: Basic mimetype detection. For images, we might need more logic or just let browser detect.
        )
    except Exception as e:
        return f"Error reading file: {str(e)}", 500


# student_details 
@app.route('/student/<username>')
def student_details(username):
    if 'user' not in session or session.get('role') not in ['faculty', 'admin', 'student']:
        return redirect(url_for('auth.login'))
    
    # Restrict students to view only their own profile
    if session.get('role') == 'student' and session.get('user') != username:
        return redirect(url_for('student_dashboard'))
        
    student = User.query.filter_by(username=username).first()
    if not student:
        flash("Student not found!", "danger")
        return redirect(url_for('faculty_dashboard'))
        
    submissions = Submission.query.filter_by(student_id=student.id).order_by(Submission.timestamp.desc()).all()
    # Fetch activities by ID (Foreign Key)
    activities = Activity.query.filter_by(student_id=student.id).order_by(Activity.date.desc()).all()
    
    # Categorize Activities
    co_curricular_types = ['Internship', 'Workshop', 'Technical', 'Paper', 'Project', 'Hackathon', 'Certification']
    extra_curricular_types = ['Sport', 'Cultural', 'Art', 'Volunteer', 'Club', 'Social']
    
    co_curricular = [a for a in activities if a.type in co_curricular_types or (a.type not in extra_curricular_types and 'Tech' in a.type)]
    extra_curricular = [a for a in activities if a not in co_curricular]
    
    # Calculate Data for Charts & Stats
    grades = []
    plagiarism_scores = []
    dates = []
    
    total_grade = 0
    grade_count = 0
    
    for sub in submissions:
        # Parse Grade
        grade_val = 0
        if sub.ai_grade and '/' in sub.ai_grade:
            try:
                grade_val = float(sub.ai_grade.split('/')[0])
                total_grade += grade_val
                grade_count += 1
            except:
                pass
        
        grades.append(grade_val) # Use 0 if pending/error for chart
        plagiarism_scores.append(sub.similarity * 100)
        dates.append(sub.timestamp.strftime('%Y-%m-%d'))
        
    avg_grade = round(total_grade / grade_count, 1) if grade_count > 0 else 0
    avg_plagiarism = round(sum(plagiarism_scores) / len(plagiarism_scores), 1) if plagiarism_scores else 0
    
    # Overall Performance Calculation (Heuristic)
    # 50% Academic (Avg Grade scaled to 50)
    # 25% Co-Curricular (2.5 pts per activity, max 10 activities = 25)
    # 25% Extra-Curricular (2.5 pts per activity, max 10 activities = 25)
    
    academic_score = (avg_grade / 100) * 50
    co_curr_score = min(len(co_curricular) * 2.5, 25)
    extra_curr_score = min(len(extra_curricular) * 2.5, 25)
    
    overall_score = round(academic_score + co_curr_score + extra_curr_score, 1)
    
    # Reverse for Chart (Oldest to Newest)
    grades.reverse()
    plagiarism_scores.reverse()
    dates.reverse()

    return render_template('student_details.html', 
                           student=student, 
                           submissions=submissions, 
                           activities=activities,
                           co_curricular=co_curricular,
                           extra_curricular=extra_curricular,
                           overall_score=overall_score,
                           avg_grade=avg_grade,
                           avg_plagiarism=avg_plagiarism,
                           grades=grades,
                           plagiarism_scores=plagiarism_scores,
                           dates=dates)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
