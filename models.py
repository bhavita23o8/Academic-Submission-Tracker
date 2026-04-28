from extensions import db
from datetime import datetime

class Class(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    students = db.relationship('User', backref='student_class', lazy=True)
    assignments = db.relationship('Assignment', backref='target_class', lazy=True)
    teacher_mappings = db.relationship('TeacherClassMapping', backref='class_entity', lazy=True)

class TeacherClassMapping(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=False)
    role_type = db.Column(db.String(20), nullable=False) # 'subject_teacher', 'class_teacher'
    subject_name = db.Column(db.String(100), nullable=True) # Required if role_type is subject_teacher

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    roll_number = db.Column(db.String(50), nullable=True) # Nullable for Admin/Faculty
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # 'student', 'teacher', 'admin'
    
    # New Fields
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=True) # For Students
    
    # Relationships
    class_mappings = db.relationship('TeacherClassMapping', backref='teacher', lazy=True)
    # submissions = db.relationship('Submission', backref='student', lazy=True) # Add later if needed

class Assignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    deadline = db.Column(db.DateTime, nullable=False)
    
    # Linking
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=False)
    subject_teacher_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    subject_name = db.Column(db.String(100), nullable=False)
    
    created_at = db.Column(db.DateTime, default=datetime.now)
    filename = db.Column(db.String(200), nullable=True) # Attachment file
    faculty_username = db.Column(db.String(100), nullable=True) # Legacy, keeping optional

class Activity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) # Changed from username
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=False)
    
    title = db.Column(db.String(200), nullable=False)
    type = db.Column(db.String(50), nullable=False) # Sport, Cultural, Internship, etc.
    description = db.Column(db.Text, nullable=True)
    date = db.Column(db.Date, nullable=False)
    proof_filename = db.Column(db.String(200), nullable=True)
    status = db.Column(db.String(20), default='pending') # approved, pending
    
    # Legacy field fallback (optional)
    student_username = db.Column(db.String(100), nullable=True)

class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) # Changed from user_id string
    assignment_id = db.Column(db.Integer, db.ForeignKey('assignment.id'), nullable=False)
    assignment = db.relationship('Assignment', backref='submissions')
    
    filename = db.Column(db.String(200), nullable=False)
    text = db.Column(db.Text, nullable=False)
    similarity = db.Column(db.Float, nullable=False)
    
    timestamp = db.Column(db.DateTime, default=datetime.now)
    status = db.Column(db.String(50), default='pending')  # pending, reviewed
    review = db.Column(db.Text, nullable=True)  # Faculty review comments
    
    # AI & Flagging
    ai_feedback = db.Column(db.Text, nullable=True)
    ai_grade = db.Column(db.String(10), nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)
    late_status = db.Column(db.Boolean, default=False) # Renamed from is_late for consistency
    risk_flag = db.Column(db.Boolean, default=False)
    is_deleted = db.Column(db.Boolean, default=False)
    
    ocr_extracted = db.Column(db.Boolean, default=False)
    
    # Legacy field
    user_id = db.Column(db.String(100), nullable=True)
