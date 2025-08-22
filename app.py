import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_LOGGING_VERBOSITY'] = '3'  # Suppress TensorFlow logs
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, session, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_mail import Mail, Message
import bcrypt
from sklearn.preprocessing import StandardScaler
from tf_keras.models import load_model
from transformers import MobileBertTokenizer, TFMobileBertModel
import joblib
import logging
from datetime import datetime, timedelta
from smtplib import SMTPException
import time
from sqlalchemy import Index, update

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
load_dotenv()

# Ensure the instance directory exists
base_dir = os.path.abspath(os.path.dirname(__file__))
instance_dir = os.path.join(base_dir, 'instance')
if not os.path.exists(instance_dir):
    os.makedirs(instance_dir)

# Database configuration with increased timeout
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(instance_dir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'connect_args': {'timeout': 30}}
app.config['SECRET_KEY'] = 'f9bf78b9a18ce6d46a0cd2b0b86df9da'

# Configure Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

mail = Mail(app)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    identifier = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    security_question = db.Column(db.String(200), nullable=False)
    security_answer = db.Column(db.String(200), nullable=False)
    parent_guardian_names = db.Column(db.String(200))
    parent_guardian_contacts = db.Column(db.String(50))
    home_address = db.Column(db.String(200))
    student_age = db.Column(db.Integer)

# Student model
class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(10), nullable=False)
    performance_label = db.Column(db.String(20), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    attendance_rate = db.Column(db.Float, nullable=False)
    assignment_score = db.Column(db.Float, nullable=False)
    quiz_score = db.Column(db.Float, nullable=False)
    exam_score = db.Column(db.Float, nullable=False)
    study_hours_per_week = db.Column(db.Float, nullable=False)
    feedback = db.Column(db.Text, nullable=False)
    access_to_internet = db.Column(db.String(3))
    study_support_at_home = db.Column(db.String(3))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    advice = db.Column(db.Text)

    __table_args__ = (
        Index('idx_student_created_at', 'created_at'),
        Index('idx_student_performance_label', 'performance_label'),
    )

# Complaint model
class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    phone_number = db.Column(db.String(50), nullable=False)
    complaint_text = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load model and components
hybrid_model, scaler, selected_features, tokenizer, mobilebert_model, text_scaler = None, None, None, None, None, None
try:
    required_files = ['hybrid_student_model.h5', 'scaler.save', 'selected_features.joblib', 'text_scaler.save']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}.")

    hybrid_model = load_model('hybrid_student_model.h5')
    scaler = joblib.load('scaler.save')
    selected_features = joblib.load('selected_features.joblib')
    text_scaler = joblib.load('text_scaler.save')
    tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
    mobilebert_model = TFMobileBertModel.from_pretrained('google/mobilebert-uncased')
    logger.info("Model and components loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")

# Encode text with MobileBERT
def encode_text(texts, max_length=64):  # Reduced max_length
    if tokenizer is None or mobilebert_model is None:
        raise ValueError("MobileBERT components not loaded.")
    start_time = time.time()
    inputs = tokenizer(texts, return_tensors="tf", padding=True, truncation=True, max_length=max_length)
    outputs = mobilebert_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    result = outputs.last_hidden_state[:, 0, :].numpy()
    logger.info(f"Text encoding took {time.time() - start_time:.2f} seconds for {len(texts)} texts")
    return result

# Hybrid prediction
def predict_with_hybrid(exam_score, quiz_score, assignment_score, attendance_rate, study_hours, feedback):
    if any(x is None for x in [hybrid_model, scaler, selected_features, text_scaler, tokenizer, mobilebert_model]):
        raise ValueError("Prediction model not available.")
    start_time = time.time()
    numerical_data = {
        'attendance_rate': attendance_rate,
        'assignment_score': assignment_score,
        'quiz_score': quiz_score,
        'exam_score': exam_score,
        'study_hours_per_week': study_hours
    }
    input_df = pd.DataFrame([[numerical_data[feat] for feat in selected_features]], columns=selected_features)
    scaled_input = scaler.transform(input_df)
    reshaped_input = scaled_input.reshape((1, 1, len(selected_features)))

    text_input = encode_text([feedback])
    text_input_scaled = text_scaler.transform(text_input)

    prediction_probs = hybrid_model.predict([reshaped_input, text_input_scaled], verbose=0)
    predicted_label = np.argmax(prediction_probs, axis=1)[0]
    label_map = {0: 'Poor', 1: 'Average', 2: 'Excellent'}
    logger.info(f"Prediction took {time.time() - start_time:.2f} seconds")
    return label_map[predicted_label]

# Personalized message
def get_personalized_message(prediction, username, attendance_rate, study_hours, avg_score):
    messages = []
    if attendance_rate <= 20 or study_hours <= 20:
        messages.append("Your attendance or study hours are low. Attend more classes and study more.")
    elif 25 <= attendance_rate <= 35 or 25 <= study_hours <= 35:
        messages.append("Improve your attendance and study hours for better performance.")
    if prediction == 'Excellent':
        messages.append(f"Excellent work, {username}!")
    elif prediction == 'Average':
        messages.append(f"Average performance, {username}. Keep working hard!")
    else:
        messages.append(f"Poor performance, {username}. Attend classes and seek help.")
    if avg_score >= 70 and (attendance_rate < 40 or study_hours < 40):
        messages.append(f"Good grades, {username}! Improve attendance and study hours.")
    return " ".join(messages)

# Admin advice
def get_admin_advice(student):
    if student.performance_label == 'Poor':
        return (f"Student {student.student_id} is struggling. Recommendations: "
                "schedule a one-on-one meeting, provide tutoring, monitor attendance.")
    elif student.performance_label == 'Average':
        return f"Student {student.student_id} is average. Encourage effort and study tips."
    else:
        return f"Student {student.student_id} is excelling. Provide advanced material."

@app.template_filter('students')
def students_filter(count):
    return f"{count} student{'s' if count != 1 else ''}"

@app.before_request
def log_session_details():
    logger.info(f"Request path: {request.path}, Session: {session.items()}, Auth: {current_user.is_authenticated}")

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('student_dashboard' if current_user.role == 'student' else 'admin_dashboard'))
    logout_user()
    session.clear()
    response = make_response(render_template('index.html'))
    response.set_cookie('session', '', expires=0)
    return response

@app.route('/role_selection')
def role_selection():
    if current_user.is_authenticated:
        return redirect(url_for('student_dashboard' if current_user.role == 'student' else 'admin_dashboard'))
    return render_template('role_selection.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('student_dashboard' if current_user.role == 'student' else 'admin_dashboard'))
    role = request.args.get('role')
    if not role or role not in ['student', 'admin']:
        flash('Invalid role.', 'error')
        return redirect(url_for('role_selection'))
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            identifier = request.form.get('identifier')
            email = request.form.get('email')
            password = request.form.get('password')
            security_question = request.form.get('security_question')
            security_answer = request.form.get('security_answer')
            parent_guardian_names = request.form.get('parent_guardian_names') if role == 'student' else None
            parent_guardian_contacts = request.form.get('parent_guardian_contacts') if role == 'student' else None
            home_address = request.form.get('home_address') if role == 'student' else None
            student_age = request.form.get('student_age') if role == 'student' else None

            if not all([username, identifier, email, password, security_question, security_answer]):
                flash('All fields required.', 'error')
                return redirect(url_for('register', role=role))
            if role == 'student' and not all([parent_guardian_names, parent_guardian_contacts, home_address, student_age]):
                flash('All student fields required.', 'error')
                return redirect(url_for('register', role=role))
            if role == 'admin' and identifier not in ['SAPPS1', 'SAPPS2', 'SAPPS3', 'SAPPS4', 'SAPPS5']:
                flash('Invalid Admin ID.', 'error')
                return redirect(url_for('register', role=role))
            if User.query.filter_by(username=username).first():
                flash('Username exists.', 'error')
                return redirect(url_for('register', role=role))
            if User.query.filter_by(identifier=identifier).first():
                flash('ID exists.', 'error')
                return redirect(url_for('register', role=role))
            if User.query.filter_by(email=email).first():
                flash('Email registered.', 'error')
                return redirect(url_for('register', role=role))

            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            hashed_answer = bcrypt.hashpw(security_answer.lower().encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            user = User(
                username=username, identifier=identifier, email=email, password=hashed_password, role=role,
                security_question=security_question, security_answer=hashed_answer,
                parent_guardian_names=parent_guardian_names, parent_guardian_contacts=parent_guardian_contacts,
                home_address=home_address, student_age=int(student_age) if student_age else None
            )
            db.session.add(user)
            db.session.commit()
            flash('Registration successful!', 'success')
            return redirect(url_for('login', role=role))
        except Exception as e:
            flash(f"Error: {str(e)}", 'error')
            db.session.rollback()
    return render_template('register.html', role=role)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('student_dashboard' if current_user.role == 'student' else 'admin_dashboard'))
    role = request.args.get('role')
    if not role or role not in ['student', 'admin']:
        flash('Invalid role.', 'error')
        return redirect(url_for('role_selection'))
    if request.method == 'POST':
        identifier = request.form.get('identifier')
        password = request.form.get('password')
        user = User.query.filter((User.username == identifier) | (User.identifier == identifier)).first()
        if user and user.role == role and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            login_user(user)
            flash('Logged in!', 'success')
            return redirect(url_for('student_dashboard' if user.role == 'student' else 'admin_dashboard'))
        flash('Invalid credentials.', 'error')
    return render_template('login.html', role=role)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash('Logged out.', 'success')
    response = make_response(redirect(url_for('role_selection')))
    response.set_cookie('session', '', expires=0)
    return response

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for('student_dashboard' if current_user.role == 'student' else 'admin_dashboard'))
    if request.method == 'POST':
        identifier = request.form.get('identifier')
        user = User.query.filter((User.username == identifier) | (User.identifier == identifier)).first()
        if user:
            session['reset_user_id'] = user.id
            return redirect(url_for('verify_security_question'))
        flash('User not found.', 'error')
    return render_template('forgot_password.html')

@app.route('/verify_security_question', methods=['GET', 'POST'])
def verify_security_question():
    if current_user.is_authenticated:
        return redirect(url_for('student_dashboard' if current_user.role == 'student' else 'admin_dashboard'))
    user_id = session.get('reset_user_id')
    if not user_id:
        flash('Session expired.', 'error')
        return redirect(url_for('forgot_password'))
    user = User.query.get(user_id)
    if request.method == 'POST':
        answer = request.form.get('security_answer')
        if bcrypt.checkpw(answer.lower().encode('utf-8'), user.security_answer.encode('utf-8')):
            return redirect(url_for('reset_password'))
        flash('Incorrect answer.', 'error')
    return render_template('verify_security_question.html', question=user.security_question)

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if current_user.is_authenticated:
        return redirect(url_for('student_dashboard' if current_user.role == 'student' else 'admin_dashboard'))
    user_id = session.get('reset_user_id')
    if not user_id:
        flash('Session expired.', 'error')
        return redirect(url_for('forgot_password'))
    user = User.query.get(user_id)
    if request.method == 'POST':
        password = request.form.get('password')
        user.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        db.session.commit()
        session.pop('reset_user_id', None)
        flash('Password reset!', 'success')
        return redirect(url_for('login', role=user.role))
    return render_template('reset_password.html')

@app.route('/student_dashboard', methods=['GET', 'POST'])
@login_required
def student_dashboard():
    if current_user.role != 'student':
        flash('Access denied.', 'error')
        return redirect(url_for('admin_dashboard'))
    
    page = request.args.get('page', 1, type=int)
    per_page = 10
    start_time = time.time()
    predictions = Student.query.filter_by(user_id=current_user.id).order_by(Student.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    logger.info(f"Student dashboard query took {time.time() - start_time:.2f} seconds")

    if request.method == 'POST':
        try:
            student_id = current_user.identifier
            gender = request.form.get('gender')
            if gender not in ['Male', 'Female']:
                raise ValueError("Gender must be 'Male' or 'Female'.")
            age = int(request.form.get('age') or 0)
            if not (10 <= age <= 30):
                raise ValueError("Age must be 10-30.")
            attendance_rate = float(request.form.get('attendance_rate') or 0)
            if not (0 <= attendance_rate <= 100):
                raise ValueError("Attendance rate must be 0-100.")
            assignment_score = float(request.form.get('assignment_score') or 0)
            if not (0 <= assignment_score <= 100):
                raise ValueError("Assignment score must be 0-100.")
            quiz_score = float(request.form.get('quiz_score') or 0)
            if not (0 <= quiz_score <= 100):
                raise ValueError("Quiz score must be 0-100.")
            exam_score = float(request.form.get('exam_score') or 0)
            if not (0 <= exam_score <= 100):
                raise ValueError("Exam score must be 0-100.")
            study_hours = float(request.form.get('study_hours_per_week') or 0)
            if not (0 <= study_hours <= 168):
                raise ValueError("Study hours must be 0-168.")
            feedback = request.form.get('feedback')
            if not feedback:
                raise ValueError("Feedback required.")
            access_to_internet = request.form.get('access_to_internet')
            if access_to_internet not in ['YES', 'NO']:
                raise ValueError("Internet access must be 'YES' or 'NO'.")
            study_support_at_home = request.form.get('study_support_at_home')
            if study_support_at_home not in ['YES', 'NO']:
                raise ValueError("Study support must be 'YES' or 'NO'.")

            prediction = predict_with_hybrid(exam_score, quiz_score, assignment_score, attendance_rate, study_hours, feedback)
            avg_score = (exam_score + quiz_score + assignment_score) / 3

            student = Student(
                student_id=student_id, performance_label=prediction, gender=gender, age=age,
                attendance_rate=attendance_rate, assignment_score=assignment_score, quiz_score=quiz_score,
                exam_score=exam_score, study_hours_per_week=study_hours, feedback=feedback,
                access_to_internet=access_to_internet, study_support_at_home=study_support_at_home,
                user_id=current_user.id, created_at=datetime.utcnow()
            )
            student.advice = get_admin_advice(student)
            db.session.add(student)
            db.session.commit()

            message = get_personalized_message(prediction, current_user.username, attendance_rate, study_hours, avg_score)
            flash(message, 'info')
            return redirect(url_for('student_dashboard'))
        except ValueError as ve:
            flash(f"Input Error: {str(ve)}", 'error')
            db.session.rollback()
        except Exception as e:
            flash(f"Server Error: {str(e)}", 'error')
            db.session.rollback()
    return render_template('student_dashboard.html', predictions=predictions)

@app.route('/submit_complaint', methods=['GET', 'POST'])
@login_required
def submit_complaint():
    if current_user.role != 'student':
        flash('Access denied.', 'error')
        return redirect(url_for('admin_dashboard'))
    if request.method == 'POST':
        try:
            full_name = request.form.get('full_name')
            email = request.form.get('email')
            phone_number = request.form.get('phone_number')
            complaint_text = request.form.get('complaint_text')
            if not all([full_name, email, phone_number, complaint_text]):
                flash('All fields required.', 'error')
                return redirect(url_for('submit_complaint'))
            complaint = Complaint(full_name=full_name, email=email, phone_number=phone_number, complaint_text=complaint_text, user_id=current_user.id)
            db.session.add(complaint)
            db.session.commit()
            flash('Complaint submitted!', 'success')
            return redirect(url_for('student_dashboard'))
        except Exception as e:
            flash(f"Error: {str(e)}", 'error')
            db.session.rollback()
    return render_template('complaint_form.html')

@app.route('/send_complaint_response/<int:complaint_id>', methods=['POST'])
@login_required
def send_complaint_response(complaint_id):
    if current_user.role != 'admin':
        flash('Access denied.', 'error')
        return redirect(url_for('student_dashboard'))
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            complaint = Complaint.query.get_or_404(complaint_id)
            message_body = request.form.get('response_message')
            if not message_body:
                flash('Response message required.', 'error')
                return redirect(url_for('admin_dashboard'))
            msg = Message(subject='Response to Your Complaint', recipients=[complaint.email], body=message_body)
            mail.send(msg)
            flash('Response sent!', 'success')
            return redirect(url_for('admin_dashboard'))
        except SMTPException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            flash(f"Email error: {str(e)}", 'error')
            db.session.rollback()
            return redirect(url_for('admin_dashboard'))
        except Exception as e:
            flash(f"Error: {str(e)}", 'error')
            db.session.rollback()
            return redirect(url_for('admin_dashboard'))

@app.route('/admin_dashboard', methods=['GET', 'POST'])
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Access denied.', 'error')
        return redirect(url_for('student_dashboard'))

    sort = request.args.get('sort', 'newest')
    time_filter = request.args.get('time_filter', 'all')
    page = request.args.get('page', 1, type=int)
    per_page = 10

    start_time = time.time()
    query = Student.query
    if time_filter == '7days':
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        query = query.filter(Student.created_at >= seven_days_ago)
    
    predictions = query.order_by(Student.created_at.desc() if sort == 'newest' else Student.created_at.asc()).paginate(page=page, per_page=per_page, error_out=False)
    logger.info(f"Admin dashboard query took {time.time() - start_time:.2f} seconds")

    # Update advice in batch
    start_time = time.time()
    updates = []
    for record in predictions.items:
        if not hasattr(record, 'advice') or record.advice is None:
            advice = get_admin_advice(record)
            updates.append({'id': record.id, 'advice': advice})
    if updates:
        try:
            db.session.execute(
                update(Student),
                [{'id': u['id'], 'advice': u['advice']} for u in updates]
            )
            db.session.commit()
            logger.info(f"Batch updated advice for {len(updates)} records in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error updating advice: {str(e)}")
            db.session.rollback()

    # Performance counts with no_autoflush
    start_time = time.time()
    with db.session.no_autoflush:
        poor_count = query.filter(Student.performance_label == 'Poor').count()
        average_count = query.filter(Student.performance_label == 'Average').count()
        excellent_count = query.filter(Student.performance_label == 'Excellent').count()
    logger.info(f"Performance counts query took {time.time() - start_time:.2f} seconds")

    complaints = Complaint.query.paginate(page=page, per_page=per_page, error_out=False)

    if request.method == 'POST':
        try:
            student_id = request.form.get('student_id')
            user = User.query.filter_by(identifier=student_id).first()
            user_id = user.id if user and user.role == 'student' else None
            gender = request.form.get('gender')
            if gender not in ['Male', 'Female']:
                raise ValueError("Gender must be 'Male' or 'Female'.")
            age = int(request.form.get('age') or 0)
            if not (10 <= age <= 30):
                raise ValueError("Age must be 10-30.")
            attendance_rate = float(request.form.get('attendance_rate') or 0)
            if not (0 <= attendance_rate <= 100):
                raise ValueError("Attendance rate must be 0-100.")
            assignment_score = float(request.form.get('assignment_score') or 0)
            if not (0 <= assignment_score <= 100):
                raise ValueError("Assignment score must be 0-100.")
            quiz_score = float(request.form.get('quiz_score') or 0)
            if not (0 <= quiz_score <= 100):
                raise ValueError("Quiz score must be 0-100.")
            exam_score = float(request.form.get('exam_score') or 0)
            if not (0 <= exam_score <= 100):
                raise ValueError("Exam score must be 0-100.")
            study_hours = float(request.form.get('study_hours_per_week') or 0)
            if not (0 <= study_hours <= 168):
                raise ValueError("Study hours must be 0-168.")
            feedback = request.form.get('feedback')
            if not feedback:
                raise ValueError("Feedback required.")
            access_to_internet = request.form.get('access_to_internet', 'YES')
            if access_to_internet not in ['YES', 'NO']:
                raise ValueError("Internet access must be 'YES' or 'NO'.")
            study_support_at_home = request.form.get('study_support_at_home', 'YES')
            if study_support_at_home not in ['YES', 'NO']:
                raise ValueError("Study support must be 'YES' or 'NO'.")

            prediction = predict_with_hybrid(exam_score, quiz_score, assignment_score, attendance_rate, study_hours, feedback)
            student = Student(
                student_id=student_id, performance_label=prediction, gender=gender, age=age,
                attendance_rate=attendance_rate, assignment_score=assignment_score, quiz_score=quiz_score,
                exam_score=exam_score, study_hours_per_week=study_hours, feedback=feedback,
                access_to_internet=access_to_internet, study_support_at_home=study_support_at_home,
                user_id=user_id, created_at=datetime.utcnow()
            )
            student.advice = get_admin_advice(student)
            db.session.add(student)
            db.session.commit()
            flash(f"Prediction for {student_id}: {prediction}.", 'info')
            return redirect(url_for('admin_dashboard'))
        except ValueError as ve:
            flash(f"Input Error: {str(ve)}", 'error')
            db.session.rollback()
        except Exception as e:
            flash(f"Server Error: {str(e)}", 'error')
            db.session.rollback()

    return render_template(
        'admin_dashboard.html', predictions=predictions, complaints=complaints, sort=sort,
        time_filter=time_filter, poor_count=poor_count, average_count=average_count, excellent_count=excellent_count
    )

def create_admin_user():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(role='admin').first():
            hashed_password = bcrypt.hashpw('adminpassword'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            hashed_answer = bcrypt.hashpw('blue'.lower().encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            admin = User(
                username='admin', identifier='SAPPS1', email='yevtechnexus@gmail.com', password=hashed_password,
                role='admin', security_question='What is your favorite color?', security_answer=hashed_answer
            )
            db.session.add(admin)
            db.session.commit()

if __name__ == '__main__':
    create_admin_user()
    app.run(debug=True, host='0.0.0.0', port=5000)