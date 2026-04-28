from flask import Blueprint, render_template, redirect, url_for, flash, request, session
from flask_bcrypt import Bcrypt

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        from models import User
        from extensions import bcrypt

        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            session['user'] = username
            session['role'] = user.role

            flash("Login successful!", "success")
            if user.role == "admin":
                return redirect(url_for('admin_dashboard'))
            elif user.role == "faculty":
                return redirect(url_for('faculty_dashboard'))
            else:
                return redirect(url_for('student_dashboard'))
        else:
            flash("Invalid credentials!", "danger")

    return render_template('login.html')

@auth.route('/logout')
def logout():
    session.clear()  # Clear entire session instead of just popping specific keys
    flash("Logged out successfully!", "info")
    return redirect(url_for('auth.login'))
