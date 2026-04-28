from app import app, db
from models import User
import sys
import os

# Force stdout flushing
sys.stdout.reconfigure(line_buffering=True)

print("Starting reproduction script...", flush=True)

try:
    with app.app_context():
        db_path = app.config['SQLALCHEMY_DATABASE_URI']
        print(f"Database URI: {db_path}", flush=True)
        
        # Check if file exists
        if 'sqlite:///' in db_path:
            path = db_path.replace('sqlite:///', '')
            # Handle relative path
            if not os.path.isabs(path):
                # If instance_path is used by Flask-SQLAlchemy
                # But here we just check if it exists in instance or root
                print(f"Checking for file at: {path}", flush=True)
                print(f"File exists in root? {os.path.exists(path)}", flush=True)
                print(f"File exists in instance? {os.path.exists(os.path.join('instance', path))}", flush=True)

        print("Forcing db.create_all()...", flush=True)
        db.create_all()
        print("db.create_all() executed.", flush=True)
        
        print("Attempting to query User table...", flush=True)
        user = User.query.first()
        print(f"User found: {user}", flush=True)
        
except Exception as e:
    print(f"Caught exception: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("Script finished successfully.", flush=True)
