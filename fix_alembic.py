from app import app, db
from flask_migrate import upgrade, stamp, current

with app.app_context():
    # Run migrations to ensure the database is up-to-date
    upgrade()
    
    # Get the current (latest) migration version
    current_version = current()
    # Check if a version is returned (current() returns a string like "53cad8d5e320 (head)" or None if no migrations exist)
    if current_version and isinstance(current_version, str) and current_version.strip():
        version_number = current_version.split()[0]  # Extract "53cad8d5e320" from "53cad8d5e320 (head)"
        print(f"Current Alembic version: {current_version}")
        stamp(revision=version_number)
        print(f"Stamped database with version: {version_number}")
    else:
        print("No migrations found. Please ensure migrations are initialized by running 'flask db init' and 'flask db migrate'.")

