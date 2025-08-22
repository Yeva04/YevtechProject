from app import app, db
from sqlalchemy import text

with app.app_context():
    try:
        with db.engine.connect() as connection:
            connection.execute(text("ALTER TABLE user RENAME COLUMN student_id TO identifier"))
        print("Column renamed successfully.")
    except Exception as e:
        print(f"Error: {e}")