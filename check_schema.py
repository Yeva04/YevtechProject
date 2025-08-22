from app import app, db
from sqlalchemy import text

with app.app_context():
    try:
        with db.engine.connect() as connection:
            result = connection.execute(text("PRAGMA table_info(user)")).fetchall()
        print("User table schema:", result)
    except Exception as e:
        print("Error querying user table:", e)