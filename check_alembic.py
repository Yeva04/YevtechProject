from app import app, db
from sqlalchemy import text

with app.app_context():
    try:
        with db.engine.connect() as connection:
            result = connection.execute(text("SELECT * FROM alembic_version")).fetchall()
        print("Alembic version:", result)
    except Exception as e:
        print("Error querying alembic_version:", e)