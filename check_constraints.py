from app import app, db
from sqlalchemy import text

with app.app_context():
    with db.engine.connect() as connection:
        result = connection.execute(text("PRAGMA index_list('user')")).fetchall()
        for index in result:
            print(index)
            # Look for the unique index on identifier
            index_name = index[1]  # SQLite index name
            if 'identifier' in index_name:
                print(f"Unique constraint name for identifier: {index_name}")