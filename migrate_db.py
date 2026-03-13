"""
Run this script ONCE to migrate existing database to new schema
"""
from sqlalchemy import text
from db import engine, SessionLocal, init_db
from utils import hash_password
import os

def migrate():
    print("🔄 Starting database migration...")
    
    # Create new tables/columns
    init_db()
    
    with engine.connect() as conn:
        # Check if migration needed by checking for emp_id column
        try:
            result = conn.execute(text("SELECT emp_id FROM persons LIMIT 1"))
            print("✅ Database already migrated")
            return
        except:
            pass
        
        print("📝 Adding new columns to persons table...")
        
        # Add new columns (SQLite way)
        new_columns = [
            ("emp_id", "VARCHAR(50)"),
            ("username", "VARCHAR(50)"),
            ("password", "VARCHAR(255)"),
            ("mobile", "VARCHAR(20)"),
            ("email", "VARCHAR(100)"),
            ("designation", "VARCHAR(100)"),
            ("department", "VARCHAR(100)"),
            ("created_at", "DATETIME"),
            ("updated_at", "DATETIME"),
            ("is_active", "INTEGER DEFAULT 1"),
        ]
        
        for col_name, col_type in new_columns:
            try:
                conn.execute(text(f"ALTER TABLE persons ADD COLUMN {col_name} {col_type}"))
                print(f"  ✅ Added column: {col_name}")
            except Exception as e:
                if "duplicate column" in str(e).lower():
                    print(f"  ⏭️ Column {col_name} already exists")
                else:
                    print(f"  ⚠️ Error adding {col_name}: {e}")
        
        conn.commit()
    
    # Update existing records with default values
    with SessionLocal() as s:
        from db import Person
        from datetime import datetime
        
        persons = s.query(Person).all()
        for i, p in enumerate(persons):
            if not p.emp_id:
                p.emp_id = f"EMP{str(i+1).zfill(4)}"
            if not p.username:
                p.username = p.name.lower().replace(" ", "_")
            if not p.password:
                p.password = hash_password("password123")  # Default password
            if not p.created_at:
                p.created_at = datetime.utcnow()
            if p.is_active is None:
                p.is_active = 1
        
        s.commit()
        print(f"✅ Updated {len(persons)} existing records")
    
    print("🎉 Migration complete!")
    print("")
    print("⚠️  IMPORTANT: Existing employees have been assigned:")
    print("   - Employee ID: EMP0001, EMP0002, etc.")
    print("   - Username: their name in lowercase (spaces replaced with _)")
    print("   - Password: 'password123' (they should change this)")

if __name__ == "__main__":
    migrate()