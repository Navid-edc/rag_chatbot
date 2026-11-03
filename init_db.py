import sqlite3

DB_PATH = "company_data.db"

def create_database():
    # Connect to SQLite database (creates company_data.db file on Mac SSD)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create Employees table to store structured HR & IT data
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        emp_id INTEGER PRIMARY KEY,
        full_name TEXT NOT NULL,
        department TEXT NOT NULL,
        role TEXT NOT NULL,
        used_leave_days INTEGER NOT NULL,
        remote_days_per_week INTEGER NOT NULL
    )
    """)

    # Clear existing rows to avoid duplicates on re-runs
    cursor.execute("DELETE FROM employees")

    # Insert sample DBA company employees
    sample_employees = [
        (101, "Navid Abbasi", "AI Engineering", "AI/LLM Engineer", 4, 2),
        (102, "Sara Ahmadi", "Software Engineering", "Senior Backend Developer", 18, 2),
        (103, "Reza Karimi", "IT Security", "DevSecOps Engineer", 10, 1),
        (104, "Mina Hosseini", "AI Engineering", "Junior Data Scientist", 0, 0)
    ]

    cursor.executemany("""
    INSERT INTO employees (emp_id, full_name, department, role, used_leave_days, remote_days_per_week)
    VALUES (?, ?, ?, ?, ?, ?)
    """, sample_employees)

    conn.commit()
    
    # Verify inserted records using a standard SQL SELECT query
    print("=== SQL Database 'company_data.db' Initialized Successfully ===")
    cursor.execute("SELECT emp_id, full_name, department, used_leave_days FROM employees")
    for row in cursor.fetchall():
        print(f"Employee ID: {row[0]} | Name: {row[1]} | Dept: {row[2]} | Used Leave: {row[3]} days")

    conn.close()

if __name__ == "__main__":
    create_database()
