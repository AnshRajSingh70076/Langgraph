import sqlite3

def setup_db():
    connection = sqlite3.connect("mydb.db")
    cursor = connection.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        emp_id INTEGER PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        hire_date TEXT NOT NULL,
        salary REAL NOT NULL
    );
    """)

    employee_data = [
        (1, "Sunny", "Savita", "sunny.sv@abc.com", "2023-06-01", 50000.00),
        (2, "Arhun", "Meheta", "arhun.m@gmail.com", "2022-04-15", 60000.00),
        (3, "Alice", "Johnson", "alice.johnson@jpg.com", "2021-09-30", 55000.00),
        (4, "Bob", "Brown", "bob.brown@uio.com", "2020-01-20", 45000.00),
    ]

    cursor.executemany("""
    INSERT OR IGNORE INTO employees (emp_id, first_name, last_name, email, hire_date, salary)
    VALUES (?, ?, ?, ?, ?, ?);
    """, employee_data)

    connection.commit()
    connection.close()
