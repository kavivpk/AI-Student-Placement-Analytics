import pandas as pd
import mysql.connector

# ── MySQL Connection ──────────────────────────────
conn = mysql.connector.connect(
    host     = 'localhost',
    user     = 'root',
    password = 'Kavi@2006',
    database = 'student_analytics'
)
cursor = conn.cursor()
print("✅ MySQL Connected!")

# ── Load CSV ──────────────────────────────────────
df = pd.read_csv('data/students_cleaned.csv')
print(f"📊 Total records to insert: {len(df)}")

# ── Insert Data ───────────────────────────────────
inserted = 0
for _, row in df.iterrows():
    sql = """
        INSERT IGNORE INTO students
        (student_id, name, department, cgpa, attendance,
         coding_score, aptitude_score, communication_score,
         projects_count, internships, placement_status, placement_encoded)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    values = (
        int(row['student_id']),
        str(row['name']),
        str(row['department']),
        float(row['cgpa']),
        float(row['attendance']),
        int(row['coding_score']),
        int(row['aptitude_score']),
        int(row['communication_score']),
        int(row['projects_count']),
        int(row['internships']),
        str(row['placement_status']),
        int(row['placement_encoded'])
    )
    cursor.execute(sql, values)
    inserted += 1

conn.commit()
print(f"✅ {inserted} records inserted successfully!")

# ── Verify ────────────────────────────────────────
cursor.execute("SELECT COUNT(*) FROM students")
count = cursor.fetchone()[0]
print(f"📋 Total records in MySQL: {count}")

cursor.execute("""
    SELECT placement_status, COUNT(*) as count
    FROM students
    GROUP BY placement_status
""")
for row in cursor.fetchall():
    print(f"   {row[0]}: {row[1]}")

cursor.close()
conn.close()
print("✅ Connection closed!")