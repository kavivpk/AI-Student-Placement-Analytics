import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

departments = ['CSE', 'IT', 'ECE', 'EEE', 'MECH']

cgpa          = np.round(np.random.uniform(5.0, 10.0, n), 2)
attendance    = np.round(np.random.uniform(50, 100, n), 2)
coding        = np.random.randint(20, 100, n)
aptitude      = np.random.randint(20, 100, n)
communication = np.random.randint(20, 100, n)
projects      = np.random.randint(0, 6, n)
internships   = np.random.randint(0, 3, n)

placement = []
for i in range(n):
    score = (
        cgpa[i] * 0.3 +
        attendance[i] * 0.01 +
        coding[i] * 0.2 +
        aptitude[i] * 0.2 +
        communication[i] * 0.15 +
        projects[i] * 2 +
        internships[i] * 3
    )
    placement.append('Placed' if score >= 35 else 'Not Placed')

df = pd.DataFrame({
    'student_id'          : range(1, n+1),
    'name'                : [f'Student_{i}' for i in range(1, n+1)],
    'department'          : np.random.choice(departments, n),
    'cgpa'                : cgpa,
    'attendance'          : attendance,
    'coding_score'        : coding,
    'aptitude_score'      : aptitude,
    'communication_score' : communication,
    'projects_count'      : projects,
    'internships'         : internships,
    'placement_status'    : placement
})

df.to_csv('data/students.csv', index=False)
print(f'✅ Dataset created! Total students: {len(df)}')
print(df['placement_status'].value_counts())