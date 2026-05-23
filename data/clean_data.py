import pandas as pd

df = pd.read_csv('data/students.csv')

print("=== Before Cleaning ===")
print("Shape:", df.shape)
print("Null values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# Clean பண்ணு
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# CGPA valid range check
df = df[(df['cgpa'] >= 0) & (df['cgpa'] <= 10)]

# Attendance valid range check
df = df[(df['attendance'] >= 0) & (df['attendance'] <= 100)]

# placement_status — binary encode (ML-க்கு தேவை)
df['placement_encoded'] = df['placement_status'].map({'Placed': 1, 'Not Placed': 0})

df.to_csv('data/students_cleaned.csv', index=False)

print("\n=== After Cleaning ===")
print("Shape:", df.shape)
print("✅ Cleaned dataset saved!")
print("\nSample:\n", df.head(3))