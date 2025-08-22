import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('student-mat.csv', sep=';')

# Map features
# Convert studytime to approximate hours
studytime_map = {1: 1, 2: 3.5, 3: 7.5, 4: 12}
df['study_hours'] = df['studytime'].map(studytime_map)

# Approximate attendance_rate (assuming max absences = 100 for simplicity)
df['attendance_rate'] = 100 - df['absences']

# Map grades to scores (scale 0-20 to 0-100)
df['quiz_score'] = df['G1'] * 5
df['assignment_score'] = df['G2'] * 5
df['exam_score'] = df['G3'] * 5

# Bin G3 into performance_label
bins = [-float('inf'), 10, 15, float('inf')]
labels = ['Poor', 'Average', 'Excellent']
df['performance_label'] = pd.cut(df['G3'], bins=bins, labels=labels, include_lowest=True)

# Generate synthetic feedback based on performance_label
feedback_map = {
    'Poor': "Needs to attend more classes and study regularly.",
    'Average': "Good effort but needs improvement in consistency.",
    'Excellent': "Excellent performance, keep it up!"
}
df['feedback'] = df['performance_label'].map(feedback_map)

# Select required columns
required_columns = ['attendance_rate', 'assignment_score', 'quiz_score', 'exam_score', 'study_hours', 'feedback', 'performance_label']
df_processed = df[required_columns]

# Save the processed dataset
df_processed.to_csv('processed_student_data.csv', index=False)
print("Dataset processed and saved as 'processed_student_data.csv'")