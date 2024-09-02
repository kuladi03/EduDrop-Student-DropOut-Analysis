import pandas as pd
import numpy as np

# Set parameters
n_samples = 1000  # Number of samples in the dataset

# Randomly generate data for each column
mother_occupation = np.random.randint(1, 12, n_samples)  # Random values from 1 to 11
father_occupation = np.random.randint(1, 12, n_samples)  # Random values from 1 to 11
course = np.random.randint(1, 18, n_samples)             # Random values from 1 to 17
curr_units_1st_sem_approved = np.random.randint(0, 15, n_samples)  # Random values from 0 to 7
curr_units_2nd_sem_approved = np.random.randint(0, 15, n_samples)  # Random values from 0 to 7
tuition_fees_up_to_date = np.random.randint(0, 2, n_samples)       # Random values 0 or 1
curr_units_1st_sem_grade = np.random.uniform(0, 40, n_samples)     # Random values between 0 and 20
curr_units_2nd_sem_grade = np.random.uniform(0, 40, n_samples)     # Random values between 0 and 20
age_at_enrollment = np.random.randint(17, 55, n_samples)           # Random values from 17 to 54

# Generate Target based on some logic (you can adjust this logic as needed)
target = np.where((curr_units_1st_sem_approved + curr_units_2nd_sem_approved > 10) & 
                  (curr_units_1st_sem_grade + curr_units_2nd_sem_grade > 20), 'Graduate', 'Dropout')

# Create a DataFrame
data = pd.DataFrame({
    "Mother's occupation": mother_occupation,
    "Father's occupation": father_occupation,
    "Course": course,
    "Curricular units 1st sem (approved)": curr_units_1st_sem_approved,
    "Curricular units 2nd sem (approved)": curr_units_2nd_sem_approved,
    "Tuition fees up to date": tuition_fees_up_to_date,
    "Curricular units 1st sem (grade)": curr_units_1st_sem_grade,
    "Curricular units 2nd sem (grade)": curr_units_2nd_sem_grade,
    "Age at enrollment": age_at_enrollment,
    "Target": target
})

# Save the dataset to a CSV file
data.to_csv('student_outcome_dataset.csv', index=False)

# Display the first few rows of the dataset to confirm
print(data.head())
