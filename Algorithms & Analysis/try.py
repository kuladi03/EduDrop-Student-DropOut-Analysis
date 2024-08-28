import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("Algorithms & Analysis/assets/downloaded.csv")  # Replace 'your_dataset.csv' with the actual path to your dataset file

# Check the column names
print("Column Names:", data.columns)

# Create label encoder object
label_encoder = LabelEncoder()

# Check if 'Target' column exists
if 'Target' in data.columns:
    # Encode the 'Target' column
    data['Target'] = label_encoder.fit_transform(data['Target'])
else:
    print("Error: 'Target' column not found in the dataset.")

# Define the features (X) and the target variable (y)
X = data.drop(columns=['Target'])  # Assuming 'Target' is the target variable
y = data['Target']  # Assuming 'Target' is the column that indicates the target variable
