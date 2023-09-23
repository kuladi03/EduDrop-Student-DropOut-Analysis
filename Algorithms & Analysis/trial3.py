import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# Load the dataset
data = pd.read_csv("assets/dataset.csv")

# Explore the dataset (e.g., check data types, missing values, etc.)
print(data.info())
# Define the features (X) and the target variable (y)
X = data[['Marital status', 'Parent\'s occupation', 'Gender', 'Debtor',
          'Tuition fees up to date']]
y = data['Target']  # Assuming 'Target' is the column that indicates dropout status

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame to store feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Create a bar chart to visualize Marital Status
plt.figure(figsize=(8, 6))
sns.countplot(x='Marital status', hue='Target', data=data)
plt.title('Marital Status vs. Dropout Status')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.legend(title='Dropout Status', loc='upper right', labels=['Dropout', 'Enrolled', 'Graduate'])
plt.xticks(rotation=45)
plt.show()

# Create a bar chart to visualize Father's Occupation
plt.figure(figsize=(12, 6))
sns.countplot(x='Father\'s occupation', hue='Target', data=data)
plt.title("Father's Occupation vs. Dropout Status")
plt.xlabel("Father's Occupation")
plt.ylabel('Count')
plt.legend(title='Dropout Status', loc='upper right', labels=['Dropout', 'Enrolled', 'Graduate'])
plt.xticks(rotation=90)
plt.show()

# Create a bar chart to visualize Gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', hue='Target', data=data)
plt.title('Gender vs. Dropout Status')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Dropout Status', loc='upper right', labels=['Dropout', 'Enrolled', 'Graduate'])
plt.xticks(rotation=45)
plt.show()

# Create a bar chart to visualize Debtor
plt.figure(figsize=(8, 6))
sns.countplot(x='Debtor', hue='Target', data=data)
plt.title('Debtor vs. Dropout Status')
plt.xlabel('Debtor')
plt.ylabel('Count')
plt.legend(title='Dropout Status', loc='upper right', labels=['Dropout', 'Enrolled', 'Graduate'])
plt.xticks(rotation=45)
plt.show()

# Create a bar chart to visualize Tuition Fees up to Date
plt.figure(figsize=(8, 6))
sns.countplot(x='Tuition fees up to date', hue='Target', data=data)
plt.title('Tuition Fees up to Date vs. Dropout Status')
plt.xlabel('Tuition Fees up to Date')
plt.ylabel('Count')
plt.legend(title='Dropout Status', loc='upper right', labels=['Dropout', 'Enrolled', 'Graduate'])
plt.xticks(rotation=45)
plt.show()
