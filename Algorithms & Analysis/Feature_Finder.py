import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("assets/new dataset.csv")

# Define the features (X) and the target variable (y)
X = data.drop(columns=['Target'])  # Assuming 'Target' is the target variable
y = data['Target']  # Assuming 'Target' is the column that indicates dropout status

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
classifier.fit(X_train, y_train)

# Get feature importances
feature_importances = classifier.feature_importances_

# Create a DataFrame to hold feature names and their importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the most important feature
most_important_feature = feature_importance_df.iloc[0]['Feature']
print(f"The most important feature is: {most_important_feature}")

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance Analysis')
plt.gca().invert_yaxis()  # Reverse the order to display the most important at the top
plt.show()
