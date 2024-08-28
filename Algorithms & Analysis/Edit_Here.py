import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset (replace 'your_dataset.csv' with the actual dataset file)
data = pd.read_csv("Algorithms & Analysis/assets/new dataset.csv")

label_encoder = LabelEncoder()
data['Target'] = label_encoder.fit_transform(data['Target'])

# Define the features (X) and the target variable (y)
X = data.drop(columns=['Target'])  # Assuming 'Target' is the target variable
y = data['Target']  # Assuming 'Target' is the column that indicates the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of classifiers to use
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegressionCV(cv=5, max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'SVM': SVC(probability=True),  # Enable probability estimates for soft voting
    'XGBoost': XGBClassifier(n_estimator = 20)
}

best_models = {}
best_accuracies = {}
top_features = {}

# Lists to store data for pie charts
accuracy_data = []
feature_importance_data = []

# Store classifier names in a list
classifier_names = list(classifiers.keys())

# Iterate through classifiers and perform feature selection and model training
for name, classifier in classifiers.items():
    print(f"Training {name}...")

    # Initialize the Random Forest classifier for feature selection
    feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the feature selector
    feature_selector.fit(X_train, y_train)

    # Get feature importances
    feature_importances = feature_selector.feature_importances_

    # Create a DataFrame to hold feature names and their importances
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    # Sort the features by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Set a threshold for feature importance (you can adjust this threshold)
    threshold = 0.01  # For example, only consider features with importance >= 0.01

    # Select the most important features based on the threshold
    selected_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]['Feature'].tolist()

    # Store the selected features for each classifier
    top_features[name] = selected_features

    # Use only the selected features for training and testing
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Train and evaluate the classifier using only the selected features
    classifier.fit(X_train_selected, y_train)
    y_pred = classifier.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    best_accuracies[name] = accuracy

    # Append accuracy to the accuracy_data list
    accuracy_data.append((name, accuracy))

    # Append feature importances to the feature_importance_data list
    feature_importance_data.append((name, feature_importance_df))

    # Print the most important features
    print(f"The most important features for {name} are: {', '.join(selected_features)}")

    # Print the accuracy using only selected features
    print(f"Accuracy with selected features for {name}: {accuracy * 100:.2f}%")
    print()

# Find the classifier with the highest accuracy
best_classifier = max(best_accuracies, key=best_accuracies.get)

# Create a soft voting classifier
soft_voting_classifier = VotingClassifier(estimators=[(name, classifier) for name, classifier in classifiers.items()], voting='soft')

# Train the soft voting classifier on the training data
soft_voting_classifier.fit(X_train, y_train)

# Predict with the soft voting classifier
y_pred_soft = soft_voting_classifier.predict(X_test)
accuracy_soft = accuracy_score(y_test, y_pred_soft)
print(f"Soft Voting Classifier Accuracy: {accuracy_soft * 100:.2f}%")

# Create a hard voting classifier
hard_voting_classifier = VotingClassifier(estimators=[(name, classifier) for name, classifier in classifiers.items()], voting='hard')

# Train the hard voting classifier on the training data
hard_voting_classifier.fit(X_train, y_train)

# Predict with the hard voting classifier
y_pred_hard = hard_voting_classifier.predict(X_test)
accuracy_hard = accuracy_score(y_test, y_pred_hard)
print(f"Hard Voting Classifier Accuracy: {accuracy_hard * 100:.2f}%")

print(f"The best classifier is {best_classifier} with accuracy {best_accuracies[best_classifier] * 100:.2f}%")

# Create a bar chart for classifier accuracies
fig, ax = plt.subplots(figsize=(10, 6))
accuracies = [accuracy[1] for accuracy in accuracy_data]
classifiers = [accuracy[0] for accuracy in accuracy_data]

ax.bar(classifiers, accuracies, color='skyblue')
ax.set_xlabel('Classifiers')
ax.set_ylabel('Accuracy')
ax.set_title('Classifier Accuracies')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Display accuracy values on top of each bar
for i, v in enumerate(accuracies):
    ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Display pie chart for feature importances of the best classifier
best_feature_importances = feature_importance_data[classifier_names.index(best_classifier)][1]
fig, ax = plt.subplots()
ax.pie(best_feature_importances['Importance'], labels=best_feature_importances['Feature'], autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title(f"Feature Importances for {best_classifier}")
plt.show()

# Display the top features for the best classifier
print(f"Top Features for the Best Classifier ({best_classifier}):")
for i, param in enumerate(top_features[best_classifier], start=1):
    print(f"{i}. {param}")
