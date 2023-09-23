import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

# Load the dataset
data = pd.read_csv("assets/new dataset.csv")

# Define the features (X) and the target variable (y)
X = data[
    ['Marital status', "Mother's occupation", "Father's occupation", 'Gender', 'Debtor', 'Tuition fees up to date']]
y = data['Target']  # Assuming 'Target' is the column that indicates dropout status

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (if needed)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize classifiers
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),  # Add Decision Tree classifier here
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    # 'XGBoost': XGBClassifier(),
    'SVM': SVC(probability=True)
}

# Perform hyperparameter tuning using GridSearchCV
param_grids = {
    'Decision Tree': {'criterion': ['gini', 'entropy']},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'Logistic Regression': {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7, 9]},
    'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]},
    # 'XGBoost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

best_models = {}
best_accuracies = {}

# Iterate through classifiers and perform GridSearchCV for Decision Tree
for name, classifier in classifiers.items():
    print(f"Training {name}...")
    grid_search = GridSearchCV(classifier, param_grids[name], cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_models[name] = grid_search.best_estimator_
    y_pred = best_models[name].predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    best_accuracies[name] = accuracy
    print(f"Best Parameters for {name}: {grid_search.best_params_}")
    print(f"Accuracy for {name}: {accuracy * 100:.2f}%")
    print()

# Find the classifier with the highest accuracy
best_classifier = max(best_accuracies, key=best_accuracies.get)
print(f"The best classifier is {best_classifier} with accuracy {best_accuracies[best_classifier] * 100:.2f}%")

# Define the ensemble of the best models including Decision Tree
ensemble_models = [
    ('Decision Tree', best_models['Decision Tree']),
    ('Random Forest', best_models['Random Forest']),
    ('Logistic Regression', best_models['Logistic Regression']),
    ('K-Nearest Neighbors', best_models['K-Nearest Neighbors']),
    ('AdaBoost', best_models['AdaBoost']),
    ('Gradient Boosting', best_models['Gradient Boosting']),
    # ('XGBoost', best_models['XGBoost']),  # Uncomment if you want to include XGBoost
    ('SVM', best_models['SVM'])
]

# Create a soft voting classifier
soft_voting_classifier = VotingClassifier(estimators=ensemble_models, voting='soft')

# Train the soft voting classifier on the training data
soft_voting_classifier.fit(X_train, y_train)

# Predict with the soft voting classifier
y_pred_soft = soft_voting_classifier.predict(X_test)
accuracy_soft = accuracy_score(y_test, y_pred_soft)
print(f"Soft Voting Classifier Accuracy: {accuracy_soft * 100:.2f}%")

# Create a hard voting classifier
hard_voting_classifier = VotingClassifier(estimators=ensemble_models, voting='hard')

# Train the hard voting classifier on the training data
hard_voting_classifier.fit(X_train, y_train)

# Predict with the hard voting classifier
y_pred_hard = hard_voting_classifier.predict(X_test)
accuracy_hard = accuracy_score(y_test, y_pred_hard)
print(f"Hard Voting Classifier Accuracy: {accuracy_hard * 100:.2f}%")
