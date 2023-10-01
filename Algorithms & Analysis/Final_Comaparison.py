import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("assets/new dataset.csv")

label_encoder = LabelEncoder()
data['Target'] = label_encoder.fit_transform(data['Target'])

# Define the features (X) and the target variable (y)
X = data[
    ["Mother's occupation", "Father's occupation", 'Course', 'Curricular units 1st sem (approved)' , 'Tuition fees up to date',
     'Curricular units 1st sem (grade)', 'Age at enrollment']]
y = data['Target'] # Assuming 'Target' is the column that indicates dropout status

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize classifiers
classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(),
    'SVM': SVC(probability=True)  # Enable probability estimates for soft voting
}

best_models = {}
best_accuracies = {}

# Iterate through classifiers and perform GridSearchCV for hyperparameter tuning
for name, classifier in classifiers.items():
    print(f"Training {name}...")
    param_grids = {}

    if name == 'Decision Tree':
        param_grids = {'criterion': ['gini', 'entropy']}
    elif name == 'Random Forest':
        param_grids = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    elif name == 'Logistic Regression':
        param_grids = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    elif name == 'K-Nearest Neighbors':
        param_grids = {'n_neighbors': [3, 5, 7, 9]}
    elif name == 'AdaBoost':
        param_grids = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
    elif name == 'Gradient Boosting':
        param_grids = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
    elif name == 'XGBoost':
        param_grids = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
    elif name == 'SVM':
        param_grids = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

    grid_search = GridSearchCV(classifier, param_grids, cv=5, n_jobs=-1, scoring='accuracy')
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

# Create a soft voting classifier
soft_voting_classifier = VotingClassifier(estimators=[(name, model) for name, model in best_models.items()], voting='soft')

# Train the soft voting classifier on the training data
soft_voting_classifier.fit(X_train, y_train)

# Predict with the soft voting classifier
y_pred_soft = soft_voting_classifier.predict(X_test)
accuracy_soft = accuracy_score(y_test, y_pred_soft)
print(f"Soft Voting Classifier Accuracy: {accuracy_soft * 100:.2f}%")

# Create a hard voting classifier
hard_voting_classifier = VotingClassifier(estimators=[(name, model) for name, model in best_models.items()], voting='hard')

# Train the hard voting classifier on the training data
hard_voting_classifier.fit(X_train, y_train)

# Predict with the hard voting classifier
y_pred_hard = hard_voting_classifier.predict(X_test)
accuracy_hard = accuracy_score(y_test, y_pred_hard)
print(f"Hard Voting Classifier Accuracy: {accuracy_hard * 100:.2f}%")

print(f"The best classifier is {best_classifier} with accuracy {best_accuracies[best_classifier] * 100:.2f}%")

