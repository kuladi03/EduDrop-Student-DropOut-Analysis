import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

# Load the dataset
data = pd.read_csv("assets/new dataset.csv")

# Define the features (X) and the target variable (y)
X = data[
    ['Marital status', "Mother's occupation", "Father's occupation", 'Gender', 'Debtor', 'Tuition fees up to date',
     'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'Unemployment rate']]
y = data['Target'] # Assuming 'Target' is the column that indicates dropout status

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize classifiers with hyperparameter grids
classifiers = {
    'Decision Tree': (DecisionTreeClassifier(), {'criterion': ['gini', 'entropy']}),
    'Random Forest': (RandomForestClassifier(), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}),
    'Logistic Regression': (LogisticRegression(), {'C': [0.001, 0.01, 0.1, 1, 10, 100]}),
    'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9]}),
    'AdaBoost': (AdaBoostClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}),
    'Gradient Boosting': (GradientBoostingClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}),
    'SVM': (SVC(probability=True), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']})
}

best_models = {}
best_accuracies = {}

# Iterate through classifiers and perform GridSearchCV
for name, (classifier, param_grid) in classifiers.items():
    print(f"Training {name}...")
    grid_search = GridSearchCV(classifier, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
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
X = data[
    ['Marital status', "Mother's occupation", "Father's occupation", 'Gender', 'Debtor', 'Tuition fees up to date']]
y = data['Target']  