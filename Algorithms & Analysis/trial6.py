# Handling missing values
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your dataset (replace 'your_dataset.csv' with your actual file)
data = pd.read_csv('assets/new dataset.csv')

# Separate features (X) and target variable (y)
X = data.drop(['Marital status', "Mother's occupation", "Father's occupation", 'Gender', 'Debtor', 'Tuition fees up to date'], axis=1)
y = data['Target']

# Data preprocessing
# - Handle missing values, encode categorical variables, etc.

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Feature Scaling (e.g., Standardization)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Using SelectKBest for feature selection with chi-squared test (for classification tasks)


selector = SelectKBest(score_func=chi2, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Example using a Random Forest Classifier with hyperparameter tuning using GridSearchCV


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
}

rf_classifier = RandomForestClassifier()
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_rf_classifier = grid_search.best_estimator_

# Example of using a Random Forest ensemble


rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10)
rf_classifier.fit(X_train, y_train)



scores = cross_val_score(best_rf_classifier, X_train, y_train, cv=5)
average_accuracy = scores.mean()

# Example using Random Oversampling


ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

