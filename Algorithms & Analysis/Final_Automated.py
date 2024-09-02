import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv("Algorithms & Analysis/assets/student_outcome_dataset.csv")

label_encoder = LabelEncoder()
data['Target'] = label_encoder.fit_transform(data['Target'])

X = data.drop(columns=['Target'])
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifiers = {
    'Random Forest': (RandomForestClassifier(random_state=42), {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    }),
    'Logistic Regression': (LogisticRegressionCV(cv=5, max_iter=1000), {}),
    'K-Nearest Neighbors': (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }),
    'SVM': (SVC(probability=True), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }),
    'XGBoost': (XGBClassifier(), {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    })
}

best_models = {}
best_accuracies = {}
top_features = {}

accuracy_data = []
feature_importance_data = []

classifier_names = list(classifiers.keys())

for name, (classifier, param_grid) in classifiers.items():
    print(f"Training {name}...")
    feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)

    feature_selector.fit(X_train, y_train)

    feature_importances = feature_selector.feature_importances_

    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    threshold = 0.01  

    selected_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]['Feature'].tolist()

    top_features[name] = selected_features

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    X_train_selected_scaled = scaler.fit_transform(X_train_selected)
    X_test_selected_scaled = scaler.transform(X_test_selected)

    grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_selected_scaled, y_train)

    best_model = grid_search.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test_selected_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    best_accuracies[name] = accuracy

    accuracy_data.append((name, accuracy))

    feature_importance_data.append((name, feature_importance_df))

    print(f"The most important features for {name} are: {', '.join(selected_features)}")

    print(f"Accuracy with selected features for {name}: {accuracy * 100:.2f}%")
    print()

best_classifier = max(best_accuracies, key=best_accuracies.get)

print(f"The best classifier is {best_classifier} with accuracy {best_accuracies[best_classifier] * 100:.2f}%")

fig, ax = plt.subplots(figsize=(10, 6))
accuracies = [accuracy[1] for accuracy in accuracy_data]
classifiers = [accuracy[0] for accuracy in accuracy_data]

ax.bar(classifiers, accuracies, color='skyblue')
ax.set_xlabel('Classifiers')
ax.set_ylabel('Accuracy')
ax.set_title('Classifier Accuracies')
plt.xticks(rotation=45) 

for i, v in enumerate(accuracies):
    ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

best_feature_importances = feature_importance_data[classifier_names.index(best_classifier)][1]
fig, ax = plt.subplots()
ax.pie(best_feature_importances['Importance'], labels=best_feature_importances['Feature'], autopct='%1.1f%%', startangle=90)
ax.axis('equal') 

plt.title(f"Feature Importances for {best_classifier}")
plt.show()

print(f"Top Features for the Best Classifier ({best_classifier}):")
for i, param in enumerate(top_features[best_classifier], start=1):
    print(f"{i}. {param}")
