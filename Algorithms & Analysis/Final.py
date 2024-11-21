import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
import io

# Load and preprocess dataset
data = pd.read_csv("Algorithms & Analysis/assets/student_outcome_dataset.csv")
label_encoder = LabelEncoder()
data['Target'] = label_encoder.fit_transform(data['Target'])

# Separate features and target
X = data.drop(columns=['Target'])
y = data['Target']
total_students = len(data)
dropout_students = data['Target'].sum()

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define classifiers and hyperparameters
classifiers = {
    'Random Forest': (RandomForestClassifier(random_state=42), {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}),
    'Logistic Regression': (LogisticRegressionCV(cv=5, max_iter=1000), {}),
    'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
    # 'SVM': (SVC(probability=True), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    'XGBoost': (XGBClassifier(), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2]})
}

best_models = {}
best_accuracies = {}
top_features = {}

# Prepare data for results and graphs
accuracy_data = []
feature_importance_data = []

# Train each model
for name, (classifier, param_grid) in classifiers.items():
    print(f"Training {name}...")

    # Feature importance and selection
    feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    feature_selector.fit(X_train, y_train)
    feature_importances = feature_selector.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    threshold = 0.01  
    selected_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]['Feature'].tolist()
    top_features[name] = selected_features

    # Selected feature set
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    X_train_selected_scaled = scaler.fit_transform(X_train_selected)
    X_test_selected_scaled = scaler.transform(X_test_selected)

    # Grid search
    grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_selected_scaled, y_train)
    best_model = grid_search.best_estimator_
    best_models[name] = best_model

    # Evaluate accuracy
    y_pred = best_model.predict(X_test_selected_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    best_accuracies[name] = accuracy
    accuracy_data.append((name, accuracy))
    feature_importance_data.append((name, feature_importance_df))

# Best model
best_classifier = max(best_accuracies, key=best_accuracies.get)
print(f"The best classifier is {best_classifier} with accuracy {best_accuracies[best_classifier] * 100:.2f}%")

# Create PDF Report
pdf_filename = "EduDrop_Report_Expanded.pdf"
doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
elements = []
styles = getSampleStyleSheet()

# 1. Introduction
introduction_text = """
EduDrop is a project aimed at reducing student dropout rates by analyzing various student features and predicting dropout risk. 
This analysis uses machine learning models to identify key factors influencing student dropouts, offering insights for targeted interventions.
The dataset contains student records with multiple features used to predict dropout likelihood.
"""
elements.append(Paragraph("EduDrop Project Report", styles['Title']))
elements.append(Paragraph("Introduction", styles['Heading2']))
elements.append(Paragraph(introduction_text, styles['BodyText']))
elements.append(Spacer(1, 12))

# 2. Summary
summary_text = f"Total Students: {total_students} <br/> Dropout Students: {dropout_students}"
elements.append(Paragraph("Summary", styles['Heading2']))
elements.append(Paragraph(summary_text, styles['BodyText']))
elements.append(Spacer(1, 12))

# 3. Classifier Details and Parameter Grid
elements.append(Paragraph("Classifier Details and Parameter Grid", styles['Heading2']))
classifier_table_data = [["Classifier", "Parameter Grid"]]
for name, (classifier, param_grid) in classifiers.items():
    classifier_table_data.append([name, str(param_grid)])
classifier_table = Table(classifier_table_data)
classifier_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                      ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                      ('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
elements.append(classifier_table)
elements.append(Spacer(1, 12))

# 4. Classifier Performance
elements.append(Paragraph("Classifier Performance", styles['Heading2']))
accuracy_table_data = [["Classifier", "Accuracy"]]
accuracy_table_data += [(name, f"{acc * 100:.2f}%") for name, acc in accuracy_data]
accuracy_table = Table(accuracy_table_data)
accuracy_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                    ('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
elements.append(accuracy_table)
elements.append(Spacer(1, 12))

# 5. Feature Importance Analysis - Top 10 Features Bar Graph
elements.append(Paragraph(f"Top 10 Feature Importances for {best_classifier}", styles['Heading2']))
best_feature_importances = feature_importance_data[list(classifiers.keys()).index(best_classifier)][1]

# Select the top 10 features
top_10_features = best_feature_importances.head(10)

# Create the bar graph
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_10_features['Feature'], top_10_features['Importance'], color='skyblue')
ax.set_xlabel('Importance')
ax.set_ylabel('Features')
ax.set_title(f"Top 10 Feature Importances for {best_classifier}")
plt.gca().invert_yaxis()  # To display the most important feature at the top
plt.tight_layout()

# Save the graph to a buffer
img_buf = io.BytesIO()
plt.savefig(img_buf, format='png')
img_buf.seek(0)
elements.append(Image(img_buf, width=400, height=400))
plt.close(fig)

# 6. Conclusion and Insights
conclusion_text = f"""
The best performing model was {best_classifier} with an accuracy of {best_accuracies[best_classifier] * 100:.2f}%. 
Key features influencing dropout rates include {', '.join(top_features[best_classifier])}. 
This analysis suggests that specific factors significantly impact dropout likelihood, enabling targeted support for at-risk students. 
Future work could explore additional models or deeper analysis of key feature interactions.
"""
elements.append(Spacer(1, 12))
elements.append(Paragraph("Conclusion and Insights", styles['Heading2']))
elements.append(Paragraph(conclusion_text, styles['BodyText']))

# Save PDF
doc.build(elements)
print(f"Expanded report saved as {pdf_filename}")