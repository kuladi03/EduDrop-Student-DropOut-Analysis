import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

data = pd.read_csv("assets/dataset.csv")
data.rename(columns={'Nationality': 'Nationality', 'Age at enrollment': 'Age'}, inplace=True)
data['Target'] = data['Target'].map({
    'Dropout': 0,
    'Enrolled': 1,
    'Graduate': 2
})

new_data = data.copy()
new_data = new_data.drop(columns=['Nacionality', 'Mother\'s qualification', 'Father\'s qualification', 'Educational special needs', 'International', 'Curricular units 1st sem (without evaluations)', 'Unemployment rate', 'Inflation rate'], axis=1)

x = new_data['Target'].value_counts().index
y = new_data['Target'].value_counts().values
df = pd.DataFrame({
    'Target': x,
    'Count_T': y
})

fig = px.pie(df,
             names='Target',
             values='Count_T',
             title='How many dropouts, enrolled & graduates are there in Target column')
fig.update_traces(labels=['Graduate', 'Dropout', 'Enrolled'], hole=0.4, textinfo='value+label', pull=[0, 0.2, 0.1])

correlations = data.corr()['Target']
top_10_features = correlations.abs().nlargest(10).index
top_10_corr_values = correlations[top_10_features]

plt.figure(figsize=(10, 11))
plt.bar(top_10_features, top_10_corr_values)
plt.xlabel('Features')
plt.ylabel('Correlation with Target')
plt.title('Top 10 Features with Highest Correlation to Target')
plt.xticks(rotation=45)
plt.show()

px.histogram(new_data['Age'], x='Age', color_discrete_sequence=['lightblue'])

plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Age', data=new_data)
plt.xlabel('Target')
plt.ylabel('Age')
plt.title('Relationship between Age and Target')
plt.show()

X = new_data.drop('Target', axis=1)
y = new_data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

dtree = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=2)
lr = LogisticRegression(random_state=42 , max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=3)
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
xbc = XGBClassifier(tree_method='hist')
svm = svm.SVC(kernel='linear', probability=True)

dtree.fit(X_train, y_train)
rfc.fit(X_train, y_train)
lr.fit(X_train, y_train)
knn.fit(X_train, y_train)
abc.fit(X_train, y_train)
xbc.fit(X_train, y_train)
svm.fit(X_train, y_train)

y_pred = dtree.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

y_pred = rfc.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

y_pred = lr.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

y_pred = knn.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

y_pred = abc.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

y_pred = xbc.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

y_pred = svm.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

ens1 = VotingClassifier(estimators=[('rfc', rfc), ('lr', lr), ('abc',abc), ('xbc',xbc)], voting='soft')
ens1.fit(X_train, y_train)

y_pred = ens1.predict(X_test)
print("Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")

ens2 = VotingClassifier(estimators=[('rfc', rfc), ('lr', lr), ('abc',abc), ('xbc',xbc)], voting='hard')
ens2.fit(X_train, y_train)

y_pred = ens2.predict(X_test)
print("Accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")
