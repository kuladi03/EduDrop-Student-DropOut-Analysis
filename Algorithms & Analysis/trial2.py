import pandas as pd
import plotly.express as px
# Load the sample dataset
data = pd.read_csv("assets/dataset.csv")

# Display the first few rows of the dataset to get an overview
print(data.head())

# Create a bar graph for Mother’s Occupation
mother_occupation_counts = data['Mother\'s occupation'].value_counts()
fig = px.bar(mother_occupation_counts, x=mother_occupation_counts.index, y=mother_occupation_counts.values,
             title='Distribution of Students by Mother’s Occupation')
fig.show()

# Create a bar graph for Father’s Occupation
father_occupation_counts = data['Father\'s occupation'].value_counts()
fig = px.bar(father_occupation_counts, x=father_occupation_counts.index, y=father_occupation_counts.values,
             title='Distribution of Students by Father’s Occupation')
fig.show()

# Create a sunburst chart for Gender
gender_counts = data['Gender'].value_counts()
fig = px.sunburst(
    names=gender_counts.index,
    parents=['', ''],
    values=gender_counts.values,
    title='Distribution of Students by Gender'
)
fig.show()

# Create a radar chart to compare Debtor and Tuition fees up to date
radar_data = data[['Debtor', 'Tuition fees up to date', 'Target']].copy()

fig = px.line_polar(
    radar_data,
    r=['Debtor', 'Tuition fees up to date'],
    theta='Target',
    line_close=True,
    title='Comparison of Debtor and Tuition fees up to date',
)
fig.show()
