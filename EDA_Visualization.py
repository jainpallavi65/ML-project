#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import warnings
import plotly.graph_objects as go
import plotly.offline as pyo
warnings.filterwarnings('ignore') 


#Read the cleaned dataset
df= pd.read_csv('/Users/pallavijain/Desktop/cleaned_merged_df.csv')


df.info()


# Distribution of Males and Females

df_for_plot = df.replace({'gender_female': {0: 'Male', 1: 'Female'}})

counts = df_for_plot['gender_female'].value_counts()


fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
             color_discrete_map={'Female': 'pink', 'Male': 'blue'})

fig.update_layout(title='Count by Gender', xaxis_title='Gender', yaxis_title='Count')
fig.update_layout(xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False))
fig.show()


# Females: 68
# 
# Males: 57




#Number of Patients by race

race_cols = ['race_asian', 'race_black', 'race_hispanic', 'race_north american native', 'race_other', 'race_white']

df['race'] = df.apply(lambda x: ', '.join([race_cols[i] for i in range(len(race_cols)) if x[race_cols[i]] == 1]), axis=1)

race_counts = df.groupby('race')['patient_id'].nunique().reset_index()


fig = px.bar(race_counts, x='race', y='patient_id', color='race',
             title='Count of Unique Patient IDs for Each Race', 
             labels={'race': 'Race', 'patient_id': 'Number of Unique Patient IDs',
                     'gender_female': 'Gender Female', 'gender_male': 'Gender Male'})
fig.update_layout(xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False))
fig.show()


# 'race_asian': 1
# 
# 'race_black': 13
# 
# 'race_hispanic': 9
# 
# 'race_north american native': 1
# 
# 'race_other': 8
# 
# 'race_white': 93 - Highest
# 




#Distribution of patients' age

fig = px.histogram(df, x='age', nbins=20, title='Histogram of Age', color_discrete_sequence=px.colors.qualitative.Dark24)
fig.update_layout(xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False))
fig.show()





#Number of patients per state in USA

state_counts = df.groupby('state')['patient_id'].nunique().reset_index()

fig = px.bar(state_counts, x='state', y='patient_id', title='Number of Patients by State', color='state' )
fig.update_layout(xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False))
fig.show()


# Highest: Colorado - 12 patients


# Percentage of deceased and survived in the model data

deceased_counts = df['deceased_flag'].value_counts()
alive = deceased_counts[0]
deceased = deceased_counts[1]


labels = ['Alive', 'Deceased']
values = [alive, deceased]

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', 
                  textfont_size=14, marker=dict(colors=['#1f77b5', '#ff7f0e']))
fig.update_layout(title='Patient Status', title_font_size=20)

fig.show()



#Distribution of patients' coverage duration

fig = px.histogram(df, x='coverage_duration')
fig.update_layout(xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False))
fig.show()



#Total of deceased and survived by gender

grouped = df.groupby(['deceased_flag', 'gender_female'])['patient_id'].count().reset_index()

fig = go.Figure(go.Bar(
    x=grouped['gender_female'],
    y=grouped['patient_id'],
    marker_color=['#1f77b5', '#ff7f0e'],
    text=grouped['patient_id'],
    textposition='auto',
    hovertemplate="Gender: %{x}<br>Count: %{y}<extra></extra>"
))

fig.update_layout(
    title='Deceased Flag by Gender',
    xaxis_title='Female Gender',
    yaxis_title='Count',
    showlegend=False
)
fig.update_layout(xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False))
fig.show()




##just for ref
df[df['deceased_flag'] == 1].groupby('gender_female').size().reset_index(name='count')



#Correlation between paid amount and charge amount
df['gender_female'] = df['gender_female'].astype(str)
fig = px.scatter(df, x='paid_amount', y='charge_amount', color='gender_female',
                 hover_name='patient_id', hover_data=['age', 'state'], size_max=10, opacity=0.7,
                 color_discrete_map={0: 'lightblue', 1: 'pink'})

fig.update_layout(xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False))
fig.show()


# - There seems to be a linear correlation bn the two variables.
# - The amount of paid and charge amount are high for females more than males


# Visualize the distribution of claim_duration

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='claim_duration', bins=20, kde=True)
plt.title('Claim Duration Distribution')
plt.show()
