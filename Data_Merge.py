#importing libraries
import pandas as pd
pd.options.display.max_columns = None
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import warnings
import plotly.graph_objects as go
warnings.filterwarnings('ignore') 

#importing datasets
df1= pd.read_csv('patient_df.csv')
df2= pd.read_csv('allergy_df.csv')
df3= pd.read_csv('condition_df.csv')
df4= pd.read_csv('coverage_df.csv')
df5= pd.read_csv('encounter_df.csv')
df6= pd.read_csv('medicalclaim_df.csv')
df7= pd.read_csv('medication_df.csv')
df8= pd.read_csv('vitalsign_df.csv')

#performing concat
df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], axis=1)

df.info()

df.head()

# exporting
df.to_csv('df.csv')
