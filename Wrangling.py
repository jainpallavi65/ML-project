#importing libraries
import pandas as pd
import numpy as np
import time
import datetime
import os
import warnings
warnings.filterwarnings('ignore') 

#plotting 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo

#reading all the datasets
medication = pd.read_csv('/Users/pallavijain/Desktop/medication.csv')
medicalclaim = pd.read_csv('/Users/pallavijain/Desktop/medical_claim.csv')
patient = pd.read_csv('/Users/pallavijain/Desktop/patient.csv')
vital_sign = pd.read_csv('/Users/pallavijain/Desktop/vital_sign.csv')
condition= pd.read_csv('/Users/pallavijain/Desktop/condition.csv')
encounter= pd.read_csv('/Users/pallavijain/Desktop/encounter.csv')
allergy= pd.read_csv('/Users/pallavijain/Desktop/allergy.csv')
coverage=pd.read_csv('/Users/pallavijain/Desktop/coverage.csv')


# # Medication dataset

medication.info()

#finding unique values for categorical columns
print("unique values in request_status:", medication['request_status'].unique())
print("unique values in dose:",medication['dose'].unique()) 
print("unique values in dose_unit:",medication['dose_unit'].unique())
print("unique values in medication_name:",medication['medication_name'].unique())
print("unique values in ndc:",medication['ndc'].unique())
print("unique values in route:",medication['route'].unique())  #how the medication was delivered
#for cols, when unique values are more than 5
print("number of unique values in route:",medication['route'].nunique())
print("number of unique values in dose_unit:",medication['dose_unit'].nunique())#trivial
print("number of unique values in ndc (drug codes):",medication['ndc'].nunique()) #only used for medication errors
print("number of unique values in medication_name:",medication['medication_name'].nunique())


#creating dummy for request status called "request_status_active"
#this variable which measures Status of the medication order (e.g. active | on-hold | cancelled | completed).
medication = pd.get_dummies(medication, columns = ['request_status'])

#creating 5 dummies for Prescribed dose: dose_1, dose_NR, dose_ONE, dose_PRN, dose_SCH, dose_STA
#unique values in dose: [nan 'ONE' 'PRN' '1' 'SCH' 'STA' 'NR']
medication = pd.get_dummies(medication, columns = ['dose'])

#request_date: Date the medication request was entered.
#filled_date: Date the prescription was filled.

#converting them to datetime format
medication["request_date"] = pd.to_datetime(medication["request_date"])
medication["filled_date"] = pd.to_datetime(medication["filled_date"])

#extracting the duration between the time it was requested and filled for the patient
medication["prescription_procurement_duration"] = (medication["filled_date"] - medication["request_date"]).dt.days

medication["prescription_procurement_duration"].describe()

medication.info()

#subsetting the medication df to only contain meanigful variables for the regression
medication_df = medication[["patient_id","encounter_id","request_status_active","medication_name",
                        "route","prescription_procurement_duration","dose_1","dose_NR","dose_ONE","dose_PRN","dose_SCH","dose_STA"]]


#exporting medication_df to csv
medication_df.to_csv("/Users/pallavijain/Desktop/medication_df.csv")


# # medical claims dataset

medicalclaim.info()

#finding unique values for categorical columns
print("unique values in claim_type:", medicalclaim["claim_type"].unique()) #e.g. professional, institutional
print("unique values in discharge_disposition_code:", medicalclaim["discharge_disposition_code"].unique())
print("unique values in hcpcs_code:", medicalclaim["hcpcs_code"].unique())
print("unique values in hcpcs_modifier_1:", medicalclaim["hcpcs_modifier_1"].unique())
print("unique values in hcpcs_modifier_2:", medicalclaim["hcpcs_modifier_2"].unique())
print("unique values in hcpcs_modifier_3:", medicalclaim["hcpcs_modifier_3"].unique())
print("unique values in hcpcs_modifier_4:", medicalclaim["hcpcs_modifier_4"].unique())
print("unique values in hcpcs_modifier_5:", medicalclaim["hcpcs_modifier_5"].unique())
print("unique values in revenue_center_code:", medicalclaim["revenue_center_code"].unique())#for professional claims only
print("unique values in place_of_service_code:", medicalclaim["place_of_service_code"].unique())#for institutional claims only

#for cols, when unique values are more than 5
print("number of unique values in hcpcs_code:", medicalclaim["hcpcs_code"].nunique())
print("number of unique values in hcpcs_modifier_1:", medicalclaim["hcpcs_modifier_1"].nunique())
print("number of unique values in hcpcs_modifier_2:", medicalclaim["hcpcs_modifier_2"].nunique())
print("number of unique values in hcpcs_modifier_3:", medicalclaim["hcpcs_modifier_3"].nunique())
print("number of unique values in revenue_center_code:", medicalclaim["revenue_center_code"].nunique())
print("number of unique values in place_of_service_code:", medicalclaim["place_of_service_code"].nunique())

#generating dummies
medicalclaim = pd.get_dummies(medicalclaim, columns = ['claim_type'])
medicalclaim = pd.get_dummies(medicalclaim, columns = ['hcpcs_modifier_4'])

#converting them to datetime format
medicalclaim["claim_start_date"] = pd.to_datetime(medicalclaim["claim_start_date"])
medicalclaim["claim_end_date"] = pd.to_datetime(medicalclaim["claim_end_date"])

#extracting the claim duration for the patient
medicalclaim["claim_duration"] = (medicalclaim["claim_end_date"] - medicalclaim["claim_start_date"]).dt.days

medicalclaim['claim_duration'].sort_values()

medicalclaim['claim_duration'].describe()

medicalclaim.info()

#subsetting the medicalclaim df to only contain meanigful variables for the regression
medicalclaim_df = medicalclaim[["patient_id", "encounter_id","claim_duration",
                                "paid_amount","charge_amount","hcpcs_modifier_4_GY","hcpcs_modifier_4_PO",
                               "claim_type_DME","claim_type_I","claim_type_P"]]

#exporting medicalclaim_df to csv
medicalclaim_df.to_csv("/Users/pallavijain/Desktop/medicalclaim_df.csv")


# # patient dataset

#checking unique values for categorical variables
print("number of unique values in state:", patient['state'].nunique())#patients from all states 
print("unique values in race:", patient['race'].unique()) # 5 total and unknown, other 
print("unique values in gender:", patient['gender'].unique())
print("unique values in deceased_flag:", patient['deceased_flag'].unique())


#club unknown and other race into one
patient = patient.replace('unknown','other')

#dummy for race and gender
patient = pd.get_dummies(patient, columns = ['race'])
patient = pd.get_dummies(patient, columns = ['gender'])

#calculatning age
patient['birth_date']= pd.to_datetime(patient['birth_date'])
today = datetime.datetime.now()

#adding age to the dataset
patient['age'] = (today - patient['birth_date']).dt.days / 365.25

#checking age distribution across 1000 patients
patient_df['age'].describe()

#checking 1 and 0 distribution across 1000 patients
patient_df['deceased_flag'].value_counts()

#subsetting the patient_df to only contain meanigful variables for the regression
patient_df = patient[[ "patient_id","state","deceased_flag","gender_female","gender_male","race_asian",
                     "race_black","race_hispanic","race_north american native","race_other","race_white", "age"]]

#exporting patient_df to csv
patient_df.to_csv("/Users/pallavijain/Desktop/patient_df.csv")


# # vital sign dataset

vital_sign.info() #Info about vital signs (e.g. heart rate, pulse ox, systolic/diastolic blood pressure, etc.).

#checking uniques
print("unique values in loinc:", vital_sign['loinc'].unique())
print("unique values in loinc_description:", vital_sign['loinc_description'].unique())#convert to data dictionary w codes

#getting dummies for LOINC 
vital_sign = pd.get_dummies(vital_sign, columns = ['loinc'])

#subsetting the vital_sign df to only contain meanigful variables for the regression
vital_sign_df = vital_sign[["encounter_id", "patient_id","loinc_description","loinc_2708-6",
                                "loinc_8310-5","loinc_8462-4","loinc_8480-6","loinc_8867-4", "loinc_9279-1"]]

#exporting vitalsign_df to csv
vital_sign_df.to_csv("/Users/pallavijain/Desktop/vitalsign_df.csv")


# # condition dataset

condition.info()

# inspecting nulls
condition.isna().sum()

#creating variable
condition['primary_diagnosis_rank_1'] = condition['diagnosis_rank'].apply(lambda x: 1 if x == 1 else 0)

#seeing values across all patients
counts_diagnosis_rank = condition['condition_type'].value_counts()

#subsetting
condition = condition[['encounter_id','patient_id', 'condition_type', 'code_type', 'diagnosis_rank']]

#exporting the cleaned file
condition.to_csv('/Users/pallavijain/Desktop/condition_df.csv')


# # encounter dataset

encounter.info()

# inspecting nulls
encounter.isna().sum()

#checking unique values for type of encounter
encounter['encounter_type'].unique()


#calculate encounter_duration

encounter['encounter_start_date'] = pd.to_datetime(encounter['encounter_start_date'], errors='coerce')
encounter['encounter_end_date'] = pd.to_datetime(encounter['encounter_end_date'], errors='coerce')

#converting into days format
encounter["encounter_duration"] = (encounter["encounter_end_date"] - encounter["encounter_start_date"]).dt.days

#subsetting
encounter=encounter[['encounter_id','patient_id', 'encounter_duration']]

#checking the distribution in descending order
encounter.sort_values('encounter_duration', ascending= False)

encounter.head()

#exporting the cleaned file
encounter.to_csv('/Users/pallavijain/Desktop/encounter_df.csv')


# # allergy dataset

#inspecting nulls
allergy.isna().sum()

# unique values 
allergy['severity'].unique()
allergy['status'].unique()

#creating variables to identify patients with active allergies (1 if presense 0 otherwise)
allergy['active_status_1']= allergy['status'].apply(lambda x:1 if x== 'active' else 0)

#subsetting
allergy= allergy[['encounter_id', 'patient_id', 'active_status_1']]

# exporting
allergy.to_csv('/Users/pallavijain/Desktop/allergy_df.csv')

# # coverage dataset

coverage.info()

#inspecting nulls
coverage.isna().sum()

#Create column duration
coverage['coverage_start_date'] = pd.to_datetime(coverage['coverage_start_date'], errors='coerce')
coverage['coverage_end_date'] = pd.to_datetime(coverage['coverage_end_date'], errors='coerce')


#extracting the duration between the time it was requested and filled for the patient
coverage["coverage_duration"] = (coverage["coverage_end_date"] - coverage["coverage_start_date"]).dt.days

#checking distribution
coverage["coverage_duration"].describe()

##Dummy for payer_type (every type is medicare)
coverage['medicare_payer_type']= coverage['payer_type'].apply(lambda x:1 if x=='medicare' else 0)

coverage['medicare_payer_type'].unique()

#subsetting
coverage = coverage[['patient_id', 'coverage_duration', 'medicare_payer_type']]

#exporting
coverage.to_csv('/Users/pallavijain/Desktop/coverage_df.csv')

