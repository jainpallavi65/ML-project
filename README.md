# ML-project
Predicting Premature Deaths from Claims Data 

Machine Learning models used are logistic regression, random forest, decision tree, K-nearest neighbors with hyperparameter tuning. Models were evaluated using confusion matrix (recall and f-1 score).

Usage instructions:
The .py files contains wrangling, EDA, merging, and predictive modeling codes. 
The Pdf file contains the project findings. 
The cleaned datasets contains the wrangled files.
The original datasets contains the originally downloaded files from Syntegra.io website.

Models: 
1.	RFC before tuning 
2.	RFC (after tuning) 
3.	RFC (after tuning – 21 backward features)
4.	RFC (after tuning – 8 forward features)
5.	DT (before tuning) 
6.	DT (after tuning)
7.	DT (after tuning – 21 backward features)
8.	DT (after tuning – 8 forward features)
9.	Logistic Regression – all 35 X variables
10.	Logit with Backward Feature Selection (21 backward features)
11.	Logit with Forward Feature Selection (8 forward features)
12.	Logit with Exhaustive Feature Selection (5 features)
13.	KNN (before tuning and without pipeline)
14.	KNN (before tuning and with pipeline) 
15.	KNN (after tuning and with pipeline)
16.	KNN (after tuning and with pipeline - 21 backward features)
17.	KNN (after tuning and with pipeline - 8 forward features)

