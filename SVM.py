import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split as dask_train_test_split
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, log_loss

print("Loading data...")
dask_data = dd.read_csv('hotDataTrain.csv').repartition(npartitions=160)  
print("Converting Dask DataFrame to Dask DataFrame for initial data preparation...")
data = dask_data
print("Preparing the data...")
X = data[['Dates_new', 'DayOfWeek_new', 'PdDistrict_new', 'Address_new', 'X_new', 'Y_new']]
y = data['Category_new']  # 'Category_new' is the target variable

print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = dask_train_test_split(X, y, test_size=0.2, random_state=23)

print("Scaling the data...")
scaler = DaskStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training the Support Vector Machine...")
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)

# Evaluate the model
print("Evaluating the model...")
y_pred = svm_model.predict(X_test_scaled.compute())
accuracy = accuracy_score(y_test.compute(), y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test.compute(), y_pred))

# Train the Support Vector Machine using DaskLinearSVC with probability estimates
# Calculate log loss with explicit labels argument
print("Training the Support Vector Machine...")
svm_model = SVC(probability=True)
svm_model.fit(X_train_scaled, y_train)
y_pred_proba = svm_model.predict_proba(X_test_scaled.compute())
y_true_pd = y_test.compute().values
logloss = log_loss(y_true_pd, y_pred_proba, labels=svm_model.classes_)
print(f"Log Loss: {logloss:.4f}")