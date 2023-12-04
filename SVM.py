import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the data from the CSV file using Dask
print("Loading data...")
dask_data = dd.read_csv('hotDataTrain.csv')  # Replace with the actual path to your CSV file

# Keep the Dask DataFrame for further processing
# No need to compute() as we're going to keep it in Dask format
print("Converting Dask DataFrame to Dask DataFrame for initial data preparation...")
data = dask_data

# Prepare the data
print("Preparing the data...")
X = data[['Descript_new', 'DayOfWeek_new', 'PdDistrict_new', 'Resolution_new']]
y = data['Category_new']  # 'Category_new' is the target variable

# Convert Dask DataFrames to Pandas DataFrames
X = X.compute()
y = y.compute()

# Split the data into training and testing sets
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Scale the data using Dask StandardScaler
print("Scaling the data...")
scaler = DaskStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Support Vector Machine using DaskLinearSVC
print("Training the Support Vector Machine...")
svm_model = SVC()

# Fit the SVM model using Dask DataFrames
svm_model.fit(X_train_scaled, y_train)

# Evaluate the model
print("Evaluating the model...")
y_pred = svm_model.predict(X_test_scaled.compute())
accuracy = accuracy_score(y_test.compute(), y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test.compute(), y_pred))
