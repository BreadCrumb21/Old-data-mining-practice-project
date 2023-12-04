import pandas as pd
import dask.dataframe as dd
from sklearn.calibration import CalibratedClassifierCV
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
from sklearn.svm import LinearSVC

# Load the data from the CSV file using Dask
print("Loading data...")
dask_data = dd.read_csv('hotDataTrain.csv')  # Replace with the actual path to your CSV file
test_data = dd.read_csv('hotDataTest.csv')  # Replace with the actual path to your CSV file


# Keep the Dask DataFrame for further processing
# No need to compute() as we're going to keep it in Dask format
print("Converting Dask DataFrame to Dask DataFrame for initial data preparation...")
data = dask_data

# Prepare the data
print("Preparing the data...")
X = data[['DayOfWeek_new', 'PdDistrict_new', 'Address_new']]
Y = data['Category_new']  # 'Category_new' is the target variable
Test = test_data[['DayOfWeek_new', 'PdDistrict_new', 'Address_new']]

# Convert Dask DataFrames to Pandas DataFrames
X = X.compute()
Y = Y.compute()
Test = Test.compute()

# Scale the data using Dask StandardScaler
print("Scaling the data...")
scaler = DaskStandardScaler()
X_train_scaled = scaler.fit_transform(X)

# Train the Support Vector Machine using DaskLinearSVC
print("Training the Support Vector Machine...")
svm_model = LinearSVC(verbose=True)

# Fit the SVM model using Dask DataFrames
clf = CalibratedClassifierCV(svm_model)
clf.fit(X_train_scaled, Y)

# Evaluate the model
print("Evaluating the model...")
y_pred = clf.predict_proba(Test)
prediction = pd.DataFrame(y_pred, columns=clf.classes_)
prediction.to_csv('result.csv')
