# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)


col_names = ['Dates','Category','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y']
# load dataset
test_data = pd.read_csv(r"C:\Users\Jaden\OneDrive\Desktop\CS\DM\Data mining project\train.csv", header=None, names=col_names)
# print(test_data.head())
# Identify columns with mixed data types
mixed_type_columns = [col for col in test_data.columns if test_data[col].apply(lambda x: type(x) == str).any()]

# Convert mixed data type columns to a consistent data type (e.g., str)
for col in mixed_type_columns:
    test_data[col] = test_data[col].astype(str)

enc = OrdinalEncoder()
#encoded_data = pd.DataFrame(enc.fit_transform(test_data))
#print(encoded_data.head())
enc.fit(test_data[["Dates","Category","Descript","DayOfWeek","PdDistrict","Resolution","Address","X","Y"]])
test_data[["Dates","Category","Descript","DayOfWeek","PdDistrict","Resolution","Address","X","Y"]] = enc.fit_transform(
    test_data[["Dates","Category","Descript","DayOfWeek","PdDistrict","Resolution","Address","X","Y"]])

col_names = ['Dates','Category','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y']
# load dataset
submit_data = pd.read_csv(r"C:\Users\Jaden\OneDrive\Desktop\CS\DM\Data mining project\test.csv", header=None, names=col_names, dtype=str)
# print(test_data.head())
# Identify columns with mixed data types
mixed_type_columns = [col for col in test_data.columns if test_data[col].apply(lambda x: type(x) == str).any()]

# Convert mixed data type columns to a consistent data type (e.g., str)
for col in mixed_type_columns:
    submit_data[col] = test_data[col].astype(str)

enc = OrdinalEncoder()
#encoded_data = pd.DataFrame(enc.fit_transform(test_data))
#print(encoded_data.head())
enc.fit(test_data[["Dates","Category","Descript","DayOfWeek","PdDistrict","Resolution","Address","X","Y"]].astype(str))
submit_data[["Dates","Category","Descript","DayOfWeek","PdDistrict","Resolution","Address","X","Y"]] = enc.fit_transform(
    submit_data[["Dates","Category","Descript","DayOfWeek","PdDistrict","Resolution","Address","X","Y"]])

# Split the data into features (x) and target variable (y)
x_train = test_data.drop(columns=['Dates', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y'])
y_train = test_data['Category']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Create and train the model
model = GaussianNB()
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Calculate accuracy and F1 score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)
