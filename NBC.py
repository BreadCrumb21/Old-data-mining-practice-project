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


x = test_data.drop(columns=['Dates','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y'])
y = test_data['Category']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=125
)

model = GaussianNB()
model.fit(x_train, y_train)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)

y_pred = model.predict(x_test)

accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)
