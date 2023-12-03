# Load libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import log_loss

# Assuming you already have col_names defined
col_names = ['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']

# Load dataset
test_data = pd.read_csv(r"C:\Users\Jaden\OneDrive\Desktop\CS\DM\Data mining project\train.csv", header=None, names=col_names)

# Identify columns with mixed data types
mixed_type_columns = [col for col in test_data.columns if test_data[col].apply(lambda x: type(x) == str).any()]

# Convert mixed data type columns to a consistent data type (e.g., str)
for col in mixed_type_columns:
    test_data[col] = test_data[col].astype(str)

enc = OrdinalEncoder()
enc.fit(test_data[["Dates", "Category", "Descript", "DayOfWeek", "PdDistrict", "Resolution", "Address", "X", "Y"]])
test_data[["Dates", "Category", "Descript", "DayOfWeek", "PdDistrict", "Resolution", "Address", "X", "Y"]] = enc.fit_transform(
    test_data[["Dates", "Category", "Descript", "DayOfWeek", "PdDistrict", "Resolution", "Address", "X", "Y"]])

# Load submit_data
submit_data = pd.read_csv(r"C:\Users\Jaden\OneDrive\Desktop\CS\DM\Data mining project\test.csv", header=None,
                          names=col_names, dtype=str)

# Identify columns with mixed data types
mixed_type_columns_submit = [col for col in submit_data.columns if
                             submit_data[col].apply(lambda x: type(x) == str).any()]

# Convert mixed data type columns to a consistent data type (e.g., str)
for col in mixed_type_columns_submit:
    submit_data[col] = submit_data[col].astype(str)

enc_submit = OrdinalEncoder()
enc_submit.fit(submit_data[["Dates", "Category", "Descript", "DayOfWeek", "PdDistrict", "Resolution", "Address", "X", "Y"]].astype(str))
submit_data[["Dates", "Category", "Descript", "DayOfWeek", "PdDistrict", "Resolution", "Address", "X", "Y"]] = \
    enc_submit.fit_transform(submit_data[["Dates", "Category", "Descript", "DayOfWeek", "PdDistrict", "Resolution", "Address", "X", "Y"]])

# Split the data into features (x) and target variable (y)
x_train = test_data.drop(columns=['Category'])
y_train = test_data['Category']

x_test = submit_data.drop(columns=['Category'])
y_test = submit_data['Category']

# Handle missing values in x_train and x_test
imputer = SimpleImputer(strategy='mean')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

# Create and train the model with hyperparameter tuning
param_dist = {
    'n_estimators': randint(10, 200),
    'max_features': ['auto', 'sqrt', 'log2', None],
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'bootstrap': [True, False]
}

model = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
random_search.fit(x_train, y_train)

# Get the best parameters
best_params = random_search.best_params_

# Use Recursive Feature Elimination (RFE) for feature selection
top_n = 5
rfe = RFE(random_search.best_estimator_, n_features_to_select=top_n)
x_train_rfe = rfe.fit_transform(x_train, y_train)
x_test_rfe = rfe.transform(x_test)

# Retrain the model on the selected features
random_search.best_estimator_.fit(x_train_rfe, y_train)

# Predict on the test set
y_pred = model.best_estimator_.predict(x_test_rfe)

# Calculate accuracy and F1 score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

y_pred = model.best_estimator_.predict_proba(x_test_rfe)
print(log_loss(y_test, y_pred))

prediction = pd.DataFrame(y_pred,columns =  model.predict)
prediction.to_csv('results.csv')

print("Best Parameters:", best_params)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
