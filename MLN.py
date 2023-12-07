import math
from random import randint

import numpy as np
import pandas
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from DataConstruction import construct_cat_value

train_data = construct_cat_value('./train.csv/train.csv', ['Descript', 'DayOfWeek', 'PdDistrict',
                                                           'Resolution'])

# Removing "### Block of" from addresses
train_data['Address'] = train_data['Address'].str.replace(r'[0-9]+ Block of ', '', regex=True)
train_data['Address'] = train_data['Address'].astype('category')
train_data['Address'] = train_data['Address'].cat.codes

# Split date into separate hours, minutes, etc.
train_data['Dates'] = pandas.to_datetime(train_data['Dates'])
train_data['Hours'] = train_data['Dates'].dt.hour
train_data['Minutes'] = train_data['Dates'].dt.minute
train_data['Month'] = train_data['Dates'].dt.month
train_data['Year'] = train_data['Dates'].dt.year

# Removing outliers from X, Y
train_data = train_data[(np.abs(stats.zscore(train_data['X'])) < 3)]
train_data = train_data[(np.abs(stats.zscore(train_data['Y'])) < 3)]

# Convert X & Y into polar coordinates, just to have another variable to play with
train_data['R'] = np.sqrt(np.power(train_data['X'], 2), np.power(train_data['Y'], 2))
train_data['theta'] = np.arctan(np.divide(train_data['Y'], train_data['X']))

# Drop relevant columns that are not needed or that will cause error
x_all = train_data.drop(['Category', 'Descript', 'Resolution', 'Dates', 'Address'], axis=1)
y_all = train_data['Category']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.25, random_state=42)

# Standardize the coordinate variables
std = StandardScaler()
x_train['X'] = std.fit_transform(x_train[['X']])
x_test['X'] = std.transform(x_test[['X']])
x_train['Y'] = std.fit_transform(x_train[['Y']])
x_test['Y'] = std.transform(x_test[['Y']])
x_train['R'] = std.fit_transform(x_train[['R']])
x_test['R'] = std.transform(x_test[['R']])
x_train['theta'] = std.fit_transform(x_train[['theta']])
x_test['theta'] = std.transform(x_test[['theta']])

test_data = construct_cat_value('./test.csv/test.csv', ['DayOfWeek', 'PdDistrict'])

# Removing "### Block of" from addresses
test_data['Address'] = test_data['Address'].str.replace(r'[0-9]+ Block of ', '', regex=True)
test_data['Address'] = test_data['Address'].astype('category')
test_data['Address'] = test_data['Address'].cat.codes

# Split date into separate hours, minutes, etc.
test_data['Dates'] = pandas.to_datetime(test_data['Dates'])
test_data['Hours'] = test_data['Dates'].dt.hour
test_data['Minutes'] = test_data['Dates'].dt.minute
test_data['Month'] = test_data['Dates'].dt.month
test_data['Year'] = test_data['Dates'].dt.year

# Convert X & Y into polar coordinates, just to have another variable to play with
test_data['R'] = np.sqrt(np.power(test_data['X'], 2), np.power(test_data['Y'], 2))
test_data['theta'] = np.arctan(np.divide(test_data['Y'], test_data['X']))

# Split data
# Standardize the coordinate variables
std = StandardScaler()
test_data['X'] = std.fit_transform(test_data[['X']])
test_data['Y'] = std.fit_transform(test_data[['Y']])
test_data['R'] = std.fit_transform(test_data[['R']])
test_data['theta'] = std.fit_transform(test_data[['theta']])

# Set up and train the MLP classifier
mln = MLPClassifier(hidden_layer_sizes=128, alpha=.00005)
mln.fit(x_train, y_train)

test_data = test_data.drop(['Id', 'Dates', 'Address'], axis=1)

# Predict results and calculate the Log Loss
y_pred = mln.predict_proba(test_data)
prediction = pd.DataFrame(y_pred, columns=mln.classes_)
prediction.to_csv('results.csv')
