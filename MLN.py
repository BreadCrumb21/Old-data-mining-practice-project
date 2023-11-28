import math

import numpy as np
import pandas
from scipy import stats
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from DataConstruction import construct_cat_value

train_data = construct_cat_value('./train.csv/train.csv', ['Category', 'Descript', 'DayOfWeek', 'PdDistrict',
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

# Set up and train the MLP classifier
mln = MLPClassifier(max_iter=1000, verbose=True)
mln.fit(x_train, y_train)

# Predict results and calculate the Log Loss
y_pred = mln.predict_proba(x_test)
print(log_loss(y_test, y_pred))
