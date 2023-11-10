import pandas
from numpy import int64
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from DataConstruction import construct_cat_value
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

train_data = construct_cat_value('./train.csv/train.csv', ['Category', 'Descript', 'DayOfWeek', 'PdDistrict',
                                                           'Resolution', 'Address'])
train_data['Dates'] = pandas.to_datetime(train_data['Dates'])
# train_data['Dates'] = train_data['Dates'].astype(int64)
x_all = train_data.drop(['Category', 'Descript', 'Resolution'], axis=1)
y_all = train_data['Category']

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.25, random_state=42)

std = StandardScaler()
x_train['X'] = std.fit_transform(x_train[['X']])
x_test['X'] = std.transform(x_test[['X']])
x_train['Y'] = std.fit_transform(x_train[['Y']])
x_test['Y'] = std.transform(x_test[['Y']])
x_train['Dates'] = std.fit_transform(x_train[['Dates']])
x_test['Dates'] = std.transform(x_test[['Dates']])

k_value = 100
classifier = KNeighborsClassifier(n_neighbors=k_value, weights='distance', metric='euclidean')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print("K-Value " + str(k_value) + ": ")
print(classification_report(y_test, y_pred, zero_division=0))
