# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split, cross_val_score # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import log_loss

te_col_names = ['Id','Dates','DayOfWeek','PdDistrict','Address','X','Y']
tr_col_names = ['Dates','Category','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y']
# load dataset
test_data = pd.read_csv('test.csv/test.csv', header=0, names=te_col_names)
train_data = pd.read_csv('train.csv/train.csv', header=0, names=tr_col_names)
#train_data = pd.read_csv("hotDataTrain.csv", header=0, names=tr_col_names)
# print(test_data.head())

enc = OrdinalEncoder()
#encoded_data = pd.DataFrame(enc.fit_transform(test_data))
#print(encoded_data.head())
enc.fit(test_data[['Id','Dates','DayOfWeek','PdDistrict','Address','X','Y']])
test_data[['Id','Dates','DayOfWeek','PdDistrict','Address','X','Y']] = enc.fit_transform(
    test_data[['Id','Dates','DayOfWeek','PdDistrict','Address','X','Y']])

enc.fit(train_data[['Dates','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y']])
train_data[['Dates','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y']] = enc.fit_transform(
    train_data[['Dates','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y']])

X_train = train_data.drop(columns=['Category','Descript','Resolution'])
y_train = train_data['Category']

# X_traindata = train_data.drop(columns=['Category','Descript','Resolution'])
# y_traindata = train_data['Category']

#test_data['Category'] = ''
X_test = test_data.drop(columns=['Id'])
#y_test = test_data['Category']

#X_train, X_test, y_train, y_test = train_test_split(X_traindata,y_traindata,test_size=0.3,random_state=1)
# X_train, y_train = train_data.drop(y, axis=1), train_data[y]
# X_test, y_test = test_data.drop(y, axis=1), test_data[y]

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

y_pred = clf.predict_proba(X_test)
prediction = pd.DataFrame(y_pred, columns=clf.classes_)
prediction.to_csv('result.csv')
# print("Log loss: ",log_loss(y_test, y_pred))

# y_pred = clf.predict(X_train)
# print("Log loss: ",log_loss(y_test, y_pred))

# print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))
# print("Precision: ",metrics.precision_score(y_test,y_pred,average="weighted"))
# print("F1 Score: ",metrics.f1_score(y_test,y_pred,average="weighted"))

