import pandas as pd
import numpy as np

train = pd.read_csv("C://Users//laxmi//OneDrive//Documents//BA//Loan Status Prediction//train.csv")
train

train.columns

train= train.fillna(0)
train

x = train[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Credit_History']]
train['dep'] = np.where(train['Loan_Status'] == 'Y',1,0)
y = train['dep']

from sklearn.linear_model import LogisticRegression
loan = LogisticRegression()
loan.fit(x,y)

y_pred = loan.predict(x)
y_pred

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y,y_pred)
print (confusion_matrix)
test = pd.read_csv("C://Users//laxmi//OneDrive//Documents//BA//Loan Status Prediction//test.csv")
test

test= test.fillna(0)
test

x_test = test[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Credit_History']]
x_test

y_pred1 = loan.predict(x_test)
y_pred1



    


