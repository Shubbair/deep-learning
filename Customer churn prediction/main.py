"""
  *** Hussain Salih Mahdi ***
  _________Shubbair__________

TODO Customer Churn Prediction

input : csv file of customer info

"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

data = pd.read_csv('input.csv')

# drop not important field
data.drop('customerID', axis='columns', inplace=True)

d1 = data[data.TotalCharges != ' ']

d1.TotalCharges = pd.to_numeric(d1.TotalCharges)
print(d1.head())

d1.replace('No internet service', 'No', inplace=True)
d1.replace('No phone service', 'No', inplace=True)

yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
for col in yes_no_columns:
    d1[col].replace({'Yes': 1, 'No': 0}, inplace=True)

d1['gender'].replace({'Female': 1, 'Male': 0}, inplace=True)
print(d1.head())

d2 = pd.get_dummies(data=d1, columns=['InternetService', 'Contract', 'PaymentMethod'])

cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
scalar = MinMaxScaler()
d2[cols_to_scale] = scalar.fit_transform(d2[cols_to_scale])

X = d2.drop('Churn', axis='columns')
y = d2['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.fit(X_train, y_train, epochs=100))

print(model.evaluate(X_test, y_test))

yp = model.predict(X_test)
print(yp[:5])

# to show the result as 0,1
y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

print(classification_report(y_test, y_pred))

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
