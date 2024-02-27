# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The problem statement for developing a neural network regression model involves predicting a continuous value output based on a set of input features. In regression tasks, the goal is to learn a mapping from input variables to a continuous target variable.
This neuron network model named 'AI' consists of 5 layers (1 input layer, 1 output layer and 3 hidden layers). The first hidden layer has units, second with 3 units, third with 4 units and output layer with 1 unit. All the hidden layers consist of activation function 'relu' Rectified Linear Unit.

The relationship between the input and the output is given by :

Output = Input * 5 + 7
in the dataset and applying 2000 epochs to minimize the error and predict the output.

## Neural Network Model
![image](https://github.com/DHARINIPV/basic-nn-model/assets/119400845/a2002189-3556-4c1d-bbee-3ddfd44a2864)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Dharini PV
### Register Number: 212222240024

### Dependencies
```python
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
```
### Data From Sheets
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('dl_exp1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
```
### Data Visualization
```python
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})
df
x=df[['INPUT']].values
y=df[['OUTPUT']].values
```
### Data split and Preprocessing
```python
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train1=scaler.transform(x_train)
```
### Regressive Model
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
AI=Sequential([
    Dense(units=5,activation='relu',input_shape=[1]),
    Dense(units=3,activation='relu'),
    Dense(units=4,activation='relu'),
    Dense(units=1)
])
AI.compile(optimizer='rmsprop',loss='mse')
AI.fit(x_train1,y_train,epochs=2000)
```
### Loss calculation
```python
loss_df = pd.DataFrame(AI.history.history)
loss_df.plot()
```
### Evaluate the model
```python
x_test1 = scaler.transform(x_test)
AI.evaluate(x_test1,y_test)
```
### Prediction
```python
x_n1 = [[5]]
x_n1_1 = scaler.transform(x_n1)
AI.predict(x_n1_1)
```
## Dataset Information
![image](https://github.com/DHARINIPV/basic-nn-model/assets/119400845/703fbae4-61ad-4d0b-a82f-820d63ca5784)

## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://github.com/DHARINIPV/basic-nn-model/assets/119400845/c713f5a7-baae-4d7c-8ae7-8b4504a7b09d)

### Training
![image](https://github.com/DHARINIPV/basic-nn-model/assets/119400845/a4a3c78b-cc66-4072-b62a-923e513d60db)

### Test Data Root Mean Squared Error
![image](https://github.com/DHARINIPV/basic-nn-model/assets/119400845/0348e31d-9684-4b01-a848-519f2b755637)

### New Sample Data Prediction
![image](https://github.com/DHARINIPV/basic-nn-model/assets/119400845/17ad74dd-495a-4599-a016-f4e07e2592ca)

![image](https://github.com/DHARINIPV/basic-nn-model/assets/119400845/62d6c6fe-a5e2-4549-858b-fc467409ab97)

## RESULT
A neural network regression model for the given dataset is developed and the prediction for the given input is obtained.
