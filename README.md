# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the placement dataset and convert the target variable into binary form.
2. Normalize the input features and initialize weights and bias.
3. Apply the sigmoid function and update parameters using gradient descent.
4. Predict the output and evaluate the model using confusion matrix, accuracy, precision, recall, and classification report.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SHAJIVE KUMAR J
RegisterNumber:  212225230258

```
~~~
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv(r"C:\Users\acer\Downloads\Placement_Data (1).csv")

# Convert target variable to binary
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

# Select features and target
X = data[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']].values
y = data['status'].values

# Feature scaling
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression using Gradient Descent
def logistic_regression(X, y, lr, epochs):
    m, n = X.shape
    w = np.zeros(n)
    b = 0

    for _ in range(epochs):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)

        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)

        w -= lr * dw
        b -= lr * db

    return w, b

# Train model
weights, bias = logistic_regression(X, y, lr=0.01, epochs=1000)

# Predictions
z = np.dot(X, weights) + bias
y_pred = sigmoid(z)
y_pred_class = (y_pred >= 0.5).astype(int)

# Confusion Matrix values
TP = np.sum((y == 1) & (y_pred_class == 1))
TN = np.sum((y == 0) & (y_pred_class == 0))
FP = np.sum((y == 0) & (y_pred_class == 1))
FN = np.sum((y == 1) & (y_pred_class == 0))

# Metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)   # Sensitivity

print("Confusion Matrix:")
print([[TN, FP],
       [FN, TP]])

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Sensitivity (Recall):", recall)

# Classification Report
print("\nClassification Report:")
print(classification_report(y, y_pred_class))
~~~
## Output:
![alt text](<ML_EX6 PNG.png>)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

