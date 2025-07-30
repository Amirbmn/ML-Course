import numpy as np
import pandas as pd

data = pd.read_csv('advertising_standardized.csv')
X = data['TV'].values
Y = data['Sales'].values

w = float(input('weight: '))
b=float(input('bias: '))

learning_rate = 0.001
n = len(X)
epochs = 100_000

for epoch in range(epochs):
    y_pred = w * X + b
    dw = (2/n) * np.sum(X * (y_pred - Y))  
    db = (2/n) * np.sum(y_pred - Y)        
    w -= dw * learning_rate              
    b -= db * learning_rate              

final_pred = w * X + b
mse = sum((Y - final_pred)**2)/n

print('\nfinal result')
print(f"Weight : {w:.4f}")
print(f"Bias : {b:.4f}")
print(f"MSE : {mse:.4f}")