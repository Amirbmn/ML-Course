import numpy as np
import pandas as pd
data = pd.read_csv('advertising_standardized.csv')
x1=data['TV'].values
x2=data['Radio'].values
x3=data['Newspaper'].values
Y=data['Sales'].values
X=np.array([x1,x2,x3])
X=X.transpose()
X_bias = np.c_[np.ones((X.shape[0], 1)), X]

learning_rate=0.001
epochs=100_000
n=len(X)

w1=float(input('insert w1 :'))
w2=float(input('insert w2 :'))
w3=float(input('insert w3 :'))
bias=float(input('insert bias :'))
theta=np.array([bias,w1,w2,w3])

for epoch in range(epochs):
        h=bias+w1*X_bias[:,1] +w2*X_bias[:,2]+w3*X_bias[:,3] 
        errors=h-Y
        grad_bias=(1/n)*np.sum(errors)
        grad_w1=(1/n)*np.sum(errors*X_bias[:,1])
        grad_w2=(1/n)*np.sum(errors*X_bias[:,2])
        grad_w3=(1/n)*np.sum(errors*X_bias[:,3])

        bias-= (learning_rate * grad_bias)
        w1-= (learning_rate * grad_w1)
        w2-= (learning_rate * grad_w2)
        w3-= (learning_rate * grad_w3)

print(f"w1 : {w1:.4f}")
print(f"w2 : {w2:.4f}")
print(f"w3 :{w3:.4f}")
print(f"bias :{bias:.4f}")
