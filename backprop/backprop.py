import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x1 = np.array([0,0,1,1])
x2 = np.array([0,1,0,1])
w1 = np.random.randn(1)*0.01
w2 = np.random.randn(1)*0.01
b = np.random.randn(1)
# w1, w2 = 1,1
# b = -1.5

def train(w1,w2, x1, x2, b):
    lr = 1e-2
    # AND TARGET
    # y_target = np.array([0,0,0,1])
    # OR TARGET
    y_target = np.array([0,1,1,1])
    for epoch in range(1000):
       for i in range(len(x1)):
           current_x1 = x1[i]
           current_x2 = x2[i]
           target = y_target[i]

           z = w1.item() * current_x1 + w2.item() * current_x2 + b.item()

           prediction = 1.0 if z >= 0 else 0.0
           error = target - prediction
           print("EPOCH NUMBER: ", epoch)
           w1 = w1 + lr*current_x1*error  #Manual Grdients
           w2 = w2 + lr*current_x2*error #Manual Gradients
           b = b + lr*error
           print("W1:", w1, "W2:", w2,"b:",b)
    return w1,w2,b
    
updated = train(w1, w2, x1, x2, b)
print(updated)
# single_perceptron = x1*w1+x2*w2+b
final_w1, final_w2, final_b = updated
final_equation = x1*final_w1+x2*final_w2+final_b
results = []
for z in final_equation:
  if z < 0:
    results.append("0")
  else:
    results.append("1")
print(results) 