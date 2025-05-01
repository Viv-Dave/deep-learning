import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x1 = np.array([0,0,1,1])
x2 = np.array([0,1,0,1])
w1 = np.random.randn(1)
w2 = np.random.randn(1)
b = np.random.randn(1)
# w1, w2 = 1,1
# b = -1.5

def train(w1, w2, x1, x2, b):
    lr = 1e-1
    # AND TARGET
    y_target = np.array([0, 0, 0, 1])
    # OR TARGET
    # y_target = np.array([0, 1, 1, 1])
    
    for epoch in range(50):
        total_absolute_error_in_epoch = 0
        for i in range(len(x1)):
            target = y_target[i]
            z = w1.item() * x1[i] + w2.item() * x2[i] + b.item()
            prediction = 1.0 if z >= 0 else 0.0
            error = target - prediction
            total_absolute_error_in_epoch += abs(error)

            print("EPOCH NUMBER:", epoch)
            w1 = w1 + lr * x1[i] * error
            w2 = w2 + lr * x2[i] * error
            b = b + lr * error

            print("W1:", w1, "W2:", w2, "b:", b)

        if total_absolute_error_in_epoch == 0 and epoch > 0:
            print(f"\nConverged successfully at epoch {epoch}!")
            break  

    return w1, w2, b

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