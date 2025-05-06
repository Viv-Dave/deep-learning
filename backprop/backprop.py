import numpy as np
import matplotlib.pyplot as plt

x1_inputs = np.array([0,0,1,1])
x2_inputs = np.array([0,1,0,1])
X = np.array([x1_inputs, x2_inputs]).T
y_target = np.array([[0], [0], [0], [1]])
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(s):
    return s * (1.0 - s)

def xavier_normal_init(rows, cols):
    fan_in = rows
    fan_out = cols
    stddev = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(rows, cols) * stddev

def zeros_init(size):
    if isinstance(size, int):
        return np.zeros(size)
    else:
        return np.zeros(tuple(size))
input_size = 2
hidden_size = 2
output_size = 1

W_h = xavier_normal_init(input_size, hidden_size)
b_h = zeros_init(hidden_size)

W_o = xavier_normal_init(hidden_size, output_size)
b_o = zeros_init(output_size)

def forward_pass(X_batch, W_h_current, b_h_current, W_o_current, b_o_current):
    z_h = np.dot(X_batch, W_h_current) + b_h_current
    a_h = sigmoid(z_h)
    z_o = np.dot(a_h, W_o_current) + b_o_current
    prediction = sigmoid(z_o)
    return prediction, a_h

def backward_pass(X_batch, y_target_batch, prediction, a_h, W_o_current):
    num_samples = X_batch.shape[0]
    error_output_signal = prediction - y_target_batch
    delta_output = error_output_signal * sigmoid_derivative(prediction)
    error_hidden_layer = np.dot(delta_output, W_o_current.T)
    delta_hidden = error_hidden_layer * sigmoid_derivative(a_h)
    grad_W_o = np.dot(a_h.T, delta_output) / num_samples
    grad_b_o = np.sum(delta_output, axis=0) / num_samples
    grad_W_h = np.dot(X_batch.T, delta_hidden) / num_samples
    grad_b_h = np.sum(delta_hidden, axis=0) / num_samples
    return grad_W_h, grad_b_h, grad_W_o, grad_b_o

learning_rate = 0.85
epochs = 10000
loss_history = []

print("\n--- Starting Training ---")
for epoch in range(epochs):
    prediction, a_h = forward_pass(X, W_h, b_h, W_o, b_o)
    loss = 0.5 * np.mean((y_target - prediction)**2)
    loss_history.append(loss)
    grad_W_h, grad_b_h, grad_W_o, grad_b_o = backward_pass(X, y_target, prediction, a_h, W_o)
    W_h = W_h - learning_rate * grad_W_h
    b_h = b_h - learning_rate * grad_b_h
    W_o = W_o - learning_rate * grad_W_o
    b_o = b_o - learning_rate * grad_b_o
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

print("--- Training Finished ---")
print(f"Final Loss: {loss_history[-1]:.6f}")

print("\n--- Testing Final Model ---")
final_predictions, _ = forward_pass(X, W_h, b_h, W_o, b_o)
print("Input | Target | Prediction | Rounded")
print("------|--------|------------|---------")
for i in range(X.shape[0]):
    pred_val = final_predictions[i][0]
    rounded = 1 if pred_val >= 0.5 else 0
    print(f"{X[i]} | {y_target[i][0]}      | {pred_val:.4f}     | {rounded}")

plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error Loss")
plt.title("MLP Training Loss (XOR Gate - Xavier Init)")
plt.grid(True)
plt.show()