import numpy as np
import matplotlib.pyplot as plt
import struct
import os

data_dir = r"D:\deep-learning\mnist\data\MNIST\raw"


train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte")
train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte")

def load_mnist_images(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"MNIST image file not found: {filename}")
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in MNIST image file {filename}")
        print(f"Loading {num_images} images ({rows}x{cols})...")
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows * cols).astype(np.float32) / 255.0
        print("Image loading complete.")
        return images

def load_mnist_labels(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"MNIST label file not found: {filename}")
    with open(filename, 'rb') as f:
        magic, num_items = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in MNIST label file {filename}")
        print(f"Loading {num_items} labels...")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        print("Label loading complete.")
        return labels

def create_one_hot(labels, num_classes=10):
    one_hot_labels = np.zeros((len(labels), num_classes), dtype=np.float32)
    one_hot_labels[np.arange(len(labels)), labels] = 1.0
    return one_hot_labels

def he_normal_init(fan_in, fan_out):
    stddev = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out).astype(np.float32) * stddev

def reLU(z):
    return np.maximum(0, z)

def reLU_derivative(a):
    return (a > 0).astype(float)

def softmax(z_in):
    if z_in.ndim == 1:
        z_in = z_in.reshape(1, -1)
    stable_z = z_in - np.max(z_in, axis=1, keepdims=True)
    expo = np.exp(stable_z)
    return expo / np.sum(expo, axis=1, keepdims=True)

def cross_entropy_loss(y_target_batch, prediction_batch):
    num_samples = y_target_batch.shape[0]
    prediction_batch = np.clip(prediction_batch, 1e-12, 1.0 - 1e-12)
    log_likelihood = -np.sum(y_target_batch * np.log(prediction_batch), axis=1)
    loss = np.mean(log_likelihood)
    return loss

X_train = load_mnist_images(train_images_path)
y_train_labels = load_mnist_labels(train_labels_path)
y_train_one_hot = create_one_hot(y_train_labels, 10)

input_size = 784
hidden_size = 128
output_size = 10

W_h = he_normal_init(input_size, hidden_size)
b_h = np.zeros(hidden_size, dtype=np.float32)
W_o = he_normal_init(hidden_size, output_size)
b_o = np.zeros(output_size, dtype=np.float32)

def forward_pass(X_batch, W_h_current, b_h_current, W_o_current, b_o_current):
    z_h = np.dot(X_batch, W_h_current) + b_h_current
    a_h = reLU(z_h)
    z_o = np.dot(a_h, W_o_current) + b_o_current
    prediction = softmax(z_o)
    return prediction, a_h

def backward_pass(X_batch, y_target_batch, prediction, a_h, W_o_current):
    num_samples = X_batch.shape[0]
    delta_output = prediction - y_target_batch
    grad_W_o = np.dot(a_h.T, delta_output) / num_samples
    grad_b_o = np.sum(delta_output, axis=0) / num_samples
    error_hidden_layer = np.dot(delta_output, W_o_current.T)
    delta_hidden = error_hidden_layer * reLU_derivative(a_h)
    grad_W_h = np.dot(X_batch.T, delta_hidden) / num_samples
    grad_b_h = np.sum(delta_hidden, axis=0) / num_samples
    return grad_W_h, grad_b_h, grad_W_o, grad_b_o

learning_rate = 0.3
epochs = 20
batch_size = 64

loss_history = []
num_samples_total = X_train.shape[0]
num_batches = num_samples_total // batch_size


for epoch in range(epochs):
    permutation = np.random.permutation(num_samples_total)
    X_shuffled = X_train[permutation]
    y_shuffled = y_train_one_hot[permutation]
    epoch_loss = 0.0

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]

        prediction, a_h = forward_pass(X_batch, W_h, b_h, W_o, b_o)
        batch_loss = cross_entropy_loss(y_batch, prediction)
        epoch_loss += batch_loss
        grad_W_h, grad_b_h, grad_W_o, grad_b_o = backward_pass(X_batch, y_batch, prediction, a_h, W_o)

        W_h -= learning_rate * grad_W_h
        b_h -= learning_rate * grad_b_h
        W_o -= learning_rate * grad_W_o
        b_o -= learning_rate * grad_b_o

    average_epoch_loss = epoch_loss / num_batches
    loss_history.append(average_epoch_loss)
    print(f"Epoch {epoch}/{epochs-1}, Average Loss: {average_epoch_loss:.6f}")

print("--- Training Finished ---")
print(f"Final Average Loss: {loss_history[-1]:.6f}")

num_test_samples = 5
X_test_sample = X_train[:num_test_samples]
y_test_labels_sample = y_train_labels[:num_test_samples]
final_predictions, _ = forward_pass(X_test_sample, W_h, b_h, W_o, b_o)

plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Average Cross-Entropy Loss")
plt.title("MLP Training Loss (MNIST)")
plt.grid(True)
plt.show()