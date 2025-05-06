import torch as torch
from torch import nn
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# x1 =  torch.tensor([0,1,0,1], dtype=torch.int32)
# x2 = torch.tensor([0,0,1,1], dtype=torch.int32)
# y_target = torch.tensor([[0],[0],[0],[1]], dtype=torch.int32)   
# X_input = torch.cat((x1,x2), dim=0)
# print(X_input)
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]],dtype=torch.float32)
xor_target = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)
or_target = torch.tensor([[0],[1],[1],[1]], dtype=torch.float32)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 3)
        self.output = nn.Linear(3,1)
        self.hidden_activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()
    def forward(self, x):
        x = self.hidden(x)
        z = self.hidden_activation(x)
        a = self.output(z)
        output = self.output_activation(a)
        return output

model =  NeuralNetwork().to(device)
print("Model Structure")
print(model)
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
def train(X, y,model, loss, optimizer):
    model.train()
    X, y = X.to(device), y.to(device)

    pred = model(X)
    loss = loss(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()
epochs = 50000
loss_history = []
print(f"<-------Starting training for {epochs} epoch's------->")
for t in range(epochs):
    current_loss = train(X, or_target, model, loss, optimizer)
    loss_history.append(current_loss)

    if t % 500 == 0: # Print every 500 epochs
        print(f"Epoch {t}, Loss: {current_loss:.6f}")
print("Done! \n")

print("<-------Testing------->\n")
model.eval()
with torch.no_grad():
    prediction = model(X)
    predicted_classes = (prediction >= 0.5).float()
print("Input | Target | Prediction | Predicted Class")
print("------|--------|------------|-----------------")
for i in range(X.shape[0]):
    print(f"{X[i].cpu().numpy()} | {or_target[i].cpu().numpy()[0]:<6} | {prediction[i].cpu().numpy()[0]:.4f}     | {int(predicted_classes[i].cpu().numpy()[0]):<15}")
