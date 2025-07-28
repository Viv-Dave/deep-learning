import torch.nn as nn
import torchvision as torchvision 
import numpy as np
import pandas as pd 
import cv2 as cv2 
import os 
import torch as torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim 
from torchvision import models 
# train_path = "ConvNet01/dataset/dogs_vs_cats/train"
# test_path = "ConvNet01/dataset/dogs_vs_cats/test"
# print(f"Path for train loaded successfully: {train_path}")

# def get_image_name(image_directory):
#     image_names = []
#     suffix = '.jpg'
#     for filename in os.listdir(image_directory):
#         if os.path.isfile(os.path.join(image_directory,filename)) and \
#         filename.lower().endswith(suffix):
#             image_names.append(filename)
#     return image_names

# cat_path = "D:/deep-learning/ConvNet01/dataset/dogs_vs_cats/train/cats"
# dog_path = "D:/deep-learning/ConvNet01/dataset/dogs_vs_cats/train/dogs"
# cat_list = np.array(get_image_name(image_directory=cat_path))
# dog_list = np.array(get_image_name(image_directory=dog_path))
# cat_df = pd.DataFrame(cat_list)
# cat_df["Value"] = 0
# dog_df = pd.DataFrame(dog_list)
# dog_df["Value"] = 1
# label = pd.concat([cat_df, dog_df]).sample(frac=1).reset_index(drop=True)

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# train_dataset = torchvision.datasets.ImageFolder(root="D:/deep-learning/ConvNet01/dataset/dogs_vs_cats/train", transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# print(train_dataset.classes) 
# print(train_dataset.class_to_idx) 

# class ConvNet(nn.Module):
#     def __init__(self):
#         super (ConvNet, self).__init__()

#         self.conv1 = nn.Conv2d(3,8,3,padding=1)
#         self.pool = nn.MaxPool2d(2,2)
#         self.conv2 = nn.Conv2d(8,16,3,padding=1)
#         self.conv3 = nn.Conv2d(16,32,3,padding=1)
#         self.conv4 = nn.Conv2d(32,64,3,padding=1)
#         self.conv5 = nn.Conv2d(64,128,3,padding=1)
#         self.conv6 = nn.Conv2d(128,256,3,padding=1)
#         self.fc1 = nn.Linear(12544, 512)
#         self.fc2 = nn.Linear(512,64)
#         self.fc3 = nn.Linear(64,1)

#     def forward(self,x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = self.pool(F.relu(self.conv4(x)))
#         x = self.pool(F.relu(self.conv5(x)))
#         x = F.relu(self.conv6(x))
#         x = torch.flatten(x,1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# ConvNet = ConvNet()
# criterion = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(ConvNet.parameters(), lr=1e-2)


# for epoch in range(2):
#     running_loss = 0.0

#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data

#         optimizer.zero_grad()

#         outputs = ConvNet(inputs)

#         labels = labels.float().unsqueeze(1)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if i % 2000 == 1999:  
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0
# print('Finished Training')
import torch
import torchvision.models as models

# Load VGG16 with pre-trained ImageNet weights
vgg16_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
from torchvision.datasets import ox