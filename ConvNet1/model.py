import PIL.Image
import torch 
import cv2 as cv2
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import PIL as PIL 
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])
path = "D:\deep-learning\ConvNet1\data\seg_train\seg_train"
dataset = torchvision.datasets.ImageFolder(path)
print(dataset.classes)        
print(dataset.class_to_idx)  
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def show_image(image, landmarks):
    plt.imshow(image)
