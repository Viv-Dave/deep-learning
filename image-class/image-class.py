# # Implementing a face recognising basic non convolutional neural net from scratch with triplet encoding. 
# # About the neural network: 1 input layer 3 hidden layers 1 output neuron
# # Input Layer: Flattened Image
# # Output Layer: Binary(y = 0 for incorrect, y = 1 for correct) 
# # Activation Function: Sigmoid for classifcation between 0 and 1, if output>0.5 then y = 1 else 0
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import pandas as pd
# import cv2 as cv2
# import os as os
# from torch import nn
# from torch.utils.data import dataloader
# from torchvision.transforms import ToTensor
# path = r"D:\deep-learning\image-class\dataset\CelebA FR Triplets\CelebA FR Triplets\images"

# triplet_path = r"image-class\dataset\CelebA FR Triplets\CelebA FR Triplets\triplets.csv"

# filename = r"image-class/dataset/CelebA FR Triplets/CelebA FR Triplets/images/000015.jpg"

# df = pd.DataFrame({"anc", "pos", "neg"})
# relation= pd.read_csv(triplet_path)
# # Function to convert and resize images into a single defined size of 128x128. 
# # Flatten them to an array of 49152 pixel values into a 49152,1 vector.
# def convert(path, size=(128,128)):
#     processed_data = []
#     for index, rows in relation.iterrows():
#         print(f"Processing Index no: {index}")
#         anc_img,pos_img, neg_img = rows["anchor"],rows["pos"],rows["neg"]
#         anc_path = os.path.join(path, anc_img)
#         pos_path = os.path.join(path, pos_img)
#         neg_path = os.path.join(path, neg_img)
#         anc_image,pos_image, neg_image = cv2.imread(anc_path), cv2.imread(pos_path), cv2.imread(neg_path)
#         anc_image_resized = cv2.resize(anc_image, size)
#         pos_image_resized = cv2.resize(pos_image,size)
#         neg_image_resized = cv2.resize(neg_image, size)
#         new_rows =  (np.array(anc_image_resized).flatten(), np.array(pos_image_resized).flatten(),np.array(neg_image_resized).flatten())
#         processed_data.append(new_rows) 
#     return processed_data
# processed_data = np.array(convert(path))
# df.concat(processed_data)

# # Neural Network
# #https://aistudio.google.com/prompts/12YLr9eIgqq8wpSfYUliwaV7lnGU-hNVU








            


