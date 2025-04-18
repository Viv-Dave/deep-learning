# Implementing a face recogniser from scratch using encoding. 
# About the neural network: 3 layer neural network
# Input Layer: Image size
# Output Layer: Binary(y = 0 for incorrect, y = 1 for correct) 
# Activation Function: Sigmoid for classifcation between 0 and 1, if output>0.5 then y = 1 else 0
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import os
