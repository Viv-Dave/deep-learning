import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
#      STEP 1: RE-DEFINE THE MODEL ARCHITECTURE
# ==============================================================================
# This class MUST be an exact copy of the one you used for training.
# Any change here will cause an error when loading the weights.

class FaceRecognition(nn.Module):
    def __init__(self):
        super(FaceRecognition, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=8, kernel_size=3, padding=1) #448*448
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16, kernel_size=3, padding=1) #224*224
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(in_channels=16,out_channels=32, kernel_size=3, padding=1) #112*112
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) #56*56
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) #28*28
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) #14*14
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1) #7*7
        self.bn7 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2,2)
        
        self.age_head = nn.Sequential(
            nn.Linear(in_features=4608, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=7)
        )
        self.ethnicity_head = nn.Sequential(
            nn.Linear(in_features=4608, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=5)
        )

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.pool(F.relu(self.bn7(self.conv7(x))))

        features = torch.flatten(x, 1)
        age_pred = self.age_head(features)
        ethnicity_pred = self.ethnicity_head(features)

        return age_pred, ethnicity_pred

# These MUST match the classes and transforms used during training.
AGE_BINS = ['0-4', '5-12', '13-19', '20-29', '30-39', '40-59', '60+']
ETHNICITY_LABELS = ['White', 'Black', 'Asian', 'Indian', 'Others'] # Based on UTKFace dataset mapping

# The transform for a single image should be the same as your validation/test transform
inference_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==============================================================================
#      STEP 3: LOAD THE MODEL AND WEIGHTS
# ==============================================================================

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model architecture
# Make sure the number of classes matches what you trained with
model = FaceRecognition().to(device)

# Path to your saved model weights
MODEL_PATH = 'D:/deep-learning/ConvNet1/age_race_weights2.pth'

# Load the state dictionary
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model weights loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model weights not found at {MODEL_PATH}")
    exit() # Exit if the model can't be loaded

# Set the model to evaluation mode
model.eval()


def predict(image_path):
    """
    Loads an image, preprocesses it, and returns the predicted age and ethnicity.
    """
    try:
        # Open the image using PIL
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        return f"Error: Image not found at {image_path}"
    
    # Preprocess the image and add a batch dimension (B, C, H, W)
    image_tensor = inference_transform(image).unsqueeze(0).to(device)
    
    # Make predictions
    with torch.no_grad():
        age_logits, ethnicity_logits = model(image_tensor)
        
    # --- Process Age Prediction ---
    # Apply softmax to get probabilities
    age_probs = F.softmax(age_logits, dim=1)
    # Get the index of the highest probability
    predicted_age_index = age_probs.argmax(1).item()
    # Map the index to the human-readable label
    predicted_age = AGE_BINS[predicted_age_index]
    age_confidence = age_probs.max().item() * 100
    
    # --- Process Ethnicity Prediction ---
    ethnicity_probs = F.softmax(ethnicity_logits, dim=1)
    predicted_ethnicity_index = ethnicity_probs.argmax(1).item()
    predicted_ethnicity = ETHNICITY_LABELS[predicted_ethnicity_index]
    ethnicity_confidence = ethnicity_probs.max().item() * 100
    
    # Display the original image
    plt.imshow(image)
    plt.axis('off')
    plt.title("Input Image")
    plt.show()

    # Print the results
    print("\n--- Predictions ---")
    print(f"Predicted Age Range: {predicted_age} (Confidence: {age_confidence:.2f}%)")
    print(f"Predicted Ethnicity: {predicted_ethnicity} (Confidence: {ethnicity_confidence:.2f}%)")


# ==============================================================================
#      STEP 5: RUN A PREDICTION
# ==============================================================================

# Provide the path to an image you want to test
TEST_IMAGE_PATH = 'D:/deep-learning/ConvNet1/testsubject.jpg'
predict(TEST_IMAGE_PATH)