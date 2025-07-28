import torch 
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(12544, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 64)
        self.fc_bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = F.relu(self.bn6(self.conv6(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = F.relu(self.fc_bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = ConvNet().to(device)    
model.load_state_dict(
    torch.load(
        'D:/deep-learning/classifier/dogs_vs_cats_model4.pth',
        map_location=torch.device('cpu')
    )
)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
path = 'D:/deep-learning/classifier/cat.10.jpg'
dog_path = 'D:/deep-learning/classifier/dogbreed.jpg'
image = Image.open(dog_path)
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)
    probability = torch.sigmoid(output)
    if probability < 0.5:
        predicted_class = "Cat"
        confidence = (1 - probability) * 100
    else:
        predicted_class = "Dog"
        confidence = probability * 100
confidence = confidence.item()
print(f"This is a {predicted_class}.")
print(f"Confidence: {confidence:.2f}%")