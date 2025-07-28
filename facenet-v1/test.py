import torch 
from PIL import Image, ImageDraw
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class FaceRecog(nn.Module):
    def __init__(self):
        super(FaceRecog, self).__init__()

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
        
        self.fc1 = nn.Linear(in_features=512 * 3 * 3, out_features=1024)
        self.dropout = nn.Dropout(p=0.4)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=4)

    def forward(self,x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.pool(F.relu(self.bn7(self.conv7(x))))
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        predicted_bbox = torch.sigmoid(self.fc2(x))
        return predicted_bbox
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = FaceRecog().to(device)    
model.load_state_dict(
    torch.load(
        'D:/deep-learning/facenet-v1/model_weights.pth',
        map_location=torch.device('cpu')
    )
)
model.eval()
transform = transforms.Compose([
    transforms.Resize((448, 448)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

path = 'D:/deep-learning/facenet-v1/Kritika1.jpg'
image = Image.open(path)
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor).cpu().squeeze()

xmin_norm, ymin_norm, xmax_norm, ymax_norm = output.numpy()

img_w, img_h = image.size

xmin = int(xmin_norm * img_w)
ymin = int(ymin_norm * img_h)
xmax = int(xmax_norm * img_w)
ymax = int(ymax_norm * img_h)

print(f"Alternative Bounding Box: ({xmin}, {ymin}), ({xmax}, {ymax})")

draw = ImageDraw.Draw(image)
draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=5) # Use a different color
image.show()