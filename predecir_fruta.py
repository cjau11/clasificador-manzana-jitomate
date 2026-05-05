import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys

# --- MiniResNet (igual que en entrenamiento) ---

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BasicBlock(3, 16)
        self.pool = nn.MaxPool2d(2, 2)
        self.layer2 = BasicBlock(16, 32)
        self.layer3 = BasicBlock(32, 64)
        self.fc = nn.Linear(64 * 16 * 16, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


# Transformaciones idénticas a test_transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MiniResNet().to(device)
model.load_state_dict(torch.load("modelo_frutas.pth", map_location=device))
model.eval()

if len(sys.argv) < 2:
    print("Uso: python predecir_fruta.py imagen.jpg")
    exit()

img_path = sys.argv[1]
img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    salida = model(img)
    _, pred = torch.max(salida, 1)

clases = ["Manzana", "Jitomate"]
print(f"Predicción: {clases[pred.item()]}")
