import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. TRANSFORMACIONES MEJORADAS PARA FONDOS REALES

train_transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.RandomResizedCrop(128, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    
    transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),

    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Transformaciones del conjunto de prueba
test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 2. CARGA DE DATASETS

train_data = datasets.ImageFolder(root='frutas_cnn/dataset/train', transform=train_transform)
test_data = datasets.ImageFolder(root='frutas_cnn/dataset/test', transform=test_transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

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

model = MiniResNet()

# 4. ENTRENAMIENTO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 12
for epoch in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Época {epoch+1}/{epochs} - Pérdida: {running_loss/len(train_loader):.4f}")

print("\nEntrenamiento terminado.")

# 5. EVALUACIÓN

model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = correct / total * 100
print(f"\nPrecisión final: {acc:.2f}%")

# 6. GUARDAR MODELO

torch.save(model.state_dict(), "modelo_frutas.pth")
print("Modelo guardado como modelo_frutas.pth")
