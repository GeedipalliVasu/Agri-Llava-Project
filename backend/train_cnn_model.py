import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image

# -----------------------
# Configuration
# -----------------------
root_path = r"C:\Users\dell5518\OneDrive\Desktop\pro2\pro\backend\PlantVillage_Small" 
batch_size = 32
num_epochs = 10
learning_rate = 0.001
model_path = "cnn_leaf_model.pth"

# -----------------------
# Transforms
# -----------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# -----------------------
# Dataset
# -----------------------
full_dataset = datasets.ImageFolder(root=root_path, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class_names = full_dataset.classes
num_classes = len(class_names)
print("---" * 20)
print(f"✅ Found {num_classes} classes.")
print(f"🌿 Classes: {class_names}")
print("---" * 20)
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

# -----------------------
# Model Definition
# -----------------------
class LeafCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -----------------------
# Train Model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeafCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("🚀 Starting Training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Val Acc: {acc:.2f}%")

# -----------------------
# Save Model
# -----------------------
torch.save(model.state_dict(), model_path)
print(f"\n✅ Model saved as {model_path}")

# -----------------------
# Test Loop
# -----------------------
model.eval()
print("\n🌿 Enter image path to test (or type 'exit' to stop):")

while True:
    img_path = input("Image path: ").strip()
    if img_path.lower() == "exit":
        print("👋 Exiting test loop.")
        break

    if not os.path.exists(img_path):
        print("⚠️ File not found. Try again.")
        continue

    try:
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            
            # --- MODIFIED: Get both probability and index ---
            probs = torch.softmax(outputs, dim=1)
            confidence_tensor, predicted_index_tensor = torch.max(probs, 1)
            
            label = class_names[predicted_index_tensor.item()]
            confidence_percent = f"{confidence_tensor.item() * 100:.2f}%"
            # --------------------------------------------------

        # <-- MODIFIED: Updated print statement
        print(f"✅ Prediction: {label.upper()} (Confidence: {confidence_percent})") 
    except Exception as e:
        print("❌ Error:", e)