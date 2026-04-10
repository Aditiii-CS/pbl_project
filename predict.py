import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load model
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load image
img_path = input("Enter image path: ")
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0)

# Predict
import torch.nn.functional as F

with torch.no_grad():
    output = model(image)
    probs = F.softmax(output, dim=1)
    confidence, predicted = torch.max(probs, 1)

if predicted.item() == 0:
    print(f"Prediction: FAKE ❌ ({confidence.item()*100:.2f}% confident)")
else:
    print(f"Prediction: REAL ✅ ({confidence.item()*100:.2f}% confident)")