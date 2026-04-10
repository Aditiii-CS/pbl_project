import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Hook for gradients
gradients = []
activations = []

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

def forward_hook(module, input, output):
    activations.append(output)

# Hook last conv layer
target_layer = model.features[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Input image
img_path = input("Enter image path: ")
image = Image.open(img_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

# Forward
output = model(image_tensor)

# Backward (for predicted class)
pred_class = output.argmax()
model.zero_grad()
output[0, pred_class].backward()

# Get gradients & activations
grads = gradients[0].detach().numpy()[0]
acts = activations[0].detach().numpy()[0]

# Compute weights
weights = np.mean(grads, axis=(1, 2))

# Create heatmap
cam = np.zeros(acts.shape[1:], dtype=np.float32)
for i, w in enumerate(weights):
    cam += w * acts[i]

cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (128, 128))
cam = cam - np.min(cam)
cam = cam / np.max(cam)

# Convert original image
img = cv2.imread(img_path)
img = cv2.resize(img, (128, 128))

# Apply heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
superimposed = heatmap * 0.4 + img

# Show
plt.imshow(cv2.cvtColor(superimposed.astype('uint8'), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Grad-CAM Heatmap")
plt.show()