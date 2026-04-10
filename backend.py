# backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# --- Load model ---
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location='cpu'))
model.eval()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

def detect_face(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return image  # no face detected, use original

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    return Image.fromarray(face)

# --- Grad-CAM helper ---
def generate_gradcam(image):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    image_tensor = transform(image).unsqueeze(0)
    output = model(image_tensor)
    pred_class = output.argmax()

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].detach().numpy()[0]
    acts = activations[0].detach().numpy()[0]

    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam,0)
    cam = cv2.resize(cam, (128,128))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    img = np.array(image.resize((128,128)))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    superimposed = heatmap*0.4 + img

    return output, superimposed

# --- API ---
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file).convert("RGB")
    
    output, heatmap_img = generate_gradcam(image)
    probs = F.softmax(output, dim=1)
    confidence, predicted = torch.max(probs,1)
    label = "FAKE" if predicted.item()==0 else "REAL"

    # Convert heatmap to list for JSON (or save as file for frontend)
    heatmap_list = np.array(heatmap_img).astype(int).tolist()

    return jsonify({
        "result": label,
        "confidence": round(confidence.item()*100,2),
        "heatmap": heatmap_list
    })

if __name__ == "__main__":
    app.run(debug=True)