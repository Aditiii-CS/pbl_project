import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
st.title("AI Fake Face Detection")

# Load model
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    if predicted.item() == 0:
        result = "FAKE ❌"
    else:
        result = "REAL ✅"

    st.subheader("Result")
    st.write(f"Prediction: {result}")
    st.write(f"Confidence: {confidence.item()*100:.2f}%")
    if confidence.item() < 0.6:
        st.warning("⚠️ Low confidence prediction. Result may not be reliable.")
    if predicted.item() == 0:
        st.error("⚠️ This image appears to be FAKE. Be cautious!")
    else:
        st.success("✅ This image appears to be REAL.")
    
    st.write(f"Confidence: {confidence.item()*100:.2f}%")
    
    st.download_button(
    label="Download Result",
    data=f"{result} ({confidence.item()*100:.2f}%)",
    file_name="result.txt"
)
    # --- Grad-CAM ---
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax()

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].detach().numpy()[0]
    acts = activations[0].detach().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (128, 128))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    img = np.array(image.resize((128, 128)))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed = heatmap * 0.4 + img

    st.image(superimposed.astype('uint8'), caption="Heatmap", use_column_width=True)

    from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # important for frontend connection

@app.route('/')
def home():
    return "Backend Running 🚀"

if __name__ == '__main__':
    app.run(debug=True)