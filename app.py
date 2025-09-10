import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# -------------------------------
# 1. Setup
# -------------------------------
st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf image and I’ll predict the disease!")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Load model
@st.cache_resource
def load_model():
    num_classes = len(class_names)
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("resnet_finetuned.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Preprocessing (must match training pipeline)
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# -------------------------------
# 2. Upload Image
# -------------------------------
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)  # fixed warning

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, pred_class = torch.max(probs, dim=0)

    st.markdown(f"### 🌱 Prediction: **{class_names[pred_class]}**")
    st.write(f"Confidence: {confidence.item()*100:.2f}%")

    # -------------------------------
    # 3. Grad-CAM (optional)
    # -------------------------------
    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None

            target_layer.register_forward_hook(self.save_activation)
            target_layer.register_full_backward_hook(self.save_gradient)  # fixed PyTorch warning

        def save_activation(self, module, input, output):
            self.activations = output.detach()

        def save_gradient(self, module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def generate(self, input_image, class_idx=None):
            self.model.zero_grad()
            output = self.model(input_image)
            if class_idx is None:
                class_idx = output.argmax().item()
            loss = output[0, class_idx]
            loss.backward()

            gradients = self.gradients[0].cpu().numpy()
            activations = self.activations[0].cpu().numpy()
            weights = gradients.mean(axis=(1, 2))

            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * activations[i, :, :]

            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (224, 224))
            cam = cam - cam.min()
            cam = cam / cam.max()
            return cam

    # Generate Grad-CAM
    gradcam = GradCAM(model, model.layer4[-1])
    heatmap = gradcam.generate(input_tensor, pred_class.item())

    # Convert image for display
    img_np = np.array(image.resize((224, 224))) / 255.0
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(img_np)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM")

    st.pyplot(plt)
