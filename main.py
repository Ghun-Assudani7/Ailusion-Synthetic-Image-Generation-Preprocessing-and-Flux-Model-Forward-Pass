import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from huggingface_hub import login

# Set your Hugging Face API Token
login(HF_TOKEN)
os.environ["HUGGINGFACE_API_TOKEN"] = HF_TOKEN

# Load Stable Diffusion Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=HF_TOKEN).to(device)

# Define prompts
prompts = [
    "A A delicious strwberry cheesecake",
    "A sea monster destroys a fishing boat during a storm.",
    "A Van Gogh painting of an orangutan
"
]

# Generate images
generated_images = []
for i, prompt in enumerate(prompts):
    image = model(prompt).images[0]
    image_path = f"generated_image_{i+1}.png"
    image.save(image_path)
    generated_images.append(image)
    print(f"Saved {image_path}")

# Image Preprocessing Function
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Preprocess first generated image
preprocessed_image = preprocess_image("generated_image_1.png")
print("Preprocessed Image Shape:", preprocessed_image.shape)
