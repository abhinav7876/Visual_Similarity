import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def get_image_embedding(img: Image.Image):
    inputs = clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    embedding = outputs.pooler_output 
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    embedding = embedding.squeeze(0).cpu().tolist()
    return embedding

