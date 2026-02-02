import torch
import numpy as np
from PIL import Image
import cv2
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model.eval()

def segment_floor(img: Image.Image):
    try:
        inputs = processor(text=[ "floor"],images=img,return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        mask = torch.sigmoid(logits)[0].cpu().numpy()
        img_np = np.array(img)
        mask_resized =cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))
        binary_mask = mask_resized > 0.5
        ys, xs = np.where(binary_mask)
        crop = img_np[min(ys):max(ys), min(xs):max(xs)]
        floor_img= Image.fromarray(crop)
        #floor_region = img_np * binary_mask[:, :, None]
        #return floor_region
        return floor_img
    except Exception as e:
        print(f"Error in floor segmentation: {e}")
        return None

