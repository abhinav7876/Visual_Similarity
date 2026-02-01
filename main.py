from src.data_preprocessing.floor_segmentation import segment_floor
from src.vector_store.hybrid_reranking import hybrid_image_search
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import io
import numpy as np
from PIL import Image


app = FastAPI(title="Floor Similarity Search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
image_base_path=r"data\Assess\sku"

@app.post("/search")
async def search_floor_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    floor_region = segment_floor(image)
    results = hybrid_image_search(floor_region, image_base_path)
    response = [
        {
            "id": match["id"],
            "score": round(match["hybrid_score"], 4),
            "name": match["image_name"]
        }
        for match in results
    ]
    return response