from src.data_preprocessing.floor_segmentation import segment_floor
from src.data_preprocessing.embedding import get_image_embedding
from PIL import Image
import os
from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone
import numpy as np
from PIL import Image

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc=Pinecone(api_key=PINECONE_API_KEY) 

index_name="visual-similarity-index"
index = pc.Index(index_name)




def semantic_retrieval(img: Image.Image):
    floor_region = segment_floor(img)
    embedding=get_image_embedding(floor_region)
    results = index.query(
        vector=embedding,
        top_k=20,
        include_metadata=True, 
    )
    return results["matches"]