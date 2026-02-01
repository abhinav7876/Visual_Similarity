from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
load_dotenv()
from PIL import Image
from src.data_preprocessing.embedding import get_image_embedding

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc=Pinecone(api_key=PINECONE_API_KEY)
index_name="visual-similarity-index"
if index_name not in pc.list_indexes():
    pc.create_index(name=index_name,
                    dimension=512,metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))   
index = pc.Index(index_name)

all_embeddings=[]
metadata=[]
ids=[]
folder_path=r"E:\Visual_Similarity\data\Assess\sku"
for file in os.listdir(folder_path):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(folder_path, file)
        img = Image.open(path).convert("RGB")
        embedding=get_image_embedding(img)
        all_embeddings.append(embedding)
        meta_d={
            "name":file,
            "type": "floor",
            "material": "wood",
            "source": "clipseg"
        }
        metadata.append(meta_d)
        name = os.path.splitext(file)[0]
        ids.append(f"floor_{name}")

index.upsert([
    {
        "id": ids[i],
        "values": all_embeddings[i],
        "metadata": metadata[i]
    }
    for i in range(len(all_embeddings))
])
print("Indexing complete.")

