from src.data_preprocessing.floor_segmentation import segment_floor
from src.data_preprocessing.embedding import get_image_embedding
from src.vector_store.retrieval import semantic_retrieval
from PIL import Image
import torch
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()


def clip_pairwise_similarity(query_img, candidate_img):
    inputs = clip_processor(
        images=[query_img, candidate_img],
        return_tensors="pt"
    )

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features.pooler_output 

    features = features / features.norm(dim=-1, keepdim=True)
    similarity = torch.dot(features[0], features[1])

    return similarity.item()

def hybrid_image_search(query_img: Image.Image, image_base_path, top_k=20):

    candidates = semantic_retrieval(query_img)
    reranked = []
    for match in candidates:
        img_name = match["metadata"]["name"]
        candidate_path = os.path.join(image_base_path, img_name)
        candidate_img = Image.open(candidate_path).convert("RGB")

        score = clip_pairwise_similarity(query_img, candidate_img)
        reranked.append({
            "id": match["id"],
            "image_name": img_name,
            "semantic_score": match["score"],
            "hybrid_score": score
        })

    reranked.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return reranked