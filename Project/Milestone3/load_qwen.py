from transformers import AutoProcessor, AutoModelForMultimodalLM
import torch


# utils/qwen_model.py
def load_qwen_model(model_name="Qwen/Qwen3-VL-Embedding-8B", device="cuda"):
    # Load model and processor
    # Return model ready for inference
    processor = AutoProcessor.from_pretrained(model_name)

    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    return processor, model
        

def get_embeddings(mask_image,processor, model):
    # Process mask → embeddings
    inputs = processor(images=mask_image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings

def cosine_similarity(embeddings1, embeddings2):
    # Compute cosine similarity between two embeddings
    embeddings1 = embeddings1 / embeddings1.norm(dim=-1, keepdim=True)
    embeddings2 = embeddings2 / embeddings2.norm(dim=-1, keepdim=True)
    return torch.mm(embeddings1, embeddings2.t())