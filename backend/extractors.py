from transformers import BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BLIP captioner (lightweight base)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

# CLIP for image embeddings
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Sentence Transformer for text embeddings
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

def generate_caption_pil(img_pil, max_length=50):
    inputs = blip_processor(images=img_pil, return_tensors="pt").to(DEVICE)
    out = blip_model.generate(**inputs, max_new_tokens=max_length)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_caption(image_path, max_length=50):
    img = Image.open(image_path).convert('RGB')
    return generate_caption_pil(img, max_length)

def image_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    inputs = clip_processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb.cpu().numpy()[0].astype('float32')
    norm = np.linalg.norm(emb) + 1e-10
    return emb / norm

def text_embedding(text):
    emb = sbert.encode(text, convert_to_numpy=True)
    emb = emb.astype('float32')
    norm = (np.linalg.norm(emb) + 1e-10)
    return emb / norm
