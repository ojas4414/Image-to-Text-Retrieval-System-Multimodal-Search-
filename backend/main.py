import os
import sys
import shutil
import tempfile

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException

# Make the sibling modules importable whether run as a module or a script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extractors import image_embedding, text_embedding, generate_caption
from indexer import FaissIndex
from rag import generate_answer

# Local (non-Colab) filesystem layout.
# PROJECT_ROOT is the parent of this backend/ folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images")
INDEX_DIR = os.path.join(PROJECT_ROOT, "indexes")
META_PATH = os.path.join(PROJECT_ROOT, "metadata.csv")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

app = FastAPI(title="Cyberpunk Image Analyser")


def _save_upload_to_temp(upload: UploadFile) -> str:
    """Persist an uploaded file to a temporary path and return it."""
    suffix = os.path.splitext(upload.filename or "")[1] or ".png"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return tmp_path


@app.post("/ingest")
def ingest():
    """Run the indexing loop and build/save the FAISS indexes."""
    meta = pd.read_csv(META_PATH)

    image_vecs = []
    image_ids = []
    text_vecs = []
    text_ids = []

    for idx, row in meta.iterrows():
        img_path = os.path.join(PROJECT_ROOT, "data", "images", row["filename"])

        if not os.path.exists(img_path):
            print("Missing", img_path, "- skipping")
            continue

        img_emb = image_embedding(img_path)
        image_vecs.append(img_emb)
        image_ids.append(row["id"])

        desc = row.get("description")
        cap = row.get("caption")

        if pd.notna(desc) and isinstance(desc, str) and desc.strip():
            text = desc
        elif pd.notna(cap) and isinstance(cap, str) and cap.strip():
            text = cap
        else:
            print(f"Using BLIP for {row['id']}")
            text = generate_caption(img_path)

        txt_emb = text_embedding(text)
        text_vecs.append(txt_emb)
        text_ids.append(row["id"])

    if len(image_vecs) > 0:
        image_vecs = np.vstack(image_vecs).astype("float32")
        text_vecs = np.vstack(text_vecs).astype("float32")

        img_dim = image_vecs.shape[1]
        img_index = FaissIndex(img_dim)
        img_index.add(image_vecs, image_ids)
        img_index.save(os.path.join(INDEX_DIR, "image_index"))

        txt_dim = text_vecs.shape[1]
        txt_index = FaissIndex(txt_dim)
        txt_index.add(text_vecs, text_ids)
        txt_index.save(os.path.join(INDEX_DIR, "text_index"))

        return {"status": "ok", "indexed": len(image_ids), "ids": image_ids}

    return {"status": "empty", "message": "No images were indexed. Upload images and re-run."}


@app.post("/search")
async def search(file: UploadFile = File(...), top_k: int = Form(5)):
    """Take an image, return top-k similar images via CLIP."""
    if not os.path.exists(os.path.join(INDEX_DIR, "image_index.index")):
        raise HTTPException(status_code=400, detail="Image index not found. Run /ingest first.")

    tmp_path = _save_upload_to_temp(file)
    try:
        img_emb = image_embedding(tmp_path)

        img_idx = FaissIndex(img_emb.shape[0])
        img_idx.load(os.path.join(INDEX_DIR, "image_index"))

        sim_images = img_idx.search(img_emb, top_k=top_k)

        meta = pd.read_csv(META_PATH)
        results = []
        for img_id, score in sim_images:
            row = meta[meta["id"] == img_id]
            filename = row.iloc[0]["filename"] if not row.empty else None
            results.append({
                "id": img_id,
                "score": score,
                "filename": filename,
                "path": os.path.join(PROJECT_ROOT, "data", "images", filename) if filename else None,
            })

        return {"results": results}
    finally:
        os.remove(tmp_path)


@app.post("/query")
async def query(file: UploadFile = File(...), question: str = Form("")):
    """Full pipeline: caption -> retrieve -> generate_answer."""
    if not os.path.exists(os.path.join(INDEX_DIR, "text_index.index")):
        raise HTTPException(status_code=400, detail="Text index not found. Run /ingest first.")

    tmp_path = _save_upload_to_temp(file)
    try:
        caption = generate_caption(tmp_path)

        img_emb = image_embedding(tmp_path)
        img_idx = FaissIndex(img_emb.shape[0])
        img_idx.load(os.path.join(INDEX_DIR, "image_index"))
        sim_images = img_idx.search(img_emb, top_k=2)

        cap_emb = text_embedding(caption)
        text_idx = FaissIndex(cap_emb.shape[0])
        text_idx.load(os.path.join(INDEX_DIR, "text_index"))
        docs = text_idx.search(cap_emb, top_k=3)

        meta = pd.read_csv(META_PATH)

        retrieved_texts = []
        for did, score in docs:
            row = meta[meta["id"] == did]
            if not row.empty:
                desc = row.iloc[0].get("description")
                cap = row.iloc[0].get("caption")
                if pd.notna(desc):
                    retrieved_texts.append(desc)
                elif pd.notna(cap):
                    retrieved_texts.append(cap)

        if question.strip() == "":
            question = (
                "Describe the scene and mention any notable objects or technology if visible. "
                "If nothing stands out, say so explicitly."
            )

        answer = generate_answer(retrieved_texts, question)

        similar = []
        for img_id, score in sim_images:
            row = meta[meta["id"] == img_id]
            filename = row.iloc[0]["filename"] if not row.empty else None
            similar.append({
                "id": img_id,
                "score": score,
                "filename": filename,
                "path": os.path.join(PROJECT_ROOT, "data", "images", filename) if filename else None,
            })

        return {
            "caption": caption,
            "similar_images": similar,
            "retrieved_texts": retrieved_texts,
            "question": question,
            "answer": answer,
        }
    finally:
        os.remove(tmp_path)
