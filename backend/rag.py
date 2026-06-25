from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
rag_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)

def generate_answer(retrieved_texts, user_question, max_length=128):
    # 1. Filter the list to ensure we only have strings
    clean_texts = []
    if isinstance(retrieved_texts, list):
        for t in retrieved_texts:
            # Only append if it is a string and not empty
            if isinstance(t, str) and t.strip():
                clean_texts.append(t)

    # 2. Handle case where no valid text was found
    if not clean_texts:
        context = ""
    else:
        context = "\n".join(clean_texts)

    # 3. Generate the answer
    prompt = f"Context: {context}\n\nQuestion: {user_question}\nAnswer concisely:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = rag_model.generate(**inputs, max_new_tokens=max_length)
    return tokenizer.decode(out[0], skip_special_tokens=True)
