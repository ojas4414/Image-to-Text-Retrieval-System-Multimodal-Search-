import faiss
import numpy as np
import pickle

class FaissIndex:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product; use normalized vectors
        self.id_map = []

    def add(self, vectors, ids):
        # vectors: (n, dim) numpy float32
        self.index.add(vectors)
        self.id_map.extend(ids)

    def search(self, qvec, top_k=5):
        qvec = qvec.reshape(1, -1).astype('float32')
        scores, idxs = self.index.search(qvec, top_k)
        results = []
        for s, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            results.append((self.id_map[i], float(s)))
        return results

    def save(self, base_path):
        faiss.write_index(self.index, base_path + ".index")
        with open(base_path + ".meta.pkl", "wb") as f:
            pickle.dump(self.id_map, f)

    def load(self, base_path):
        self.index = faiss.read_index(base_path + ".index")
        with open(base_path + ".meta.pkl", "rb") as f:
            self.id_map = pickle.load(f)
