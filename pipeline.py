# ================================================
# SNAKE SEARCH PIPELINE - Tìm rắn bằng tiếng Việt
# ================================================
import os, json
import numpy as np
from PIL import Image
from inference_sdk import InferenceHTTPClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import faiss

# Tự động đọc file .env
load_dotenv()

# ---- CẤU HÌNH ----
API_KEY    = os.getenv('ROBOFLOW_API_KEY', 'your_key_here')
MODEL_ID   = 'snakedetection-krhf0/9'
IMG_FOLDER = './snake_images'
INDEX_FILE = 'snake.index'
META_FILE  = 'snake_meta.json'

# ---- KHỞI TẠO CLIENT ----
CLIENT = InferenceHTTPClient(
    api_url='https://serverless.roboflow.com',
    api_key=API_KEY
)
clip = SentenceTransformer('clip-ViT-B-32-multilingual-v1')


# ---- Hàm build_index() - Chạy 1 lần duy nhất ----
def build_index(folder):
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    embeddings, meta = [], []
    for i, fname in enumerate(files):
        path = os.path.join(folder, fname)
        try:
            result = CLIENT.infer(path, model_id=MODEL_ID)
            preds  = result.get('predictions', [])
            img    = Image.open(path).convert('RGB')
            if preds:
                b = max(preds, key=lambda x: x['confidence'])
                img = img.crop((
                    int(b['x'] - b['width']  / 2),
                    int(b['y'] - b['height'] / 2),
                    int(b['x'] + b['width']  / 2),
                    int(b['y'] + b['height'] / 2)))
            emb = clip.encode(img)
            embeddings.append(emb)
            meta.append({'path': path, 'filename': fname,
                         'has_snake': bool(preds)})
            if (i + 1) % 100 == 0:
                print(f'  Xong {i+1}/{len(files)}')
        except Exception as e:
            print(f'  Loi {fname}: {e}')

    arr = np.array(embeddings).astype('float32')
    faiss.normalize_L2(arr)
    idx = faiss.IndexFlatIP(arr.shape[1])
    idx.add(arr)
    faiss.write_index(idx, INDEX_FILE)
    json.dump(meta, open(META_FILE, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(f'Done: {len(embeddings)} anh')
    return idx, meta


# ---- Hàm search() - Tìm kiếm bằng tiếng Việt ----
def search(query, idx, meta, top_k=5):
    q = clip.encode(query).astype('float32').reshape(1, -1)
    faiss.normalize_L2(q)
    scores, ids = idx.search(q, top_k)
    return [{'score': round(float(s), 4),
             'file': meta[i]['filename']}
            for s, i in zip(scores[0], ids[0])]


# ---- CHẠY CHƯƠNG TRÌNH ----
if __name__ == '__main__':
    if not os.path.exists(INDEX_FILE):
        print('Building index...')
        idx, meta = build_index(IMG_FOLDER)
    else:
        idx  = faiss.read_index(INDEX_FILE)
        meta = json.load(open(META_FILE, encoding='utf-8'))
        print(f'Loaded: {len(meta)} images')

    # --- TEST ---
    queries = [
        'ran ho mang dau banh mau den',
        'ran luc duoi do mau xanh la',
        'ran cap nong van den vang',
    ]
    for q in queries:
        print(f'\nQuery: {q}')
        for r in search(q, idx, meta):
            print(f'  {r["score"]:.3f} | {r["file"]}')
