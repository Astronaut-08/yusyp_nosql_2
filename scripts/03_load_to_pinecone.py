import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Створюємо індекс (якщо не існує)
exists_indexes = [idx['name'] for idx in pc.list_indexes()] # витягуємо створені індекси

if INDEX_NAME not in exists_indexes:
    pc.create_index(name=INDEX_NAME, dimension=VECTOR_DIM, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))

index = pc.Index(INDEX_NAME)

# Завантажуємо дані
df = pd.read_parquet(INPUT_PARQUET)
embed = np.load(INPUT_EMBEDDINGS)

# Готуємо дані для завантаження:
# функція щоб обробляти записи батчами (наприклад, по 200 елементів);
# для кожного запису об’єкт із:
# унікальним id вигляду "paper_<номер>";
# ембеддингами;
# метаданими: arxiv_id, title, abstract (до 500 символів), authors (до 200 символів), year, category.
def prepare_batch(start_idx, size):
    data = []
    for i in range(start_idx, min(start_idx + size, len(df))):
        paper = f'paper_{i}'
        embedding = embed[i].tolist()
        metadata = {
            'arxiv_id': str(df.loc[i, 'id']),
            'title': str(df.loc[i, 'title']),
            'abstract': str(df.loc[i, 'abstract'])[:500],
            'authors': str(df.loc[i, 'authors'])[:200],
            'year': int(df.loc[i, 'year']),
            'category': str(df.loc[i, 'category'])
        }
        
        data.append((paper, embedding, metadata))
    
    return data

# print(df.columns.tolist()) ТУТ ПЕРЕВІРКА НА НАЯВНІСТЬ КОЛОНОК В РАЗІ ПОМИЛКИ

# Завантажуємо дані в Pinecone батчами та виводимо прогрес за допомогою tqdm.
for i in tqdm(range(0, len(df), BATCH_SIZE)):
    batch = prepare_batch(i, BATCH_SIZE)
    index.upsert(vectors=batch)

info = index.describe_index_stats()
print(f'Загальна кількість векторів: {info["total_vector_count"]}')