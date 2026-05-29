import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

# Підключення до індексу та завантаження моделі
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")  # для отримання повного abstract

def encode_query(query):
    '''Кодує запит у вектор за допомогою моделі'''
    return model.encode(query).tolist()

# Задаємо запит і отримуємо топ статей
query = "teaching machines to recognize objects in pictures"
query_vec = encode_query(query)

result = index.query(vector=query_vec, top_k=TOP_K, include_metadata=True)

# Виводимо результати
for i, match in enumerate(result['matches']):
    print(f'TOP: {i+1}')
    print(f'title: {match["metadata"]["title"]}')
    print(f'abstract: {match["metadata"]["abstract"][:100]}')
    print(f'year: {match["metadata"]["year"]}')
    print(f'category: {match["metadata"]["category"]}')
    print('-'*20)

print('\n' + '='*40 + '\n')
#  пошук з фільтрацією:
# - приклад A: статті по reinforcement learning за останні 5 років і категорія cs.LG;
query_a = 'reinforcement learning'
query_vec_a = encode_query(query_a)
filter_a = {
    'year': {'$gte': 2020},
    'category': {'$eq': 'cs.LG'}
}

result_a = index.query(vector=query_vec_a, include_metadata=True, filter=filter_a, top_k=TOP_K)

# - приклад B: більш старі статті (до 2015 року), будь-яка категорія;
query_b = 'reinforcement learning'
query_vec_b = encode_query(query_b)
filter_b = {
    'year': {'$lte': 2015},
}

result_b = index.query(vector=query_vec_b, include_metadata=True, filter=filter_b, top_k=TOP_K)

# - порівняти видачу і пояснити відмінності.
print('Результати для статей по reinforcement learning за останні 5 років і категорія cs.LG:')
if result_a['matches']:
    for i, match in enumerate(result_a['matches']):
        print(f'TOP: {i+1}')
        print(f'title: {match["metadata"]["title"]}')
        print(f'abstract: {match["metadata"]["abstract"][:100]}')
        print(f'year: {match["metadata"]["year"]}')
        print(f'category: {match["metadata"]["category"]}')
        print('-'*20)
else:
    print("Немає результатів для цього запиту.")

print('\n' + '='*40 + '\n')

print('Результати для більш старих статей (до 2015 року), будь-яка категорія:')
if result_b['matches']:
    for i, match in enumerate(result_b['matches']):
        print(f'TOP: {i+1}')
        print(f'title: {match["metadata"]["title"]}')
        print(f'abstract: {match["metadata"]["abstract"][:100]}')
        print(f'year: {match["metadata"]["year"]}')
        print(f'category: {match["metadata"]["category"]}')
        print('-'*20)
else:
    print("Немає результатів для цього запиту.")

print('\n' + '='*40 + '\n')

# Порівнюємо всі метрики схожості
# завантажуємо всі ембеддинги з embeddings/embeddings.npy;
embed = np.load('embeddings/embeddings.npy')

# для заданого запиту обчислюємо:
# cosine similarity;
# dot product;
# L2-distance;
cosine_sim = np.dot(embed, query_vec) / (np.linalg.norm(embed, axis=1) * np.linalg.norm(query_vec))
dot_product = np.dot(embed, query_vec)
l2_distance = np.linalg.norm(embed - query_vec, axis=1)

# виводимо топ-5 статей для кожної метрики
print(f'ТОП-5 статей (cosine similarity): {df.iloc[np.argsort(cosine_sim)[-TOP_K:][::-1]]}')
print('\n' + '='*40 + '\n')
print(f'ТОП-5 статей (dot product): {df.iloc[np.argsort(dot_product)[-TOP_K:][::-1]]}')
print('\n' + '='*40 + '\n')
print(f'ТОП-5 статей (L2-distance): {df.iloc[np.argsort(l2_distance)[:TOP_K]]}')
