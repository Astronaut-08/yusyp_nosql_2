import os
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10   # беремо ширше, щоб RRF міг переранжувати

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

# 1. Побудувати локальний BM25-індекс за заголовками і анотаціями всіх статей.
token_corpus = (df['title'] + ' ' + df['abstract']).apply(lambda x: str(x).lower().split())
bm25 = BM25Okapi(token_corpus.tolist())

# Реалізуємо Reciprocal Rank Fusion (RRF) для об’єднання ранжованих списків BM25 і векторного пошуку:
def rrf(bm25_results, vector_results, k=TOP_K) -> list:
    rrf_scores = {}
    
    # Додаємо BM25 результати
    for rank, doc_id in enumerate(bm25_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + 60)
    
    # Додаємо векторні результати
    for rank, doc_id in enumerate(vector_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rank + 60)
    
    # Сортуємо документи і повертаємо топ-K
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:k]


# Реалізуємо функції пошуку:
# BM25;
# векторний (Pinecone);
# гібридний (BM25 + векторний через RRF).
def search_bm25(query, k=TOP_K):
    token_query = query.lower().split()
    return bm25.get_top_n(token_query, df['id'].tolist(), n=k)

def search_vector(query, k=TOP_K):
    query_embed = model.encode([query])[0].tolist()
    matches = index.query(vector=query_embed, top_k=k, include_metadata=False)['matches']
    
    results = []
    for match in matches:
        # вирішуємо проблему з ID приводячи його до нормального формату
        idx_str = str(match['id'] if isinstance(match, dict) else match.id)
        idx_str = idx_str.replace('paper_', '').split('_chunk')[0]

        try:
            row_idx = int(idx_str)
            real_id = str(df.loc[row_idx, 'id'])

            if real_id not in results:
                results.append(real_id)

        except (ValueError, IndexError):
            continue

        if len(results) >= k: break
    
    return results

def search_hybrid(query, k=TOP_K):
    bm25_results = search_bm25(query, k=TOP_K*2)
    vector_results =  search_vector(query, k=TOP_K*2)
    return rrf(bm25_results, vector_results, k=k)


# Для демонстрації виконуємо три запити:
# точний термін ("BERT fine-tuning");
# ім’я автора ("Yann LeCun convolutional networks");
# перефразування без явних термінів ("making computers understand human emotions from text").
queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text"
]

# Виводимо результати для кожного методу і порівнюємо:
# топ-5 BM25;
# топ-5 векторного пошуку;
# топ-5 гібридного пошуку з RRF, включаючи RRF-скор.
def help_out(doc_id):
    clean_id = str(doc_id).replace('paper_', '')
    match = df[df['id'] == clean_id]

    if not match.empty:
        return match['title'].values[0], str(match['abstract'].values[0])[:200] + '...'
    
    return 'N/A', 'N/A'

for query in queries:
    print(f'\n{'='*50} Запит: "{query}" {'='*50}')

    bm25_results = search_bm25(query)
    print('\nBM25 результати TOP-5:')
    for i, doc_id in enumerate(bm25_results, start=1):
        if i > 5: break
        title, abstract = help_out(doc_id)
        print(f'  {i}. {title}')
        print(f'     {abstract}\n')

    vector_results = search_vector(query)
    print('\nВекторний результати TOP-5:')
    for i, doc_id in enumerate(vector_results, start=1):
        if i > 5: break
        title, abstract = help_out(doc_id)
        print(f'  {i}. {title}')
        print(f'     {abstract}\n')

    hybrid_results = search_hybrid(query)
    print('\nГібридний результати TOP-5:')
    for i, (doc_id, rrf_score) in enumerate(hybrid_results, start=1):
        if i > 5: break
        title, abstract = help_out(doc_id)
        print(f'  {i}. {title} (RRF: {rrf_score:.4f})')
        print(f'     {abstract}\n')
