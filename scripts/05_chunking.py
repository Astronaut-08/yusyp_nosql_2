import os
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

# Вибираємо 30 статей із найдовшими анотаціями.
df['abstract_length'] = df['abstract'].apply(lambda x: len(re.findall(r'\w+', x)))
df_long_k30 = df.nlargest(30, 'abstract_length').reset_index(drop=True)

# Розбиваємо тексти на чанки двома стратегіями:
# Fixed-size chunking: фіксована кількість слів з невеликим перекриттям між чанками;
def fixed_size_chunk(text, size=200, overlap=20) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = ' '.join(words[i:i + size])
        if chunk: chunks.append(chunk)
    return chunks

# Semantic chunking: об’єднання речень до досягнення максимальної кількості слів, щоб зберегти зміст.
def semantic_chunk(text, size=200) -> list:
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if len(current_chunk) + len(sentence.split()) <= size:
            current_chunk.append(sentence)
        else:
            if current_chunk: chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]

    if current_chunk: chunks.append(' '.join(current_chunk))

    return chunks

# 3. Створюємо окремі індекси в Pinecone для кожного типу чанків (arxiv-chunks-fixed і arxiv-chunks-semantic).
spec = ServerlessSpec(cloud='aws', region='us-east-1')
exist_idxs = [idx.name for idx in pc.list_indexes()]

for idx_name in ['arxiv-chunks-fixed', 'arxiv-chunks-semantic']:
    if idx_name not in exist_idxs:
        pc.create_index(
            name=idx_name,
            dimension=VECTOR_DIM,
            metric='cosine',
            spec=spec
        )
    # Pinecone час на ініціалізацію
    while not pc.describe_index(idx_name).status['ready']:
        time.sleep(1)

# створюємо ембеддинг за допомогою моделі allenai/specter2_base;
embed = model.encode(df_long_k30['abstract'].tolist(), show_progress_bar=True)

# сформовуємо об’єкт з унікальним id, ембеддингом і метаданими: arxiv_id, title, текст чанка, номер чанка, рік, категорія.
def create_obj(df, chunking_strategy) -> list:
    data_to_upload = []
    
    # Закидаємо все батчами
    all_chunks_text = []
    metadata_list = []

    for _, row in df.iterrows():
        chunks = chunking_strategy(row['abstract'])

        for i, text in enumerate(chunks):
            all_chunks_text.append(text)
            metadata_list.append({
                'id': f'{i**2%1000}_chunk_{i}',
                'arxiv_id': str(row.get('arxiv_id', 'unknown')),
                'title': row.get('title', 'Unknown Title'),
                'chunk_text': text,
                'chunk_number': i,
                'year': int(row.get('year', 0)) if pd.notnull(row.get('year')) else 0,
                'category': str(row.get('categories', 'unknown'))
            })
    # Створюємо ембеддинги для всіх чанків одразу та формуємо об’єкти для завантаження
    embed = model.encode(all_chunks_text, show_progress_bar=True)

    for meta, emb in zip(metadata_list, embed):
        vec_id = meta.pop('id')
        data_to_upload.append({
            'id': vec_id,
            'values': emb.tolist(),
            'metadata': meta
        })

    return data_to_upload

# Завантажуємо чанки в Pinecone батчами і відображаємо прогрес.
def upload_chunks(vectors, index_name):
    index = pc.Index(index_name)
    print(f'Завантаження в {index_name}')
    for i in tqdm(range(0, len(vectors), 100)):
        batch = vectors[i:i+100]
        index.upsert(vectors=batch)

fixed_vectors = create_obj(df_long_k30, fixed_size_chunk)
upload_chunks(fixed_vectors, 'arxiv-chunks-fixed')

semantic_vectors = create_obj(df_long_k30, semantic_chunk)
upload_chunks(semantic_vectors, 'arxiv-chunks-semantic')

# Реалізуємо функцію пошуку по чанках:
def search_chunks(query, index_name, top_k=5) -> list:
    query_embed = model.encode([query])[0]
    index = pc.Index(index_name)

    results = index.query(
        vector=query_embed.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    print(f'Результати пошуку для: "{query}" (Index: {index_name})')
    for idx, match in enumerate(results['matches']):
        meta = match['metadata']
        title = meta.get('title', 'N/A')
        text = meta.get('chunk_text', 'N/A')
        score = match.get('score', 0)

        print(f'{idx+1}. Заголовок: {title}\n   Текст: {text[:200]}...\n   Score: {score:.4f}\n')


# Виконуємо пошук за кількома тестовими запитами;
test_queries = [
    "How do transformer architectures improve zero-shot learning capabilities in large language models?",
    "Techniques for mitigating catastrophic forgetting in continual learning networks.",
    "What are the most robust optimization algorithms for training deep generative models?",
    "Addressing bias and fairness issues in natural language processing datasets.",
    "Applications of graph theory and spectral clustering in complex network analysis.",
    "How is advanced linear algebra utilized in non-linear dimensionality reduction techniques like t-SNE or UMAP?",
    "Probabilistic frameworks for uncertainty estimation in Bayesian neural networks.",
    "Convergence rates of stochastic gradient descent in non-convex optimization problems.",
    "Energy-efficient consensus mechanisms in distributed ledger systems.",
    "Recent advancements in error correction codes for quantum computing architectures.",
    "Optimizing memory allocation and parallelism in high-performance GPU computing.",
    "What methods are used to determine the atmospheric parameters and chemical abundances of parent stars in transiting planet systems?",
    "How do the mass, radius, and evolutionary age of a parent star constrain the parameters of its transiting exoplanet?",
    "Applications of detailed Local Thermodynamic Equilibrium (LTE) analysis in stellar spectroscopy.",
    "How do astronomers improve the accuracy of stellar and planetary parameters for recently discovered transiting systems like TrES-2?"
]

# Виводимо топ-5 результатів для кожного типу чанків з назвою статті і частиною тексту чанка.
for query in test_queries:
    search_chunks(query, 'arxiv-chunks-fixed')
    search_chunks(query, 'arxiv-chunks-semantic')
