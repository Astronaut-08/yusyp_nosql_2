import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Завантажуєм датасет
data = pd.read_parquet('data/arxiv_subset.parquet')

data['union_tsepa'] = data['title'] + ' [SEP]' + data['abstract']

# Генеруємо та закодовуємо ембединги
model = SentenceTransformer('allenai/specter2_base')

embed = model.encode(
    data['union_tsepa'].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

# Виводимо в консоль:
# загальну кількість оброблених текстів;
# розмірність ембеддингів (очікується 768);
# норму першого ембеддингу (повинна бути близька до 1.0).
print(f'Загальна кількість опрацьованих текстів: {len(embed)}')
print(f'Розмірність ембедингів: {embed.shape[1]}')
print(f'Норма першого ембедингу: {np.linalg.norm(embed[0])}')

# Зберігаємо наші дані з перевіркою чи існує директорія
os.makedirs('embeddings', exist_ok=True)
np.save('embeddings/embeddings.npy', embed)
