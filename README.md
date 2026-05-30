
# Project Title

A brief description of what this project does and who it's for

# Частина 1 — Підготовка даних і вибір інструментів

## 1.2. Вибір інструментів

### 1. Чим Pinecone відрізняється від Qdrant і Chroma за моделлю розгортання, ліцензією і продуктивністю? У якому сценарії ви б обрали кожен із них?
**Pinecone** розгортається тіки в хмарі, його не можливо запустити на своєму сервері. Це ліцезований комерційний продукт, відповідно вихідний код закритий. Має хорошу продуктивність.

**Qdrant** може бути розгонутим на власному сервері, або на хмарному сервісі, тобто гібридний формат. Перевага в тому що вихідний код відкритий, а також всі дачі, які можуть бути чутливими зберігаються на власному сервері за необхідності. Також має хорошу продуктивність.

**Chroma** розгортається максимально просто, встановлюється бібліотека і імпортується. Розміщується на локальному сервері, має відкритий вихідний код. По продуктивності не передбачена для роботи з великими обсягами даних.

### 2. Чому для задачі пошуку по науковим текстам обрана модель specter2_base, а не універсальна all-MiniLM-L6-v2? Знайдіть картку моделі на HuggingFace і процитуйте, для яких задач вона навчена.

`specter2_base` модель навчена на понад 6 мільйонах триплетах цитувань наукових статей, тому це набагато кращий варіант моделі для семантичоного пошуку по наукових статтях на відміну від універсальної `all-MiniLM-L6-v2`. Згідно з HuggingFace `specter2_base` передбачена для таких задач:
- класифікація (сортування за категоріями або галузю знань)
- Регресія (прогнозування майбутніх показників, наприклад кількості цитувань)
- Пошук схожого (документі за змістом які є найближчими один до одного)
- Adhoc-пошук (пошук за запитом)

### 3. Що написано у картці моделі про рекомендовану метрику схожості? Чому це важливо при створенні індексу?

Модель використуває різні метрики вимірювання, та в прикладах застосовують найчастіше косинусну схожість. При створенні індексу, косинусна схожість буде враховувати близкість векторів, а не їх довжину, тим самим нівелюючи розмір статті. Щодо конкретно рекомендованих метрик схожості які були б написані в катці моделі інформації я не знайшов.

## 1.3 Отримання ембеддингів

### Поясніть, чому при використанні нормалізованих ембеддингів (одиничної довжини) косинусна схожість (cosine similarity) еквівалентна скалярному добутку (dot product)?

Їхній еквівалент випливає з математичної сторони, фактично формула косинусної схожості це (dot product / довжини векторів). Таким чином при нормалізації цих довжин ми приводимо їх до одиничної форми. Відповідно якщо довжина одинична то це (dot product / 1), а відповідно приходимо до висновку, що якщо нормалізувати ембединги то косинусна схожіть = dot product, тобто скалярному добутку.

# Частина 2 — Завантаження даних і метадані

![alt text](image.png)

# Частина 3 — Пошукові запити

## Пояснення до пункту 4

Запит А не виві нічого тому що статей за останні 5 років нових не було, натомість більшість статей 2007 року.

### Чи збігаються топ-5 для cosine і dot product і чому?
Результати повністю збігаються, це тому що ми нормалізували довжини щоб при пошуку модель не опиралась на обсяг статті, а на її семантичний зміст.

### Чи відрізняються результати для L2 і чому?
Результати не відрізняються. Насправді цього разу ми просто не обертали список, а брали його дані такими як вони є, тому що при Евклідовій відстані ми отримуємо результати в порядку спадання. Фактично косинусна подібність та евклідова відстань також пов'язані формулою, за однієї відмінністю, чим більша косинусна схожість, тим менша Евклідова відстань, і навпаки.

### Що сталося б, якби ембеддинги не були нормалізовані?
- Косинусна схожість підбирала б статті лише за напрямком, тим самим даючи результати найкращі за семантичним змістом

- скалярний добуток надавав би перевагу більше статтям за обсягом, тобто якщо в одній статті повторюється частіше якесь слово ніж в іншій, вектор би роздувавася, тим самим ми отримували б більше великих за обсягом статей 

- Евклідова відстань чутлива до відстані векторів, тому якщо вектори близькі за семантичним змістом, але сильно відрізняються за обсягом, то будуть вважатись далекими один від одного

# Частина 4 — Chunking

```html
(.venv) PS D:\neoversity\NoSQL\yusyp_nosql_2> & d:/neoversity/NoSQL/yusyp_nosql_1/.venv/Scripts/python.exe d:/neoversity/NoSQL/yusyp_nosql_2/scripts/05_chunking.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 37934.21it/s]
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:08<00:00,  8.89s/it]
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:12<00:00,  6.44s/it]
Завантаження в arxiv-chunks-fixed
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.82s/it]
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:08<00:00,  8.41s/it]
Завантаження в arxiv-chunks-semantic
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.66s/it]
Результати пошуку для: "How do transformer architectures improve zero-shot learning capabilities in large language models?" (Index: arxiv-chunks-fixed)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: $M_\star = 0.980\pm0.062 M_\odot$ and $R_\star = 1.000_{-0.033}^{+0.036} R_\odot$, and an evolutionary age of $5.1^{+2.7}_{-2.3}$ Gyr, in good agreement with other constraints based on the strength of...
   Score: 0.5609

2. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and chemical abundance of the parent star of the recently discovered transiting planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5463

Результати пошуку для: "How do transformer architectures improve zero-shot learning capabilities in large language models?" (Index: arxiv-chunks-semantic)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and
chemical abundance of the parent star of the recently discovered transiting
planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5119

Результати пошуку для: "Techniques for mitigating catastrophic forgetting in continual learning networks." (Index: arxiv-chunks-fixed)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: $M_\star = 0.980\pm0.062 M_\odot$ and $R_\star = 1.000_{-0.033}^{+0.036} R_\odot$, and an evolutionary age of $5.1^{+2.7}_{-2.3}$ Gyr, in good agreement with other constraints based on the strength of...
   Score: 0.5353

2. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and chemical abundance of the parent star of the recently discovered transiting planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5217

Результати пошуку для: "Techniques for mitigating catastrophic forgetting in continual learning networks." (Index: arxiv-chunks-semantic)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and
chemical abundance of the parent star of the recently discovered transiting
planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.4861

Результати пошуку для: "What are the most robust optimization algorithms for training deep generative models?" (Index: arxiv-chunks-fixed)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: $M_\star = 0.980\pm0.062 M_\odot$ and $R_\star = 1.000_{-0.033}^{+0.036} R_\odot$, and an evolutionary age of $5.1^{+2.7}_{-2.3}$ Gyr, in good agreement with other constraints based on the strength of...
   Score: 0.5970

2. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and chemical abundance of the parent star of the recently discovered transiting planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5894

Результати пошуку для: "What are the most robust optimization algorithms for training deep generative models?" (Index: arxiv-chunks-semantic)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and
chemical abundance of the parent star of the recently discovered transiting
planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5540

Результати пошуку для: "Addressing bias and fairness issues in natural language processing datasets." (Index: arxiv-chunks-fixed)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and chemical abundance of the parent star of the recently discovered transiting planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5381

2. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: $M_\star = 0.980\pm0.062 M_\odot$ and $R_\star = 1.000_{-0.033}^{+0.036} R_\odot$, and an evolutionary age of $5.1^{+2.7}_{-2.3}$ Gyr, in good agreement with other constraints based on the strength of...
   Score: 0.5368

Результати пошуку для: "Addressing bias and fairness issues in natural language processing datasets." (Index: arxiv-chunks-semantic)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and
chemical abundance of the parent star of the recently discovered transiting
planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.4989

Результати пошуку для: "Applications of graph theory and spectral clustering in complex network analysis." (Index: arxiv-chunks-fixed)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and chemical abundance of the parent star of the recently discovered transiting planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5817

2. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: $M_\star = 0.980\pm0.062 M_\odot$ and $R_\star = 1.000_{-0.033}^{+0.036} R_\odot$, and an evolutionary age of $5.1^{+2.7}_{-2.3}$ Gyr, in good agreement with other constraints based on the strength of...
   Score: 0.5600

Результати пошуку для: "Applications of graph theory and spectral clustering in complex network analysis." (Index: arxiv-chunks-semantic)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and
chemical abundance of the parent star of the recently discovered transiting
planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5416

Результати пошуку для: "How is advanced linear algebra utilized in non-linear dimensionality reduction techniques like t-SNE or UMAP?" (Index: arxiv-chunks-fixed)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: $M_\star = 0.980\pm0.062 M_\odot$ and $R_\star = 1.000_{-0.033}^{+0.036} R_\odot$, and an evolutionary age of $5.1^{+2.7}_{-2.3}$ Gyr, in good agreement with other constraints based on the strength of...
   Score: 0.5814

2. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and chemical abundance of the parent star of the recently discovered transiting planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5676

Результати пошуку для: "How is advanced linear algebra utilized in non-linear dimensionality reduction techniques like t-SNE or UMAP?" (Index: arxiv-chunks-semantic)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and
chemical abundance of the parent star of the recently discovered transiting
planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5284

Результати пошуку для: "Probabilistic frameworks for uncertainty estimation in Bayesian neural networks." (Index: arxiv-chunks-fixed)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: $M_\star = 0.980\pm0.062 M_\odot$ and $R_\star = 1.000_{-0.033}^{+0.036} R_\odot$, and an evolutionary age of $5.1^{+2.7}_{-2.3}$ Gyr, in good agreement with other constraints based on the strength of...
   Score: 0.5739

2. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and chemical abundance of the parent star of the recently discovered transiting planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5623

Результати пошуку для: "Probabilistic frameworks for uncertainty estimation in Bayesian neural networks." (Index: arxiv-chunks-semantic)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and
chemical abundance of the parent star of the recently discovered transiting
planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5299

Результати пошуку для: "Convergence rates of stochastic gradient descent in non-convex optimization problems." (Index: arxiv-chunks-fixed)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: $M_\star = 0.980\pm0.062 M_\odot$ and $R_\star = 1.000_{-0.033}^{+0.036} R_\odot$, and an evolutionary age of $5.1^{+2.7}_{-2.3}$ Gyr, in good agreement with other constraints based on the strength of...
   Score: 0.5784

2. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and chemical abundance of the parent star of the recently discovered transiting planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5671

Результати пошуку для: "Convergence rates of stochastic gradient descent in non-convex optimization problems." (Index: arxiv-chunks-semantic)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and
chemical abundance of the parent star of the recently discovered transiting
planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5296

Результати пошуку для: "Energy-efficient consensus mechanisms in distributed ledger systems." (Index: arxiv-chunks-fixed)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: $M_\star = 0.980\pm0.062 M_\odot$ and $R_\star = 1.000_{-0.033}^{+0.036} R_\odot$, and an evolutionary age of $5.1^{+2.7}_{-2.3}$ Gyr, in good agreement with other constraints based on the strength of...
   Score: 0.5601

2. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and chemical abundance of the parent star of the recently discovered transiting planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5394

Результати пошуку для: "Energy-efficient consensus mechanisms in distributed ledger systems." (Index: arxiv-chunks-semantic)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and
chemical abundance of the parent star of the recently discovered transiting
planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5023

Результати пошуку для: "Recent advancements in error correction codes for quantum computing architectures." (Index: arxiv-chunks-fixed)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: $M_\star = 0.980\pm0.062 M_\odot$ and $R_\star = 1.000_{-0.033}^{+0.036} R_\odot$, and an evolutionary age of $5.1^{+2.7}_{-2.3}$ Gyr, in good agreement with other constraints based on the strength of...
   Score: 0.5874

2. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and chemical abundance of the parent star of the recently discovered transiting planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5812

Результати пошуку для: "Recent advancements in error correction codes for quantum computing architectures." (Index: arxiv-chunks-semantic)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and
chemical abundance of the parent star of the recently discovered transiting
planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5431

Результати пошуку для: "Optimizing memory allocation and parallelism in high-performance GPU computing." (Index: arxiv-chunks-fixed)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: $M_\star = 0.980\pm0.062 M_\odot$ and $R_\star = 1.000_{-0.033}^{+0.036} R_\odot$, and an evolutionary age of $5.1^{+2.7}_{-2.3}$ Gyr, in good agreement with other constraints based on the strength of...
   Score: 0.5669

2. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and chemical abundance of the parent star of the recently discovered transiting planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5561

Результати пошуку для: "Optimizing memory allocation and parallelism in high-performance GPU computing." (Index: arxiv-chunks-semantic)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and
chemical abundance of the parent star of the recently discovered transiting
planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.5200

Результати пошуку для: "What methods are used to determine the atmospheric parameters and chemical abundances of parent stars in transiting planet systems?" (Index: arxiv-chunks-fixed)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and chemical abundance of the parent star of the recently discovered transiting planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.7255

2. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: $M_\star = 0.980\pm0.062 M_\odot$ and $R_\star = 1.000_{-0.033}^{+0.036} R_\odot$, and an evolutionary age of $5.1^{+2.7}_{-2.3}$ Gyr, in good agreement with other constraints based on the strength of...
   Score: 0.6665

Результати пошуку для: "What methods are used to determine the atmospheric parameters and chemical abundances of parent stars in transiting planet systems?" (Index: arxiv-chunks-semantic)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and
chemical abundance of the parent star of the recently discovered transiting
planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.6780

Результати пошуку для: "How do the mass, radius, and evolutionary age of a parent star constrain the parameters of its transiting exoplanet?" (Index: arxiv-chunks-fixed)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and chemical abundance of the parent star of the recently discovered transiting planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.7487

2. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: $M_\star = 0.980\pm0.062 M_\odot$ and $R_\star = 1.000_{-0.033}^{+0.036} R_\odot$, and an evolutionary age of $5.1^{+2.7}_{-2.3}$ Gyr, in good agreement with other constraints based on the strength of...
   Score: 0.6928

Результати пошуку для: "How do the mass, radius, and evolutionary age of a parent star constrain the parameters of its transiting exoplanet?" (Index: arxiv-chunks-semantic)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and
chemical abundance of the parent star of the recently discovered transiting
planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.7107

Результати пошуку для: "Applications of detailed Local Thermodynamic Equilibrium (LTE) analysis in stellar spectroscopy." (Index: arxiv-chunks-fixed)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and chemical abundance of the parent star of the recently discovered transiting planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.6481

2. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: $M_\star = 0.980\pm0.062 M_\odot$ and $R_\star = 1.000_{-0.033}^{+0.036} R_\odot$, and an evolutionary age of $5.1^{+2.7}_{-2.3}$ Gyr, in good agreement with other constraints based on the strength of...
   Score: 0.6135

Результати пошуку для: "Applications of detailed Local Thermodynamic Equilibrium (LTE) analysis in stellar spectroscopy." (Index: arxiv-chunks-semantic)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and
chemical abundance of the parent star of the recently discovered transiting
planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.6000

Результати пошуку для: "How do astronomers improve the accuracy of stellar and planetary parameters for recently discovered transiting systems like TrES-2?" (Index: arxiv-chunks-fixed)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and chemical abundance of the parent star of the recently discovered transiting planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.6999

2. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: $M_\star = 0.980\pm0.062 M_\odot$ and $R_\star = 1.000_{-0.033}^{+0.036} R_\odot$, and an evolutionary age of $5.1^{+2.7}_{-2.3}$ Gyr, in good agreement with other constraints based on the strength of...
   Score: 0.6699

Результати пошуку для: "How do astronomers improve the accuracy of stellar and planetary parameters for recently discovered transiting systems like TrES-2?" (Index: arxiv-chunks-semantic)
1. Заголовок: Improving Stellar and Planetary Parameters of Transiting Planet Systems:
  The Case of TrES-2
   Текст: We report on a spectroscopic determination of the atmospheric parameters and
chemical abundance of the parent star of the recently discovered transiting
planet {TrES-2}. A detailed LTE analysis of a s...
   Score: 0.6545
```

### Яка стратегія дає більш осмислені чанки?
Більш осмислені чанки дає однозначно семантичний підхід, та варто зауважити що коли ми відібрали для цього завдання найдовші статті, вони всі переважно стосуються астрофізики, а не наших запитів які стосуються комп'ютерних наук. Я залишив запити без змін, тому що так навіть цікавіше для оцінки отриманих результатів.

### Чи є випадки розрізаних речень і як це впливає на ембеддинги?
Випадки розрізаних речень є, це робота фіксованого підходу. Через те що чанки поділені суто по кількості слів, без врахування контексту, такий підхід видає нам формули, чи обрізані шматки тексту які подекуди не несуть жодної суті

### Як розмір overlap впливає на кількість чанків і покриття тексту?
Чим більший overlap тим менший крок. Чим менший крок тим більше чанків. Чим більше чанків тим більше кількість векторів у бд. Відповідно, зміст зберігається краще, але витрати на швидкодію та інфрастуктуру більші.

# Частина 5 — Гібридний пошук
```html
(.venv) PS D:\neoversity\NoSQL\yusyp_nosql_2> & d:/neoversity/NoSQL/yusyp_nosql_1/.venv/Scripts/python.exe d:/neoversity/NoSQL/yusyp_nosql_2/scripts/06_hybrid_search.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 44382.99it/s]

================================================== Запит: "BERT fine-tuning" ==================================================

BM25 результати TOP-5:
  1. The NMSSM Solution to the Fine-Tuning Problem, Precision Electroweak
  Constraints and the Largest LEP Higgs Event Excess
     We present an extended study of how the Next to Minimal
  Supersymmetric Model easily avoids fine-tuning in electroweak symmetry
breaking for a SM-like light Higgs with mass in the vicinity of $100\ge...

  2. Fine-Tuning in Brane-antibrane Inflation
     I give a brief overview of brane-antibrane inflation, with emphasis on the
problems of tuning to get a flat potential in the KKLMMT framework, and recent
work on the nature of superpotential correctio...

  3. Conformal dynamics in gauge theories via non-perturbative
  renormalization group
     The dynamics at the IR fixed point realized in the $SU(N_c)$ gauge theories
with massless Dirac fermions is studied by means of the non-perturbative
renormalization group. The analysis includes the IR...

  4. Inverse Monte-Carlo determination of effective lattice models for SU(3)
  Yang-Mills theory at finite temperature
     This paper concludes our efforts in describing SU(3)-Yang-Mills theories at
different couplings/temperatures in terms of effective Polyakov-loop models.
The associated effective couplings are determin...

  5. Eternal Inflation is "Expensive"
     The discovery of the string theory landscape has recently brought attention
to the eternal nature of inflation. In contrast to the common belief that
eternal inflation may be a generic feature of most...


Векторний результати TOP-5:
  1. Misere quotients for impartial games: Supplementary material
     We provide supplementary appendices to the paper Misere quotients for
impartial games. These include detailed solutions to many of the octal games
discussed in the paper, and descriptions of the algor...

  2. Introduction to Phase Transitions in Random Optimization Problems
     Notes of the lectures delivered in Les Houches during the Summer School on
Complex Systems (July 2006)....

  3. Abstract Convexity and Cone-Vexing Abstractions
     This talk is a write-up on some origins of abstract convexity and afew vexing
limitations on the range of abstraction in convexity....

  4. The Compositions of the Differential Operations and Gateaux Directional
  Derivative
     In this paper we determine the number of the meaningful compositions of
higher order of the differential operations and Gateaux directional derivative....

  5. Experimental local realism tests without fair sampling assumption
     Following the theoretical suggestion of Ref. [1,2], we present experimental
results addressed to test restricted families of local realistic models, but
without relying on the fair sampling assumption...


Гібридний результати TOP-5:
  1. The NMSSM Solution to the Fine-Tuning Problem, Precision Electroweak
  Constraints and the Largest LEP Higgs Event Excess (RRF: 0.0167)
     We present an extended study of how the Next to Minimal
  Supersymmetric Model easily avoids fine-tuning in electroweak symmetry
breaking for a SM-like light Higgs with mass in the vicinity of $100\ge...

  2. Misere quotients for impartial games: Supplementary material (RRF: 0.0167)
     We provide supplementary appendices to the paper Misere quotients for
impartial games. These include detailed solutions to many of the octal games
discussed in the paper, and descriptions of the algor...

  3. Fine-Tuning in Brane-antibrane Inflation (RRF: 0.0164)
     I give a brief overview of brane-antibrane inflation, with emphasis on the
problems of tuning to get a flat potential in the KKLMMT framework, and recent
work on the nature of superpotential correctio...

  4. Introduction to Phase Transitions in Random Optimization Problems (RRF: 0.0164)
     Notes of the lectures delivered in Les Houches during the Summer School on
Complex Systems (July 2006)....

  5. Conformal dynamics in gauge theories via non-perturbative
  renormalization group (RRF: 0.0161)
     The dynamics at the IR fixed point realized in the $SU(N_c)$ gauge theories
with massless Dirac fermions is studied by means of the non-perturbative
renormalization group. The analysis includes the IR...


================================================== Запит: "Yann LeCun convolutional networks" ==================================================

BM25 результати TOP-5:
  1. On Punctured Pragmatic Space-Time Codes in Block Fading Channel
     This paper considers the use of punctured convolutional codes to obtain
pragmatic space-time trellis codes over block-fading channel. We show that good
performance can be achieved even when puncturati...

  2. Trellis-Coded Quantization Based on Maximum-Hamming-Distance Binary
  Codes
     Most design approaches for trellis-coded quantization take advantage of the
duality of trellis-coded quantization with trellis-coded modulation, and use
the same empirically-found convolutional codes ...

  3. Response of degree-correlated scale-free networks to stimuli
     The response of degree-correlated scale-free attractor networks to stimuli is
studied. We show that degree-correlated scale-free networks are robust to
random stimuli as well as the uncorrelated scale...

  4. Numerical evaluation of the upper critical dimension of percolation in
  scale-free networks
     We propose a numerical method to evaluate the upper critical dimension $d_c$
of random percolation clusters in Erd\H{o}s-R\'{e}nyi networks and in
scale-free networks with degree distribution ${\cal P...

  5. On Automorphism Groups of Networks
     We consider the size and structure of the automorphism groups of a variety of
empirical `real-world' networks and find that, in contrast to classical random
graph models, many real-world networks are ...


Векторний результати TOP-5:
  1. Multilayer Perceptron with Functional Inputs: an Inverse Regression
  Approach
     Functional data analysis is a growing research field as more and more
practical applications involve functional data. In this paper, we focus on the
problem of regression and classification with funct...

  2. The Netsukuku network topology
     In this document, we describe the fractal structure of the Netsukuku
topology. Moreover, we show how it is possible to use the QSPN v2 on the high
levels of the fractal....

  3. The Compositions of the Differential Operations and Gateaux Directional
  Derivative
     In this paper we determine the number of the meaningful compositions of
higher order of the differential operations and Gateaux directional derivative....

  4. Modeling the field of laser welding melt pool by RBFNN
     Efficient control of a laser welding process requires the reliable prediction
of process behavior. A statistical method of field modeling, based on
normalized RBFNN, can be successfully used to predic...

  5. Adaptive classification of temporal signals in fixed-weights recurrent
  neural networks: an existence proof
     We address the important theoretical question why a recurrent neural network
with fixed weights can adaptively classify time-varied signals in the presence
of additive noise and parametric perturbatio...


Гібридний результати TOP-5:
  1. Optimization in Gradient Networks (RRF: 0.0308)
     Gradient networks can be used to model the dominant structure of complex
networks. Previous works have focused on random gradient networks. Here we
study gradient networks that minimize jamming on sub...

  2. On Punctured Pragmatic Space-Time Codes in Block Fading Channel (RRF: 0.0167)
     This paper considers the use of punctured convolutional codes to obtain
pragmatic space-time trellis codes over block-fading channel. We show that good
performance can be achieved even when puncturati...

  3. Multilayer Perceptron with Functional Inputs: an Inverse Regression
  Approach (RRF: 0.0167)
     Functional data analysis is a growing research field as more and more
practical applications involve functional data. In this paper, we focus on the
problem of regression and classification with funct...

  4. Trellis-Coded Quantization Based on Maximum-Hamming-Distance Binary
  Codes (RRF: 0.0164)
     Most design approaches for trellis-coded quantization take advantage of the
duality of trellis-coded quantization with trellis-coded modulation, and use
the same empirically-found convolutional codes ...

  5. The Netsukuku network topology (RRF: 0.0164)
     In this document, we describe the fractal structure of the Netsukuku
topology. Moreover, we show how it is possible to use the QSPN v2 on the high
levels of the fractal....


================================================== Запит: "making computers understand human emotions from text" ==================================================

BM25 результати TOP-5:
  1. An Automated Evaluation Metric for Chinese Text Entry
     In this paper, we propose an automated evaluation metric for text entry. We
also consider possible improvements to existing text entry evaluation metrics,
such as the minimum string distance error rat...

  2. On the Development of Text Input Method - Lessons Learned
     Intelligent Input Methods (IM) are essential for making text entries in many
East Asian scripts, but their application to other languages has not been fully
explored. This paper discusses how such too...

  3. Towards Understanding the Origin of Genetic Languages
     Molecular biology is a nanotechnology that works--it has worked for billions
of years and in an amazing variety of circumstances. At its core is a system
for acquiring, processing and communicating in...

  4. Detecting anchoring in financial markets
     Anchoring is a term used in psychology to describe the common human tendency
to rely too heavily (anchor) on one piece of information when making decisions.
A trading algorithm inspired by biological ...

  5. Database Manipulation on Quantum Computers
     Manipulating a database system on a quantum computer is an essential aim to
benefit from the promising speed-up of quantum computers over classical
computers in areas that take a vast amount of storag...


Векторний результати TOP-5:
  1. Opinion Dynamics and Sociophysics
     No abstract given. Contents:
  I. Definition and Introduction
  II. Schelling Model
  III. Opinion Dynamics
  IV. Languages, Hierarchies and Football
  V. Future Directions...

  2. On the Development of Text Input Method - Lessons Learned
     Intelligent Input Methods (IM) are essential for making text entries in many
East Asian scripts, but their application to other languages has not been fully
explored. This paper discusses how such too...

  3. Extracting the hierarchical organization of complex systems
     Extracting understanding from the growing ``sea'' of biological and
socio-economic data is one of the most pressing scientific challenges facing
us. Here, we introduce and validate an unsupervised met...

  4. Novelty and Collective Attention
     The subject of collective attention is central to an information age where
millions of people are inundated with daily messages. It is thus of interest to
understand how attention to novel items propa...

  5. Narratives within immersive technologies
     The main goal of this project is to research technical advances in order to
enhance the possibility to develop narratives within immersive mediated
environments. An important part of the research is c...


Гібридний результати TOP-5:
  1. On the Development of Text Input Method - Lessons Learned (RRF: 0.0328)
     Intelligent Input Methods (IM) are essential for making text entries in many
East Asian scripts, but their application to other languages has not been fully
explored. This paper discusses how such too...

  2. Detecting anchoring in financial markets (RRF: 0.0298)
     Anchoring is a term used in psychology to describe the common human tendency
to rely too heavily (anchor) on one piece of information when making decisions.
A trading algorithm inspired by biological ...

  3. An Automated Evaluation Metric for Chinese Text Entry (RRF: 0.0167)
     In this paper, we propose an automated evaluation metric for text entry. We
also consider possible improvements to existing text entry evaluation metrics,
such as the minimum string distance error rat...

  4. Opinion Dynamics and Sociophysics (RRF: 0.0167)
     No abstract given. Contents:
  I. Definition and Introduction
  II. Schelling Model
  III. Opinion Dynamics
  IV. Languages, Hierarchies and Football
  V. Future Directions...

  5. Towards Understanding the Origin of Genetic Languages (RRF: 0.0161)
     Molecular biology is a nanotechnology that works--it has worked for billions
of years and in an amazing variety of circumstances. At its core is a system
for acquiring, processing and communicating in...
```

### Який метод дав кращий результат і чому?

Всі методи дали хороший результат, та найкращим виявся гібридний підхід. BM25 дав статті в яких знайшов збіги слів із запиту, цей алгоритм знайшов статті не за контектом, а за збігом. Векторний пошук як і очікувалось надав результати більш за семантичним змістом. Та саме гібрийдний об'єднав точні збіги слів із семантичним змістом, що об'єктивно можна назвати кращим підходом.

### Чи є документи в топ-5 гібридного пошуку, яких немає в топ-5 окремих методів, і чому?
Так, це стаття "*Optimization in Gradient Networks*" із запиту "*Yann LeCun convolutional networks*". Чому так трапилось? BM25 витіснив дану статтю ймовірно через не збіги слів, а векторний пошук через семантику, адже оптимізація градієнту це більше конкрено математична складова, аніж нейномережі. Та гібридний пошук сумує *score* обох, тим самим виштовхнувши дану статтю на 1 місце.

### Як зміна параметра k в RRF впливає на видачу (наприклад, k=60 vs k=1)?
k - це фактично зглажування для гібридного пошуку. Щоб побачити вплив варто подивитись на формулу (1/k+rank):
- Ранг 1 k=1: $\frac{1} {2} = 0.5$
- Ранг 10 k=1: $\frac{1} {11} = 0.09$
- Ранг 1 k=60: $\frac{1} {61} = 0.164$
- Ранг 10 k=60: $\frac{1} {61} = 0.143$

Звертаємо увагу на різницю між рангами при виводі. Бачимо що перебити результат 0.5 буде важко, якщо цей результат дала BM25, а векторний пошук, ту ж статтю важає сміттям. Тому для гібридного пошуку існує стандарт в k=60.

# Частина 6 — Аналіз і висновки

### 1. Семантичний пошук vs BM25. Наведіть конкретні приклади запитів із вашої роботи, де кожен метод виграв. Сформулюйте загальне правило: для яких типів запитів варто надати перевагу кожному з них?

### 2. Вплив розміру чанка. Що відбувається з якістю пошуку, якщо чанк занадто маленький (10–15 слів)? Якщо занадто великий (500+ слів)? Чи є оптимальний розмір або він залежить від задачі?

### 3. Невідповідна метрика. Що сталося б, якби ми створили індекс Pinecone з метрикою euclidean (L2), але використовували модель, яка повертає нормалізовані вектори? Обґрунтуйте відповідь математично: виведіть зв’язок між L2 і cosine для одиничних векторів.

### 4. Обмеження Pinecone Starter. З якими обмеженнями безкоштовного тіру ви зіткнулися (або могли б зіткнутися)? Як би ви вирішили задачу, якби датасет був не 10000, а 10 мільйонів статей?