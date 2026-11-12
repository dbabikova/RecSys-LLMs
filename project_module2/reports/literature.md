# Литература: Ranker / Encoder / Hybrid на FAR-Trans

Этап 3 Session 1. Поиск — 2026-11-13 (Бабикова Д. с Claude). Курс требует цитировать только то,
что **сам открыл**: колонка «прочитано» показывает, что открывалось при поиске. Перед финальным
отчётом каждый автор отчёта должен открыть выбранные статьи сам.

## Ключевые слова поиска

- Финансы / датасет: `FAR-Trans`, `financial asset recommendation LLM`, `LLM stock / fund recommendation`,
  `LLM financial advisor`, `financial product recommendation LLM reranking`
- Ranker: `LLM zero-shot ranker recommender`, `LLM reranking candidate list`, `two-stage LLM ranking`
- Encoder: `LLM embeddings item representation sequential recommendation`, `long-tail users LLM`,
  `representation learning LLM recommendation`
- Hybrid: `LLM + collaborative filtering alignment`, `retriever + LLM ranker`
- Специфика наших данных (69% повторных покупок): `repeat consumption aware recommendation`

## Таблица (основные кандидаты для §2)

| # | Работа (авторы, год, площадка) | Трек | Метод | Данные и оценка | Ограничения и связь с нашим вопросом | Прочитано |
|---|---|---|---|---|---|---|
| [1] | Sanz-Cruzado, Droukas, McCreadie, 2024, Fin-RecSys @ IJCAI | данные, бейзлайны | 11 алгоритмов: price-based (LR, RF, LightGBM), Pop, LightGCN, MF, UB-kNN, ARM, Hybrid (LightGBM, LambdaMART) | FAR-Trans; 61 срез раз в 2 нед. с 2019-08, тест 6 мес.; Rel = покупка; повторные пары удалены из теста; nDCG@10 и ROI@10. LightGCN 0.3404, Pop 0.2710 | LLM-методов нет. Даёт протокол, бейзлайны и уровень nDCG. Popularity сильная, у CF низкий ROI | полный текст |
| [2] | Spadea, Seneviratne, 2025, ICAIF '25 (FLARKO) | LLM на FAR-Trans (генерация + fine-tuning) | Qwen3 0.6–8B, LoRA + KTO; транзакции и рынок сериализованы как KG (JSON-LD) в промпт; LLM выдаёт список ISIN | FAR-Trans; train до 2021-11-30, тест 2021-12…2022-11; Pref@3 / Prof@3 / Comb@3 (hit-rate, топ-3) | **Ближайшая работа.** Нет nDCG@10, нет ранжированного Top-10, нет runtime и cost, CI пересекаются, нет контроля «та же информация у классики» | полный текст (HTML) |
| [3] | Hou et al., 2024, ECIR (LNCS 14609) | Ranker | Zero-shot LLM ранжирует кандидатов по истории (ChatGPT, GPT-4, LLaMA-2 и др.); recency-промпт, bootstrapping | ML-1M, Amazon Games; 20 кандидатов; NDCG@K | Позиционный и popularity-bias, LLM плохо видит порядок истории; на кандидатах из нескольких генераторов уступает SASRec. Для нас: риск усилить и без того сильную популярность | полный текст (HTML) |
| [4] | Yue et al., 2023, PGAI @ CIKM (LlamaRec) | Ranker / Hybrid two-stage | Ретривер LRURec → топ-20 → Llama-2-7B (QLoRA), verbalizer: одна forward-проходка по буквам-индексам | ML-100k, Beauty, Games; leave-one-out; MRR/NDCG/Recall@5,10; < 1 с на пользователя | Ранжировщик не вернёт релевантный актив, которого нет в кандидатах (потолок = recall ретривера). Шаблон для дизайна «тот же список, другой порядок» | полный текст |
| [5] | Liu et al., 2024, NeurIPS (LLM-ESR) | Encoder | Замороженные LLM-эмбеддинги айтемов и пользователей (text-embedding-ada-002) + коллаборативный вид + retrieval self-distillation | Yelp, Fashion, Beauty; leave-one-out, 100 сэмпл. негативов; H@10, N@10 | Наибольший прирост — у пользователей с короткой историей (tail). LLM не вызывается в инференсе. Для нас: подгруппа с ≤ 2–4 активами | полный текст |
| [6] | Liu et al., 2025, AAAI (LLMEmb) | Encoder | Fine-tuning LLM на атрибутах айтема (supervised contrastive) + выравнивание с SRS | 3 датасета, несколько SRS-бэкбонов | Цель — long-tail айтемы; у нас мало текста об активах (имя, категория, сектор), эффект может быть слабее | только abstract / страница |
| [7] | Ren et al., 2024, WWW (RLMRec) | Encoder / Hybrid | LLM-профили пользователей и айтемов → эмбеддинги → выравнивание с CF-представлениями (контрастивно / генеративно) | Несколько бэкбонов (LightGCN и др.) | Model-agnostic «усилитель» LightGCN — сильнейшего бейзлайна FAR-Trans [1] | только abstract / страница |
| [8] | Ariannezhad et al., 2022, SIGIR (ReCANet) | контекст (не LLM) | Явное моделирование повторных покупок | 6 датасетов grocery; next-basket | Объясняет, почему у нас «повтори свои покупки» даёт nDCG ≈ 0.59 (вариант B): repeat-aware бейзлайн обязателен, если повторы считаются релевантными | только abstract / страница |
| [9] | Takayanagi et al., 2025, SIGIR (LLM-advisors) | бизнес-контекст | Лабораторное исследование, 64 участника, LLM-советник | Предпочтения, качество решений, доверие | Пользователи плохо отличают хороший совет от плохого → нужна офлайн-оценка против реального поведения, а не впечатление от объяснений | abstract + HTML |

Для §2 нужно 3–5 работ, рекомендуемый набор зависит от трека:
- **Ranker:** [1], [2], [3], [4] (+ [8] при варианте B)
- **Encoder:** [1], [2], [5], [7] (или [6])
- **Hybrid:** [1], [2], [4], [5]

## Nearest prior work и gap

**Ближайшая работа — FLARKO [2]**: единственная найденная LLM-работа на FAR-Trans. Ответила на вопрос:
может ли дообученная небольшая LLM с KG-контекстом выдавать покупаемые и одновременно прибыльные активы
(top-3 hit-rate). Осталось открытым для нашего условия:
- ранжирующее качество Top-10 в nDCG@10 по протоколу [1] (FLARKO выдаёт неранжированный top-3);
- вклад именно LLM при той же информации у классической модели (same-information control);
- отдельный вклад encoder- и ranker-части;
- runtime и стоимость.

Формулировки для отчёта (без глобальных утверждений о новизне):
- «Мы не нашли работу, которая сообщает nDCG@10 для LLM-реранкинга кандидатов классического рекомендера на FAR-Trans.»
- «Мы не нашли работу, которая сравнивает LLM-эмбеддинги активов и истории с ID-only бейзлайном на FAR-Trans для пользователей с короткой историей.»
- [3], [4] показывают пользу LLM-ранкера на фильмах и e-commerce с богатыми названиями. На финансовых
  активах с бедным текстом и сильной популярностью этот эффект не проверялся: в найденных нами работах
  такой проверки нет.

## Как литература стыкуется с EDA (этап 2)

| Наблюдение в данных | Что говорит литература | Следствие для дизайна |
|---|---|---|
| Топ-10 активов = 39% пар, Popularity ≈ 0.19–0.29 | LLM-ранкеры склонны к popularity-bias [3] | Мерить, не «съехала» ли выдача к популярным; Popularity — обязательный бейзлайн |
| Медиана 2 актива на клиента | LLM-эмбеддинги помогают tail-пользователям [5] | Подгруппа hist ≤ 2–4 как population для Encoder-вопроса |
| Бедный текст активов | Методы [5–7] опираются на заголовки и описания | Строить текст: название + категория + сектор + отрасль + рынок; это риск для Encoder |
| 69% покупок повторные | Repeat-aware модели сильны [8] | Решение по варианту A/B определяет, нужен ли repeat-бейзлайн |
| LightGCN — лучший nDCG в [1] | RLMRec усиливает LightGCN [7]; LlamaRec ранжирует топ-20 [4] | Естественный кандидатогенератор / бейзлайн |

## Черновые варианты RQ (для обсуждения в команде, не финал)

- **Ranker:** We test whether zero-shot LLM reranking of the top-20 candidates of a classical recommender
  (LightGCN / Popularity) improves nDCG@10 over the original order and over a LambdaMART reranker with
  the same features, for the common FAR-Trans evaluation users.
- **Encoder:** We test whether LLM text embeddings of assets and purchase histories improve nDCG@10 over an
  ID-only collaborative recommender for FAR-Trans users with at most 4 distinct prior assets.
- **Hybrid:** We test whether an LLM-encoder retriever combined with an LLM reranker improves nDCG@10 over
  encoder-only and reranker-only variants for the common FAR-Trans evaluation users.

## References (IEEE)

[1] J. Sanz-Cruzado, N. Droukas, and R. McCreadie, "FAR-Trans: An investment dataset for financial asset recommendation," in *Proc. IJCAI-2024 Workshop on Recommender Systems in Finance (Fin-RecSys)*, Jeju, South Korea, 2024. [Online]. Available: https://arxiv.org/abs/2407.08692

[2] F. Spadea and O. Seneviratne, "Aligning language models with investor and market behavior for financial recommendations," in *Proc. 6th ACM Int. Conf. AI in Finance (ICAIF '25)*, Singapore, 2025. [Online]. Available: https://arxiv.org/abs/2510.15993

[3] Y. Hou, J. Zhang, Z. Lin, H. Lu, R. Xie, J. McAuley, and W. X. Zhao, "Large language models are zero-shot rankers for recommender systems," in *Proc. ECIR 2024*, LNCS vol. 14609, pp. 364–381, 2024, doi: 10.1007/978-3-031-56060-6_24.

[4] Z. Yue, S. Rabhi, G. de Souza Pereira Moreira, D. Wang, and E. Oldridge, "LlamaRec: Two-stage recommendation using large language models for ranking," in *PGAI Workshop @ CIKM 2023*, 2023. [Online]. Available: https://arxiv.org/abs/2311.02089

[5] Q. Liu, X. Wu, Y. Wang, Z. Zhang, F. Tian, Y. Zheng, and X. Zhao, "LLM-ESR: Large language models enhancement for long-tailed sequential recommendation," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 37, 2024. [Online]. Available: https://arxiv.org/abs/2405.20646

[6] Q. Liu, X. Wu, W. Wang, Y. Wang, Y. Zhu, X. Zhao, F. Tian, and Y. Zheng, "LLMEmb: Large language model can be a good embedding generator for sequential recommendation," in *Proc. AAAI*, vol. 39, no. 11, pp. 12183–12191, 2025, doi: 10.1609/aaai.v39i11.33327.

[7] X. Ren, W. Wei, L. Xia, L. Su, S. Cheng, J. Wang, D. Yin, and C. Huang, "Representation learning with large language models for recommendation," in *Proc. ACM Web Conf. (WWW '24)*, pp. 3464–3475, 2024, doi: 10.1145/3589334.3645458.

[8] M. Ariannezhad, S. Jullien, M. Li, M. Fang, S. Schelter, and M. de Rijke, "ReCANet: A repeat consumption-aware neural network for next basket recommendation in grocery shopping," in *Proc. 45th Int. ACM SIGIR Conf.*, pp. 1240–1250, 2022.

[9] T. Takayanagi, K. Izumi, J. Sanz-Cruzado, R. McCreadie, and I. Ounis, "Are generative AI agents effective personalized financial advisors?," in *Proc. SIGIR 2025*, 2025. [Online]. Available: https://arxiv.org/abs/2504.05862
