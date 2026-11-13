# Литература — LLM4Rec / FAR-Trans, Session 1

**Дата поиска и проверки:** 2026-11-13. **Индивидуальная работа:** Бабикова Дарья.
Обзор подготовлен AI-ассистентом OpenAI GPT-6.1 Sol; личное прочтение и
подтверждение Дарьей не предполагаются автоматически. Командного согласования
RQ этот документ не заменяет. Трек, RQ, модель и официальный протокол ещё не выбраны.

## 1. Границы обзора и правила чтения

- Это целевой поиск по направлениям, а не систематический обзор с доказательством
  полноты. Использованы веб-поиск, открытые HTML arXiv, ACL Anthology, Crossref
  и авторский репозиторий для проверки библиографии.
- В таблицах **«полный текст»** означает прочитанный текст статьи, включая метод,
  экспериментальную постановку, результаты и заключение, а не только abstract.
  Для RLMRec и Fund2Persona также проверены приложения, существенные для выводов.
  Это не воспроизведение кода, не независимая проверка чисел авторов и не заявление
  о ручной проверке каждого пикселя рисунков PDF.
- **«Только аннотация»** — прочитана открытая первичная страница статьи с abstract;
  детали, которых в ней нет, помечены как непроверенные. Такая работа не служит
  доказательством экспериментального результата или отсутствия исследования.
- Поисковые сниппеты и чужие пересказы не считаются прочитанной статьёй.
  Недоступные работы не включены в основную таблицу. Отдельно отмечен UR4Rec:
  открыта первичная аннотация, но PDF данным инструментом не разобран.
- Год публикации отделён от года первого препринта; указана конкретная прочитанная
  версия. Для accepted-препринтов статус не превращается в подтверждённые страницы
  опубликованного сборника.
- Предыдущий полный разбор статьи FAR-Trans выполнен на этапе данных в этой же
  AI-сессии; здесь повторно проверена её библиографическая страница.
- Для финального PDF Session 1 из этой расширенной карты нужно выбрать **3–5**
  наиболее связанных исследований после выбора одного RQ, а не переносить всю карту.

### Как понимаем треки

**Ranker:** LLM оценивает/переставляет доступных кандидатов.
**Encoder:** языковая модель даёт векторы объектов/истории для retrieval или CF.
**Hybrid / alignment:** совместное использование семантических и коллаборативных
представлений либо языкового encoder и ranker. Это не обязательно две отдельные LLM.
Обычный retriever → LLM-ranker сам по себе не доказывает использование
**LLM-encoder + LLM-ranker**. Ниже не приписываем такую архитектуру LlamaRec
с её ID-based LRURec или SBERT-based пайплайну [3].
Recformer — encoder-ориентированная PLM-работа на Longformer, не современная
инструкционная decoder-only LLM. Она включена как методический предшественник.

### Факты нашей постановки, с которыми сопоставляем статьи

Источники локальных чисел: [eda.md](eda.md), [leakage.md](leakage.md),
[split.md](split.md). Новых вычислений данных на этом этапе не выполнялось.

- Цель курса — Top-10, средний **nDCG@10 по будущим Buy**, не прибыль и не оценка
  убедительности ответа. Общие cut, пользователи, universe и метки ещё pending.
- **68.6507%** — event-level повторы среди Buy с неотрицательными ID;
  **60.7344%** — с восстановленными Buy. Это доля строк сверх первой покупки пары,
  а не доля повторных distinct-меток в будущем окне и не средний basket repetition ratio.
- Топ-10 концентрируют **48.2271%** всех Buy; акции дают **88.9264%** Buy.
  Медиана числа разных купленных активов — **1**, но это НЕ означает одну сделку
  на клиента: одно и то же имя может многократно повторяться в истории.
- Медиана непустого названия актива — **4 слова**, свободных описаний нет;
  sector отсутствует у **47.6427%**, industry — у **53.7221%** unique-активов.
  Отзывов и ответов клиентских анкет нет.
- В последних профилях риск оценён у **24.5480%**, ёмкость — у **25.1564%** клиентов.
  У **1 209** выбранных as-of снимков дата анкеты позже предварительного cut.
  Выявлено **38 019** backdated Buy с отрицательным ID; происхождение остальной
  синтетики, переносов дат и ценовых поправок не полностью маркировано.
- В предварительной B Repeat+Popularity дал **0.588685** nDCG@10; в A знакомые
  активы запрещены и этот ориентир совпадает с Popularity. Это локальные
  диагностические числа с оговорками об утечках, не официальные результаты
  и не сопоставимые напрямую с числами из статей.

## 2. Проверенные работы

Номера отсылают к IEEE References в конце. В последнем столбце отделены
результат авторов и наша интерпретация применимости. Названия датасетов,
метрики и протоколы не переносятся на FAR-Trans автоматически.

### 2.1. Ranker и двухэтапная рекомендация

| Работа: авторы, год, площадка | Трек / метод | Данные и оценка | Ограничения и связь с нашей задачей | Прочитано |
|---|---|---|---|---|
| **[1] Hou et al., “Large Language Models are Zero-Shot Rankers for Recommender Systems,” ECIR 2024**; препринт 2023, прочитана v2 | **Ranker, zero-shot.** GPT-3.5/GPT-4 и открытые LLM получают историю и кандидатов; recency-focused prompting, перестановки и агрегация рангов уменьшают biases | MovieLens-1M, Amazon Games; 5-core, leave-one-out; nDCG@1/5/10/20. Есть искусственный pool из 1 positive + 19 negatives, hard negatives и более реалистичная объединённая выдача нескольких retrievers без обязательного добавления positive. Большинство результатов усреднены по ≥3 запускам; отдельная дорогая модельная таблица — один запуск на подвыборке | Авторы показывают position/popularity bias и слабое восприятие порядка истории; результаты зависят от способа построения pool. Для FAR-Trans нельзя гарантированно вставлять будущий Buy в кандидаты. Нужны полный end-to-end nDCG@10, Recall кандидатов, перестановки и запрет внепуловых ID. Методически близкий источник для zero-shot Ranker, но не финансы и не шестимесячная многометочная оценка | **Полный текст**, HTML v2 |
| **[2] Yue, Rabhi, Moreira, Wang, Oldridge, “LlamaRec,” 2023**; PGAI@CIKM по авторскому репозиторию; цитируем доступный arXiv v1 | **Ranker, дообучение; retriever → LLM.** ID-based LRURec выдаёт top-20, Llama 2-7B дообучается QLoRA. Verbalizer берёт logits букв кандидатов за один forward pass вместо генерации длинного списка | MovieLens (в тексте обозначен ML-100k), Amazon Beauty/Games; 5-core, leave-one-out; MRR/nDCG/Recall@5/10, общий каталог в end-to-end оценке. Отдельно показан **valid retrieval subset**, где positive есть в top-20; есть сравнение скорости verbalizer и генерации | Самый близкий технический рецепт дообученного реранкера. Улучшение на valid subset не равно качеству по всем пользователям. Заголовки и статистики MovieLens в препринте неоднородны; не исправляем их молча. Verbalizer после обучения не доказывает качество zero-shot verbalizer и не даёт калиброванную вероятность покупки. LRURec здесь не LLM-encoder | **Полный текст**, HTML v1; отдельно открыт README авторов |
| **[3] Kapetanovic, Duricic, Mercep, Lacic, “Retrieval, Scoring, and Decoding Shape Performance and Stability in LLM-based Conversational Recommendation,” CIKM 2026**, arXiv v1 | **Ranker + encoder→ranker pipeline.** SBERT `all-mpnet-base-v2` кодирует metadata, CBF строит pool; его заменяют EASE/SASRec. Proprietary/open-weight LLM и LoRA-Qwen2.5-7B ранжируют pool | ReDial: стандартный test, 1 025 диалогов, 6 924 фильма; Top-10, бинарный nDCG@10, Hit@10, coverage/popularity. Сравниваются pools 250/500/1000/full-catalog и генерация без pool; strict scoring не засчитывает внепуловые ответы. Есть candidate recall/oracle и повторные генерации на stability-подвыборке | Особенно полезная **отрицательная** проверка: все проверенные open-weight реранкеры уступили EASE в matched semantic pool, proprietary модели лучше. Но сами авторы подчёркивают: **same candidates ≠ same information** — LLM видит сырой диалог, CF — историю. Для нас нужны одинаковые доступные поля, а не только одинаковые ISIN. SBERT — языковой encoder, не отдельно обученный генеративный LLM-encoder; один ReDial и нет систематического cost/latency benchmark | **Полный текст**, HTML v1 |
| **[7] H. Zhang, Q. Zhu, Z. Dou, “Enhancing Reranking for Recommendation with LLMs through User Preference Retrieval” (UR4Rec), COLING 2025** | **Ranker / preference retrieval.** Малый transformer-retriever извлекает из истории релевантные кандидатам предпочтения; LLM создаёт полезное знание для реранкинга вместо избыточного текста | Abstract сообщает три публичных real-world датасета. **Их названия, метрики, split и детали дообучения по полному тексту не проверены**, поэтому чисел и детального протокола здесь нет | Полезный след по candidate-conditioned context compression. Не используем как доказательство преимущества на nDCG@10 или как подтверждённый dual-LLM Hybrid. Полный PDF ещё требуется проверить перед использованием в основном аргументе PDF | **Только аннотация**, первичная страница ACL Anthology; PDF-инструмент вернул unsupported content type |

### 2.2. Encoder и Hybrid / alignment

| Работа: авторы, год, площадка | Трек / метод | Данные и оценка | Ограничения и связь с нашей задачей | Прочитано |
|---|---|---|---|---|
| **[4] J. Li et al., “Text Is All You Need: Learning Language Representations for Sequential Recommendation” (Recformer), KDD 2023**, v2 | **Encoder (PLM).** Key–value атрибуты превращаются в текст; Longformer-based модель кодирует актив/последовательность, cosine similarity используется для retrieval. MLM + contrastive pretraining, двухэтапный fine-tuning | Предобучение на категориях Amazon, оценка на шести других: Scientific, Instruments, Arts, Office, Games, Pet. 5-core, удалены отсутствующие titles; leave-one-out, ranking по всем items, nDCG@10/Recall@10/MRR; есть zero-shot/low-resource/cold-item эксперименты | Авторы показывают пользу переносимых языковых представлений. Но это не замена энкодера без обучения и не доказательство выигрыша на бедных метаданных FAR-Trans. “Two-stage fine-tuning” здесь — обучение encoder, **не retriever→LLM-ranker**. Нужен структурный same-information контроль, отдельная проверка временной доступности атрибутов | **Полный текст**, HTML v2 |
| **[5] X. Ren et al., “Representation Learning with Large Language Models for Recommendation” (RLMRec), WWW 2024**, прочитана arXiv v5 | **Encoder + collaborative alignment.** ChatGPT генерирует профили user/item из описаний/отзывов; text-embedding-ada-002 кодирует их; contrastive или masked-generative alignment согласует семантику с CF. Инференс выполняет рекомендательный backbone | Amazon-book, Yelp, Steam; k-core, split 3:1:1, all-item ranking; Recall/nDCG@5/10/20, пять инициализаций; отдельный Yelp temporal-pretraining эксперимент. Подключены LightGCN и другие CF-backbones; есть шумовые и shuffled-embedding абляции | Показано, что semantic-only embeddings слабы, а alignment может улучшать CF. Это **не** последовательность LLM-encoder→отдельный LLM-ranker. В основной оценке split не описан как наш общий chronological cut. Богатых отзывов у нас нет; нельзя дополнять промпт выдуманными предпочтениями. В статье также приведён контрпример: прямой ChatGPT-reranking уступает LightGCN; размер pool в introduction и appendix описан неодинаково | **Полный текст**, HTML v5, включая Appendix A |
| **[6] Liao et al., “LLaRA: Large Language-Recommendation Assistant,” SIGIR 2024**, v4 | **Hybrid representation / alignment с LLM-ranker.** Frozen sequential recommender даёт ID embeddings; MLP-projector переводит их в behavioral tokens; они соединяются с текстовыми токенами для Llama 2-7B. Curriculum: text-only → hybrid prompts, LoRA + projector | MovieLens-100K, Steam, LastFM; хронологическое деление последовательностей 8:1:1, последние 10 взаимодействий в контексте; sampled candidate set с обязательным правильным следующим item; **HitRatio@1 и valid ratio**, пять seed runs | Подтверждает полезность сочетания behavioral representations и языкового решения, но **не** nDCG@10 по всем будущим Buy и **не** два независимых LLM-этапа. Требует обучения и нескольких абляций. V2 имела другое название, шесть авторов и два датасета; здесь используется v4 с семью авторами и LastFM — версии не смешаны | **Полный текст**, HTML v4; ранняя v2 просмотрена, но не источник финального протокола |

### 2.3. Финансовые активы, FAR-Trans и LLM-советники

| Работа: авторы, год, площадка | Трек / метод | Данные и оценка | Ограничения и связь с нашей задачей | Прочитано |
|---|---|---|---|---|
| **[8] Sanz-Cruzado, Droukas, McCreadie, “FAR-Trans: An Investment Dataset for Financial Asset Recommendation,” Fin-RecSys / IJCAI Workshop 2024**, v1 | **Финансы, non-LLM benchmark.** Random, price regressors, Popularity, CF/ARM, LightGCN, гибридные LightGBM/LambdaMART | FAR-Trans; бинарные Buy, **новые пары**; пользователи с train/test взаимодействиями, активы с ценовым покрытием; 61 временной вариант, шаг около двух недель, горизонт 6 месяцев; nDCG@10 и ROI@10. Подбор гиперпараметров по ROI на отдельных датах | Главный источник benchmark. Показывает различие ranking accuracy и доходности; опубликованные результаты — средние по другим срезам, не наш локальный результат. Восстановленные владения, скорректированные цены/количества и оценённые профили требуют собственного leakage-аудита. LambdaMART уже есть в классической части, но это не LLM vs same-information reranker эксперимент | **Полный текст**, HTML v1 на этапе данных; библиография перепроверена |
| **[9] Spadea, Seneviratne, “Aligning Language Models with Investor and Market Behavior for Financial Recommendations” (FLARKO), ICAIF 2025**, v1 | **Финансы, LLM generation + KG + preference alignment.** Personal/Market KG в JSON-LD, Qwen3, LoRA/quantization, KTO; централизованный и federated варианты | FAR-Trans; training history до 2021-11-30, test prompts с 2021-12-01 по 2022-06-02, будущие метки на **180 дней**; три рекомендации. **Pref@3, Prof@3, Comb@3**: совпадение с покупками, прибыльность, пересечение. KTO desirability требует Buy **и** положительную доходность | Ближайший доменный LLM-предшественник: реальные финансовые истории и будущие покупки на том же наборе. Авторы сообщают преимущество по Pref/Comb, но это не наш nDCG@10, не permutation-only fixed-pool тест и не изоляция эффекта LLM при одинаковых входных полях у всех контролей. Таблицы абляций не поддерживают универсальное “больше контекста лучше”: combined иногда хуже PKG, MKG или nothing; прирост не монотонен по размеру модели | **Полный текст**, HTML v1, §§3–7 и таблицы/заключение |
| **[10] Takayanagi, Izumi, Sanz-Cruzado, McCreadie, Ounis, “Are Generative AI Agents Effective Personalized Financial Advisors?,” SIGIR 2025**, v2 | **Финансы, conversational advisor.** Llama-3.1-8B выявляет предпочтения, затем обсуждает активы; сравниваются personalization и persona | Лабораторная study: 64 recruited, 60 valid participants; три экспертно заданных investor profiles и по четыре обсуждаемых актива; elicitation accuracy, Spearman correlation с экспертным ranking, trust/satisfaction questionnaires. Контекст — цены/описания/отчёты, не FAR-Trans purchase prediction | При успешном выявлении предпочтений решения ближе к экспертным, при ошибках советник может ухудшить их. Более приятная persona может вызывать больше доверия при худшем совете. Это основание **не** подменять observed Buy нормативной пригодностью и **не** считать generated profile наблюдаемым. Результаты не дают nDCG@10 по покупкам и не доказывают клиентскую доходность | **Полный текст**, HTML v2 |
| **[11] Park et al., “Fund2Persona: A Framework for Building and Refining Financial Advisor Personas from Fund Disclosure Data,” EMNLP Industry 2026**, arXiv v4 | **Финансы, grounded personas / advisor.** SEC disclosures + holdings → persona; actor–scorer–patcher refinement, отдельная validation, затем фиксированная persona; matched/mismatched controls | Главная оценка — 69 U.S. equity funds, chronological replay/validation/held-out holdings reconstruction; Accuracy/Macro-F1 активных изменений, commentary alignment. **FAR-Trans используется в §5.2 для синтеза профилей 50 инвесторов и пятиходовых диалогов**, оценённых pairwise LLM-judge с pilot human validation | Важное более новое LLM-применение FAR-Trans: нельзя писать “из LLM-работ есть только FLARKO”. Но Top-10 здесь ограничивает **исходные holdings кандидатов**, а не оценивает Top-10 будущих Buy клиентов через nDCG. Reconstruction получает realized returns периода: этот контекст нельзя переносить в ex-ante прогноз нашего cut. Judge preference и synthetic dialogue не равны фактическим инвестиционным действиям | **Полный текст**, HTML v4, включая Appendices A–C; площадка сверена в ACL |
| **[12] Y. Wang et al., “Conv-FinRe: A Conversational and Longitudinal Benchmark for Utility-Grounded Financial Recommendation,” 2026**, SIGIR Resource Track — acceptance указана на arXiv, v2 pre-camera-ready | **Финансы, LLM ranking/advisory benchmark.** Разделяет user choice, calibrated utility, momentum и risk; inverse optimization задаёт reference utility; LLM агрегирует конкурирующие advisory signals | Отдельный benchmark, **не FAR-Trans**: 10 участников, 10 акций, 230 prefix-conditioned instances; рыночный горизонт Aug–Sep 2025, контролируемые синтезированные диалоги по реальным траекториям. uNDCG по utility, MRR/HR@1/3 по choice, Kendall τ по expert views | Показывает расхождение имитации поведения и utility; **uNDCG не равно бинарному purchase nDCG@10**. Небольшая и зависимая по пользователям выборка, модельно заданная utility и синтез диалогов ограничивают перенос. Нельзя добавлять такие utility labels как новую основную цель курса | **Полный текст**, HTML v2; acceptance/status — из первичной arXiv-страницы |
| **[13] Unno, Sakurai, Watanabe, Maeda, Ogawa, Haseyama, “One Basket, Many Experts: Decorrelated Re-ranking for Financial Asset Recommendation” (DSER), 2026**, accepted ICAIF 2026, v1 от 4 ноября | **Финансы, non-LLM reranker.** Expert-following core сохраняется, satellite slots заполняются активами с меньшей корреляцией доходностей; один контролируемый integer knob | FAR-Trans, очищенный universe **92 equities**, Top-10; nDCG@10/HR@10 по покупкам, ROI/Sharpe, walk-forward и более строгие неперекрывающиеся окна. **Buy-more setting** разрешает знакомые активы; выдаётся одна общая для пользователей корзина | Уже есть финансовый reranking с nDCG@10: широкая “новизна реранкинга на FAR-Trans” неверна. DSER может улучшать Sharpe ценой relevance; это другая цель. Не LLM, не персональный prompt, не строго перестановка фиксированного малого pool — satellites выбираются из universe. Stocks-only U92 не официальный multi-asset scope курса; экономический эффект не доказывает рост принятия советов | **Полный текст**, HTML v1, включая robustness/limitations; статус accepted подтверждён первичной страницей |

### 2.4. Повторное потребление — два non-LLM предшественника

| Работа: авторы, год, площадка | Трек / метод | Данные и оценка | Ограничения и связь с нашей задачей | Прочитано |
|---|---|---|---|---|
| **[14] P. Ren, Chen, J. Li, Z. Ren, Ma, de Rijke, “RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-Based Recommendation,” AAAI 2019**, препринт v1 2018 | **Repeat-aware, non-LLM.** GRU session encoder, repeat/explore gate и два декодера; старые и новые items получают разные вероятности | YOOCHOOSE, DIGINETICA, LASTFM; next-event/session task, preprocessing и split вслед за session-based benchmark; MRR/Recall@10/20; абляции repeat/attention, отдельно повторные и новые события. Точные календарные границы не дописываем там, где статья ссылается на предшествующий протокол | Обосновывает разделение механизма повторов и открытия нового. Авторы также отмечают склонность избыточно повторять и ухудшение на некоторых non-repeat подмножествах. Финансовая покупка ≠ click/session и ≠ прослушивание; повтор может быть увеличением позиции, а не “полюбившимся товаром”. Для A repeat-head не получает допустимых меток | **Полный текст**, HTML v1; год/площадка/DOI сверены через Crossref |
| **[15] M. Li, Jullien, Ariannezhad, de Rijke, “A Next Basket Recommendation Reality Check,” ACM TOIS 2023**, arXiv v2 | **Repeat/explore evaluation, non-LLM.** Frequency/KNN/deep baselines; **GP-TopFreq** сначала заполняет своими частыми товарами, затем глобально популярными; анализ вклада повторов и новых items | TaFeng, Dunnhumby, Instacart; последний basket — target, train/valid/test разделены **по пользователям 72/8/20**, пять split repetitions, не наш глобальный time cut. Recall/nDCG/PHR@10/20; repeat/explore recall, состав выдачи и группы пользователей | Сильное основание для нашего Repeat+Popularity, но не заявление о дословной реализации GP-TopFreq: у нас другие события, tie-break и окно меток. Авторы показывают, что “успех сложной модели” часто объясняется повторением, а частотные baselines могут быть сильнее. Не доказывает этот результат в финансах; в статье нет сопоставимого runtime анализа | **Полный текст**, HTML v2, включая выводы/limitations |

## 3. Проверка библиографии и места в тексте

Это первичные места для проверки выводов; ссылки на библиографические сервисы
использованы только для авторов/года/площадки, не вместо чтения метода.

| Ref | Открытый текст / проверка библиографии | Ключевые места |
|---|---|---|
| [1] | [HTML v2](https://arxiv.org/html/2305.08845v2), [arXiv metadata](https://arxiv.org/abs/2305.08845), [Crossref](https://api.crossref.org/works/10.1007/978-3-031-56060-6_24) | §§2–3, Tables 2/4, §6 Limitations |
| [2] | [HTML v1](https://arxiv.org/html/2311.02089v1), [авторский README](https://github.com/Yueeeeeeee/LlamaRec) | §§3.2, 4 Evaluation/Main Results/Efficiency; Tables 1/3 |
| [3] | [HTML v1](https://arxiv.org/html/2609.00086v1), [metadata](https://arxiv.org/abs/2609.00086) | §2.1 явно same-pool ≠ same-information; §§2.2–2.4; Tables 1/2; Discussion |
| [4] | [HTML v2](https://arxiv.org/html/2305.13731v2), DOI в заголовке статьи | §§2.2–2.3, §3.1, §3.4 |
| [5] | [HTML v5](https://arxiv.org/html/2310.15950v5), DOI/WWW в заголовке | §§3.2–3.4, §4.1, Table 1; Appendix A.3 (пример неудачного reranking) |
| [6] | [HTML v4](https://arxiv.org/html/2312.02445v4), [metadata/version history](https://arxiv.org/abs/2312.02445) | §§4–5, особенно §5.1.3; не переносить протокол v2 на v4 |
| [7] | [ACL Anthology](https://aclanthology.org/2025.coling-main.45/) | Только Abstract, authors, venue/year/pages; полный PDF не разобран |
| [8] | [HTML v1](https://arxiv.org/html/2407.08692v1), [metadata](https://arxiv.org/abs/2407.08692) | Dataset, §§5.1–5.2; подробный аудит в source.md/schema.md/leakage.md |
| [9] | [HTML v1](https://arxiv.org/html/2510.15993v1), [metadata](https://arxiv.org/abs/2510.15993) | §§3–4, §6.1/6.3/6.5; Tables 2–4 и §7 |
| [10] | [HTML v2](https://arxiv.org/html/2504.05862v2), DOI подтверждается библиографией [11] | §§3–5, Tables 3–5; в HTML есть шаблонная неправильная дата конференции, она не используется |
| [11] | [HTML v4](https://arxiv.org/html/2606.29793v4), [ACL metadata](https://aclanthology.org/2026.emnlp-industry.249/) | §§3–5, особенно §5.2; Appendices A.2, B.8/B.12, C.4 |
| [12] | [HTML v2](https://arxiv.org/html/2602.16990v2), [arXiv acceptance metadata](https://arxiv.org/abs/2602.16990) | §§3.2–3.4, Tables 3–5; шаблонный Woodstock/2018 в HTML не считается площадкой |
| [13] | [HTML v1](https://arxiv.org/html/2611.05952v1), [arXiv acceptance metadata](https://arxiv.org/abs/2611.05952) | §4.1: universe/buy-more; §§5–7: ablation, overlapping windows, limitations |
| [14] | [HTML v1](https://arxiv.org/html/1812.02646v1), [Crossref](https://api.crossref.org/works/10.1609/aaai.v33i01.33014806) | §§3–5; Table 4/5: repeat vs non-repeat |
| [15] | [HTML v2](https://arxiv.org/html/2109.14233v2), [Crossref](https://api.crossref.org/works/10.1145/3587153) | §3.1/3.2, §§5–6; GP-TopFreq и limitations |

Замечания к качеству источников: в LlamaRec HTML остались ACM-template
Woodstock/2018 и placeholder DOI — они не свидетельство площадки. Использован
авторский README с PGAI@CIKM 2023, а в References сохранён arXiv identifier.
В LLaRA зафиксирована финальная v4, в RLMRec — поздняя v5: не объявляем её
побуквенно тождественной proceedings-версии WWW.
Для DSER на дату обзора конференция ещё впереди, поэтому написано **accepted**,
хотя DOI уже указан в авторском тексте.

## 4. Nearest prior work и узкий пробел

### Ближайший доменный LLM-предшественник — FLARKO [9]

**Почему он:** тот же FAR-Trans, реальные истории клиентов и будущие покупки,
а не только финансовые тексты, sentiment, портфель фонда или выдуманный диалог.
Это ближе к нашей purchase-prediction цели, чем Fund2Persona [11], хотя [11]
новее и тоже использует FAR-Trans в отдельной advisory study.

**Что показал:** авторы продемонстрировали применимость KG-grounded Qwen3 с KTO,
в централизованном и федеративном режимах, и сообщили улучшение Pref@3/Comb@3
относительно выбранных baseline. Они также показали зависимость от вида контекста,
размера модели и федеративной настройки. Нельзя из общей формулировки заключения
делать вывод, что объединённый PKG+MKG всегда лучше: этому противоречат отдельные
строки опубликованных абляций.

**Что остаётся открытым именно для нас:** Top-3 hit-style metrics не определяют
Top-10 position-discounted качество. KTO-desirability по Buy∩profit оптимизирует
другую цель; генерация по KG не равна перестановке фиксированных кандидатов.
В прочитанном тексте не представлена искомая matched-information проверка
zero-shot реранкера по purchase nDCG@10 с явно разделёнными A/B и нашим аудитом
ретроспективных входов. Это ограничение покрытия статьи, не обвинение в утечке
и не утверждение о неработоспособности FLARKO.

**Формулировка пробела (ограниченная выполненным поиском):**

> Мы не нашли работу, которая сообщает изменение среднего nDCG@10 по будущим
> покупкам пользователей FAR-Trans от zero-shot LLM-переранжирования фиксированного
> набора кандидатов относительно non-LLM ranker, при условии одинаковых доступных
> к моменту прогноза входных полей, единого временного среза и eligible universe,
> явного разграничения новых и повторных покупок и отчётности о runtime/cost.

Это рабочий кандидат на аргумент для Ranker, **не зафиксированный RQ** и не
обещание положительного результата. Не утверждаем, что каждое отдельное условие
ново. Не добавляем сюда произвольную будущую дату нашего split как искусственный
критерий “новизны”. Перед итоговым PDF следует повторить целевой поиск по
сформулированному одному вмешательству и перечитать выбранные 3–5 источников.
Для официального сравнения нужно принять правило меток преподавателя;
A/B сейчас — диагностика, не два самовольно введённых общих benchmark.

### Не путать доменную и методическую близость

- **Zero-shot Ranker:** [1] — почти тот же тип вмешательства вне финансов.
  [3] — особенно близкая проверка candidate control, сильного baseline и stability,
  но **не same-information** эксперимент и не FAR-Trans.
- **Дообученный Ranker:** [2] — retriever→verbalizer-ranker; его время не переносим
  на наше оборудование или на zero-shot вариант.
- **Encoder:** [4] и [5] — ближайшие прочитанные представительные рецепты;
  после выбора вмешательства nearest prior work для encoder-RQ может стать [5],
  а не FLARKO. Проверять надо прирост от кодирования/выравнивания, а не наличие LLM.
- **Hybrid:** [6] — behavioral tokens + LLM, [5] — semantic↔CF alignment.
  Они не доказывают преимущество композиции двух независимых LLM-этапов.
- **Финансовый reranking без LLM:** [13] уже сообщает nDCG@10 на FAR-Trans.
  Поэтому пробел “на FAR-Trans не изучен reranking / nDCG@10” отвергается.
- **Повторы:** [15] — ближайшая оценочная аналогия для B и уже реализованного
  Repeat+Popularity; сама высокая частота повторов новым вкладом не является.

**Научная важность.** После выбора одного вмешательства полезно установить,
переносятся ли преимущества языковой семантики/ранжирования на финансовые Buy
при бедном тексте и сильном повторном поведении, или наблюдаемый эффект объясняется
информацией, retrieval и правилом меток. Это следует из неодинаковых результатов
и протоколов [1]–[6], различия целей в [8], [9], [12] и роли повторов в [15].
Честный нулевой результат тоже уточняет границы применимости метода.

**Бизнес-ценность.** Для владельца рекомендательного продукта банка или брокера
результат может изменить решение о том, оправдано ли включать LLM в retrieval/
ranking с её задержкой и стоимостью вместо структурного baseline или простого
repeat-aware решения. Для compliance-reviewer важна маркировка оценённых
предпочтений и граница между предсказанием поведения и suitability [10]–[12].
Этот offline эксперимент не измеряет прибыль, принятие советов или безопасность
реального консультирования и не обещает их улучшения.

## 5. Риски для каждого трека: литература × наши данные

Ниже слова “нужно проверить” означают план будущего дизайна, не уже выполненную
абляцию. Нулевой или отрицательный результат допустим.

### 5.1. Ranker

| Риск из литературы | Почему опасен на FAR-Trans по нашим отчётам | Требование к будущему сравнению |
|---|---|---|
| Position/popularity bias и слабая чувствительность к порядку истории [1]; нестабильность при сходном среднем качестве [3] | Покупки концентрированы; список в порядке классического score может навязать LLM исходный ranking. Повторяющаяся история одного актива не равна богатому предпочтению | Зафиксировать pool и его исходный score; перестановки кандидатов с агрегированием и стоимостью; recency/frequency как отдельные доступные поля, не скрытая подсказка только LLM |
| Retrieval ceiling и optimistic sampled/valid-subset оценка [1]–[3] | Relevant Buy не обязаны попадать в pool; у A/B разные пользователи и степень лёгкости | Не добавлять будущие Buy в кандидаты, не отбрасывать пользователей с candidate miss; сообщать Recall@K/oracle отдельно, основной nDCG считать по общему eligible universe |
| Same-pool не означает same-information [3] | LLM легко получает риск, ёмкость, sector и названия, а обычный CF — только ID | Non-LLM same-information ranker, например LambdaMART с теми же доступными полями; выбор baseline ещё впереди. Исходный retrieval order — отдельный reference, не достаточный контроль |
| Out-of-candidate generation [1], [5], [6] | Тикеры/имена могут быть неоднозначны; ISIN должен принадлежать разрешённым кандидатам | Валидировать ID, уникальность и длину Top-10; правила invalid outputs и fallback зафиксировать заранее и учитывать стоимость |
| Preference hallucination, доверие не равно качеству [10] | Часть профилей estimated, некоторые анкеты позже cut; названия активов могут активировать поздние знания модели | Не выдавать inferred preference за observed; маскировать недоступные поля, проверять анонимизацию названий. Ground truth — реальные допустимые Buy, не LLM-judge |

### 5.2. Encoder

| Риск из литературы | Почему опасен у нас | Требование к будущему сравнению |
|---|---|---|
| Семантика сама по себе не равна collaborative relevance [5]; transfer в [4] требует специального обучения | Нет rich descriptions/reviews; почти половина sector отсутствует, история бедна разнообразием | Frozen text/LLM encoder сравнивать с one-hot/структурным кодом и простым текстовым encoder на тех же полях; тот же retrieval/backbone, ablation дополнительных источников |
| Галлюцинация при создании user/item profiles [5], [10] | Нельзя восстановить ответы анкет, новости или инвестиционные цели из нескольких повторяющихся Buy | Генерируемое описание — estimated; запрещать добавление непредоставленных фактов; отдельно сравнить template-only и LLM-summary представления |
| Утечка при предвычислении представлений на всём релизе | Backdated покупки рассчитаны из будущих продаж; late questionnaire может попасть в “старый” snapshot; имена/ценовые поправки ретроспективны | Encoder, CF graph, aggregates и cache строить по разрешённому префиксу среза; хранить provenance. Простое timestamp≤cut не гарантирует чистоту исходного релиза |
| Неэквивалентный temporal protocol [4], [5] | Leave-one-out или общий 3:1:1 split не равны будущему окну всех Buy при одном cut | Не импортировать опубликованные числа и не объявлять candidate recall равным purchase nDCG@10; обучать/подбирать только на прошлых окнах |

### 5.3. Hybrid / alignment / encoder→ranker

| Риск из литературы | Почему опасен у нас | Требование к будущему сравнению |
|---|---|---|
| Изменение retriever меняет nDCG LLM не меньше, чем смена ranker [3] | Замена encoder может улучшить доступность релевантных активов, а не качество рассуждения | Разделить baseline, encoder-only, ranker-only, both. Для изоляции ranker фиксировать pool; для end-to-end encoder не навязывать искусственно один промежуточный pool всем методам |
| Ошибочное отождествление alignment и двух LLM-стадий [5], [6] | Дополнительный CF graph/projector/LoRA меняет сразу несколько компонентов | Точно назвать изменяемый компонент; сравнить одинаковые backbone, поля, бюджет настройки; убрать только изучаемый механизм в абляции |
| Сложность, цена и attribution | На малом разнообразии историй и бедном тексте обе части могут просто воспроизводить популярность/повторы | До Hybrid проверить простые ориентиры; раздельно учитывать offline generation/encoding/training и online retrieval/reranking, кеширование и полный runtime/cost |
| Двойное распространение загрязнённых признаков | Утечка может запечься в embeddings и затем снова попасть в prompt/KG; full-period profitability особенно опасна | Один общий as-of feature contract для обеих частей; не использовать maxDate/priceMaxDate/profitability как входы, исключить распознаваемую синтетику по согласованной политике |

### 5.4. Финансовые цели и повторное потребление — общие риски

- [8]–[13] различают behavioral accuracy, profit, utility и suitability. [10]
  показывает, что пользователи могут больше доверять худшему совету. Наш nDCG
  не доказывает доходность, пригодность, рост принятия советов или causal effect.
- В [11] “Top-10 holdings” — не наш “Top-10 будущих покупок”; [12] uNDCG на
  calibrated utility нельзя сравнивать с бинарным Buy-nDCG. Внешние realized
  returns из reconstruction нельзя давать нашему методу в момент cut.
- В B работа [15] требует сильного **personal frequency + global popularity**
  baseline; у нас он уже есть. Одного сравнения с глобальной популярностью мало.
  В A repeat-преимущество структурно запрещено, а не “не сумело обучиться”.
- По [14], [15] среднее качество может скрывать проигрыш на новых items.
  Если официальный протокол разрешит B, полезны отдельные диагностические
  результаты на repeat/explore targets и группах по **train-only** активности.
  Группировать/настраивать deployable модель по будущей склонности к повторам нельзя.
- Финансовое увеличение позиции и basket re-consumption — разные механизмы.
  Нет оснований считать 69% event-level повторов нормативной потребностью
  пользователя или целевой долей повторов в выдаче.
- [13] уже использует buy-more setting, а [8] исключает знакомые пары:
  даже публикации на одном FAR-Trans не задают единственное общее правило меток.
  Ни U92, ни stocks-only scope, ни метрики Sharpe не заменяют будущий протокол курса.

## 6. Журнал запросов, отбора и ограничений поиска

Все строки относятся к текущему поиску 2026-11-13. Номера Q — журнал действий,
не число найденных публикаций. Цитируемые работы открыты по первичным ссылкам
в §3. Поиск включает backward snowballing по References открытых статей,
но работы из чужой библиографии без собственного открытия в таблицы не добавлялись.

| ID | Выполненный запрос | Что найдено / решение |
|---|---|---|
| Q1 | `FAR-Trans LLM recommendation FLARKO reranking financial assets` | FLARKO [9], FAR-Trans [8], RURA, GitHub-проекты, UR4Rec. Открыты [8], [9], [7]; GitHub-проект не принят за научную статью |
| Q2 | `large language models recommendation encoder ranker alignment collaborative RLMRec LLaRA RecFormer LLMRank` | RLMRec [5], LLaRA [6], Recformer [4], tutorial и обзоры; выбраны первичные методы, не tutorial как доказательство эффекта |
| Q3 | `"Large Language Models are Zero-Shot Rankers" authors ECIR 2024` | [1], arXiv/ECIR/авторский код; авторы и acceptance проверены, DOI/year сверены отдельно |
| Q4 | `"LlamaRec" two stage recommendation efficient large language models authors venue` | [2]; не приписана площадка WSDM. PGAI@CIKM — по открытому авторскому README, основной доступный источник — препринт |
| Q5 | `"large language models" "financial advisors" paper personalized investment recommendations` | [10], FinPersona, обзоры/популярные статьи, financial advice работы. [10] выбран как primary user study; популярные объяснения не цитируются |
| Q6 | `"RepeatNet" "A Next Basket Recommendation Reality Check"` | Результаты смешивали ReCANet, next-period и next-basket; запрос разделён в Q7/Q8 |
| Q7 | `"RepeatNet" Ren 2019 arxiv` | [14]; открыты arXiv abstract и полный HTML; publication year проверен через Crossref |
| Q8 | `"Next Basket Recommendation" "Reality Check" authors` | [15]; выбран нужный next-basket title, не другая next-period работа; открыты полный HTML и Crossref |
| Q9 | `"FAR-Trans" "LLM" 2026 recommendation` | Fund2Persona [11], Conv-FinRe [12], FLARKO [9]; **[11] действительно использует FAR-Trans в case study, [12] только сопоставляет benchmark и использует свои данные** |
| Q10 | `"RepeatNet" "10.1609"` | DOI [14]; publisher страница защищена anti-bot, но статья открыта в arXiv HTML, библиография в Crossref |
| Q11 | `"Large Language Models are Zero-Shot Rankers" "364" ECIR` | Страницы/DOI [1]; Springer page не загрузилась, сведения проверены через Crossref, полный текст уже открыт в arXiv |
| Q12 | `"Are Generative AI Agents Effective Personalized Financial Advisors" SIGIR 2025 doi` | DOI и SIGIR 2025 для [10]; сверено с открытым полным текстом и библиографией [11], а не ошибочной template-датой |
| Q13 | `"Enhancing Reranking for Recommendation" "arxiv"` | Первичной альтернативной полной HTML-версии [7] не установлено. Найдены новые reranking работы и [3]; [7] оставлен только как abstract-level lead |
| Q14 | `"Risk-Aware Utility Re-Ranking for Financial Asset" Sakurai 2026` | RURA и свежая DSER [13]; RURA publisher вернул 403, авторская страница подтверждает название/venue, но не заменяет текст. В основную таблицу RURA **не включена**, открыта и включена [13] |
| Q15 | `recommendation "LLM encoder" "LLM reranker" two stage` | Много Text2SQL/RAG и сторонних пересказов. Они не включены как recommender evidence; не выдан generic retrieval за пользовательскую рекомендацию |
| Q16 | `LLM recommendation "encoder" "ranker" "hybrid" embeddings retrieval paper 2026` | Дополнительные cold-start/retrieval следы и блоги. Зафиксированы как нерассмотренные дальше; основной encoder/ranker coverage основан на открытых [1]–[6], не заявляется полнота всех работ 2026 |
| Q17 | `"FAR-Trans" "LLM" "nDCG" reranking` | Поиск не дал проверенного совпадения нужного controlled design; найденный автоматический пересказ не используется. Отсутствие точного search hit не доказывает отсутствие работы |
| Q18 | `"FAR-Trans" "LambdaMART" "LLM"` | В выдаче в основном общие LTR-материалы; не найден проверенный matched-information LLM-vs-LambdaMART FAR-Trans эксперимент. Общие блоги не включены в References |

Дополнительно по ссылкам из выдачи/прочитанных статей открыты:

- [11] arXiv abstract → полный HTML v4 → ACL metadata; проверены назначение
  FAR-Trans и отличие ex-post reconstruction от ex-ante purchase prediction.
- [12] abstract → полный HTML; проверены собственный сбор и utility-based labels.
- [13] abstract → полный HTML от 4 ноября; это существующая финансовая
  nDCG@10 reranking работа, а не LLM-работа.
- [3] abstract → полный HTML; прочитано явное ограничение same-information,
  которое особенно важно для честного будущего контроля.
- [6] сначала открыта v2, затем обнаружена и полностью прочитана v4; старые
  title/authors/datasets не смешаны с новой версией.

**Что отброшено и почему:** generic RAG/search/Text2SQL — другая задача;
торговые агенты/price prediction сами по себе не являются персональной Top-10
purchase recommendation; tutorial, Medium, автоматические обзоры и GitHub-only
проекты не заменяют scientific paper. RURA — недостаточно доступного первичного
текста. FinPersona и дополнительные cold-start работы — отсечены по объёму после
выбора более релевантных primary studies, не объявляются нерелевантными вообще.
UR4Rec не используется для сильных выводов, пока не прочитан полный PDF.
Отсечение по объёму и неполный поиск ограничивают надёжность заявления о пробеле.

## 7. IEEE References

Для всех онлайн-источников дата обращения — **Nov. 13, 2026**.
Ссылки на полные прочитанные версии даны в §3; ниже bibliographic citation,
а не утверждение о тождестве разных версий текста.

[1] Y. Hou, J. Zhang, Z. Lin, H. Lu, R. Xie, J. McAuley, and W. X. Zhao,
“Large Language Models are Zero-Shot Rankers for Recommender Systems,” in
*Advances in Information Retrieval (ECIR 2024)*, 2024, pp. 364–381,
doi: [10.1007/978-3-031-56060-6_24](https://doi.org/10.1007/978-3-031-56060-6_24).
[Online]. Available: https://arxiv.org/abs/2305.08845v2

[2] Z. Yue, S. Rabhi, G. de Souza Pereira Moreira, D. Wang, and E. Oldridge,
“LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking,”
arXiv:2311.02089, 2023. [Online]. Available:
https://arxiv.org/abs/2311.02089v1

[3] A. Kapetanovic, T. Duricic, A. Mercep, and E. Lacic,
“Retrieval, Scoring, and Decoding Shape Performance and Stability in LLM-based
Conversational Recommendation,” in *Proc. 35th ACM Int. Conf. Information and
Knowledge Management (CIKM)*, 2026,
doi: [10.1145/3799682.3840066](https://doi.org/10.1145/3799682.3840066).
[Online]. Available: https://arxiv.org/abs/2609.00086v1

[4] J. Li, M. Wang, J. Li, J. Fu, X. Shen, J. Shang, and J. McAuley,
“Text Is All You Need: Learning Language Representations for Sequential
Recommendation,” in *Proc. 29th ACM SIGKDD Conf. Knowledge Discovery and Data
Mining*, 2023, pp. 1258–1267,
doi: [10.1145/3580305.3599519](https://doi.org/10.1145/3580305.3599519).
[Online]. Available: https://arxiv.org/abs/2305.13731v2

[5] X. Ren, W. Wei, L. Xia, L. Su, S. Cheng, J. Wang, D. Yin, and C. Huang,
“Representation Learning with Large Language Models for Recommendation,” in
*Proc. ACM Web Conference (WWW)*, 2024,
doi: [10.1145/3589334.3645458](https://doi.org/10.1145/3589334.3645458).
[Online]. Available: https://arxiv.org/abs/2310.15950v5

[6] J. Liao, S. Li, Z. Yang, J. Wu, Y. Yuan, X. Wang, and X. He,
“LLaRA: Large Language-Recommendation Assistant,” in *Proc. 47th Int. ACM SIGIR
Conf. Research and Development in Information Retrieval*, 2024,
doi: [10.1145/3626772.3657690](https://doi.org/10.1145/3626772.3657690).
[Online]. Available: https://arxiv.org/abs/2312.02445v4

[7] H. Zhang, Q. Zhu, and Z. Dou, “Enhancing Reranking for Recommendation with
LLMs through User Preference Retrieval,” in *Proc. 31st Int. Conf.
Computational Linguistics (COLING)*, 2025, pp. 658–671.
[Online]. Available: https://aclanthology.org/2025.coling-main.45/

[8] J. Sanz-Cruzado, N. Droukas, and R. McCreadie, “FAR-Trans: An Investment
Dataset for Financial Asset Recommendation,” in *IJCAI Workshop on Recommender
Systems in Finance (Fin-RecSys)*, 2024. [Online]. Available:
https://arxiv.org/abs/2407.08692v1

[9] F. Spadea and O. Seneviratne, “Aligning Language Models with Investor and
Market Behavior for Financial Recommendations,” in *Proc. 6th ACM Int. Conf.
AI in Finance (ICAIF)*, 2025,
doi: [10.1145/3768292.3770399](https://doi.org/10.1145/3768292.3770399).
[Online]. Available: https://arxiv.org/abs/2510.15993v1

[10] T. Takayanagi, K. Izumi, J. Sanz-Cruzado, R. McCreadie, and I. Ounis,
“Are Generative AI Agents Effective Personalized Financial Advisors?,” in
*Proc. 48th Int. ACM SIGIR Conf. Research and Development in Information
Retrieval*, 2025, pp. 286–295,
doi: [10.1145/3726302.3729897](https://doi.org/10.1145/3726302.3729897).
[Online]. Available: https://arxiv.org/abs/2504.05862v2

[11] S. Park, H. Lee, Z. Wang, A. Lopez-Lira, Y. Cha, C. Choi, J. Choi,
and Y. Lee, “Fund2Persona: A Framework for Building and Refining Financial
Advisor Personas from Fund Disclosure Data,” in *Proc. Conf. Empirical Methods
in Natural Language Processing: Industry Track (EMNLP)*, 2026, pp. 4212–4229,
doi: [10.18653/2026.emnlp-industry.249](https://doi.org/10.18653/2026.emnlp-industry.249).
[Online]. Available: https://aclanthology.org/2026.emnlp-industry.249/

[12] Y. Wang, Y. Han, L. Qian, Y. He, X. Peng, D. Feng, Z. Xie,
V. J. Zhang, R. Guo, F. Mo, J. Huang, Y. Chen, X. Liu, and J.-Y. Nie,
“Conv-FinRe: A Conversational and Longitudinal Benchmark for Utility-Grounded
Financial Recommendation,” arXiv:2602.16990, 2026, accepted SIGIR 2026 Resource
Track (pre-camera-ready version). [Online]. Available:
https://arxiv.org/abs/2602.16990v2

[13] R. Unno, K. Sakurai, K. Watanabe, K. Maeda, T. Ogawa, and M. Haseyama,
“One Basket, Many Experts: Decorrelated Re-ranking for Financial Asset
Recommendation,” accepted to *7th ACM Int. Conf. AI in Finance (ICAIF)*, 2026,
doi: [10.1145/3858063.3858783](https://doi.org/10.1145/3858063.3858783).
[Online]. Available: https://arxiv.org/abs/2611.05952v1

[14] P. Ren, Z. Chen, J. Li, Z. Ren, J. Ma, and M. de Rijke,
“RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-Based
Recommendation,” in *Proc. AAAI Conf. Artificial Intelligence*, vol. 33,
no. 1, 2019, pp. 4806–4813,
doi: [10.1609/aaai.v33i01.33014806](https://doi.org/10.1609/aaai.v33i01.33014806).
[Online]. Available: https://arxiv.org/abs/1812.02646v1

[15] M. Li, S. Jullien, M. Ariannezhad, and M. de Rijke,
“A Next Basket Recommendation Reality Check,” *ACM Transactions on
Information Systems*, vol. 41, no. 4, pp. 1–29, 2023,
doi: [10.1145/3587153](https://doi.org/10.1145/3587153).
[Online]. Available: https://arxiv.org/abs/2109.14233v2
