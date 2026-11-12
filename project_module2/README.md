# project_module2 — LLM4Rec, FAR-Trans (Session 1: Define)

Курс LLM4Rec, НИУ ВШЭ (когорта GSB/AVITO). Преподаватель: Seungmin Jin.
Команда: Кожокарь Алина, Целищева Ульяна, Бабикова Дарья.
Отчёт по Session 1 — индивидуальный (у каждого свой PDF + экспорт AI-сессии).

## Общая задача проекта

Для каждого оцениваемого пользователя — Top-10 активов, метрика nDCG@10 против
будущих покупок (Buy). Одинаковые пользователи, universe активов, временной сплит
и метрика для всех команд. Скоуп активов, time cut и список пользователей
**pending** — преподаватель зафиксирует до Session 2; все наши сплиты — допущения.

Данные: FAR-Trans (University of Glasgow, CC BY 4.0), <https://researchdata.gla.ac.uk/1658/>.
Сырые данные в репозиторий не коммитятся (`data/` в `.gitignore`).

## План Session 1

| # | Этап | Результат | Раздел отчёта |
|---|---|---|---|
| 1 | Разбор задания, структура папки | этот README | — |
| 2 | Изучение данных: схема, record-пример, таймстемпы, утечки, EDA | `reports/data_example.md`, `reports/eda.md` | §3 Method |
| 3 | Литература: 3–5 открытых статей, nearest prior work, gap | `reports/literature.md` | §2 Related Work |
| 4 | Выбор трека (Encoder / Ranker / Hybrid) и одного RQ | `DECISIONS.md` | §1 Introduction |
| 5 | Дизайн эксперимента: baseline, same-information control, ablation, nDCG@10 + runtime/cost, риски | `reports/study_plan.md` | §3–§4 |
| 6 | Командное обсуждение: варианты, согласие/расхождения, личный вклад | заметки | §5 Discussion |
| 7 | Сборка PDF (≤5 стр. без ссылок) + AI-disclosure + экспорт сессии | `<имя>_session1_report.pdf`, `<имя>_session1_session.json` | §6 |

## Структура

```
project_module2/
  README.md        этот файл: план и контекст
  DECISIONS.md     журнал решений (кто предложил, почему)
  data/            сырые CSV FAR-Trans (не коммитятся)
  scripts/         скрипты: схема, пример записи, EDA
  reports/         сгенерированные артефакты для отчёта
```
