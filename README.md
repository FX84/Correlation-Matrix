# Корреляционная матрица для акций и криптовалют

Скрипт `correlation_matrix.py` строит корреляционную матрицу между выбранными инструментами (акции, ETF, крипто, Forex), визуализирует её в виде **heatmap**, находит **топ-пары** с наибольшей по модулю корреляцией и сохраняет результаты в файлы (CSV/PNG). Подходит для быстрого ресёрча корзины тикеров, проверки диверсификации и исследования межрыночных связей.

---

## Возможности

* Загрузка данных:

  * с **Yahoo Finance** по тикерам (включая `BTC-USD`, `EURUSD=X`, ETF и т.д.);
  * из **локального CSV** (wide/long формат, авто-детект).
* Поддержка интервалов: `1d`, `1wk`, `1mo`.
* Методы корреляции: `pearson`, `spearman`, `kendall`.
* Лог-доходности (`ln(P_t/P_{t-1})`) для сопоставимого расчёта.
* **Heatmap** корреляций с опцией аннотаций и маски верхнего треугольника.
* Отчёт по **топ-парам** с порогом по модулю корреляции.
* Экспорт:

  * `correlation_matrix.csv` — полная матрица,
  * `correlation_pairs.csv` — пары `(asset_1, asset_2, correlation, n_common_obs)`,
  * `correlation_heatmap.png` — изображение тепловой карты.

---

## Установка

```bash
# Клонировать репозиторий
git clone https://github.com/FX84/Correlation-Matrix.git
cd Correlation-Matrix

# (рекомендуется) создать виртуальное окружение
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Установить зависимости
pip install -r requirements.txt
# или:
pip install pandas numpy matplotlib seaborn yfinance python-dateutil
```

> Python 3.9+

---

## Быстрый старт

### Пример 1 — Yahoo Finance (дневные данные)

```bash
python correlation_matrix.py \
  --tickers AAPL,MSFT,SPY,TLT,GLD,BTC-USD \
  --start 2023-01-01 --end 2025-11-01 \
  --interval 1d --price-field adjclose \
  --method pearson --min-periods 60 \
  --abs-threshold 0.7 --top-k 8 \
  --annot --upper-triangle --figsize 12 10 \
  --output-dir ./output
```

Ожидаемые файлы:

```
./output/correlation_matrix.csv
./output/correlation_pairs.csv
./output/correlation_heatmap.png
```

### Пример 2 — Локальный CSV

```bash
python correlation_matrix.py \
  --csv ./data/prices.csv \
  --method spearman \
  --abs-threshold 0.6 \
  --top-k 12 \
  --output-dir ./out
```

---

## Формат входного CSV

Скрипт автоматически определяет один из двух форматов.

**Wide формат**

```csv
Date,AAPL,MSFT,SPY
2024-01-02,185.10,375.20,475.10
2024-01-03,186.05,377.15,474.30
...
```

* Столбец `Date` + числовые столбцы с тикерами.

**Long формат**

```csv
Date,Ticker,Close
2024-01-02,AAPL,185.10
2024-01-02,MSFT,375.20
...
```

* Обязательные колонки: `Date`, `Ticker`, `Close` (или `Price`).

> Даты конвертируются автоматически и сортируются по возрастанию.

---

## Параметры CLI

```bash
python correlation_matrix.py --help
```

| Параметр           | Тип/значения                                   |       По умолчанию | Описание                                                  |
| ------------------ | ---------------------------------------------- | -----------------: | --------------------------------------------------------- |
| `--tickers`        | `str` (через запятую)                          |                  — | Тикеры Yahoo Finance. Обязателен, если не указан `--csv`. |
| `--csv`            | `str`                                          |                  — | Путь к локальному CSV (wide/long).                        |
| `--start`          | `YYYY-MM-DD`                                   | сегодня − 365 дней | Дата начала выборки.                                      |
| `--end`            | `YYYY-MM-DD`                                   |            сегодня | Дата конца выборки.                                       |
| `--interval`       | `1d` | `1wk` | `1mo`                           |               `1d` | Интервал данных.                                          |
| `--price-field`    | `close` | `adjclose` | `open` | `high` | `low` |         `adjclose` | Поле цены.                                                |
| `--method`         | `pearson` | `spearman` | `kendall`             |          `pearson` | Тип корреляции.                                           |
| `--min-periods`    | `int`                                          |               `30` | Минимум общих наблюдений в паре.                          |
| `--abs-threshold`  | `float [0..1]`                                 |              `0.7` | Порог для отбора сильных корреляций.                      |
| `--top-k`          | `int`                                          |               `10` | Сколько топ-пар вывести.                                  |
| `--output-dir`     | `str`                                          |         `./output` | Куда сохранять результаты.                                |
| `--figsize`        | `float float`                                  |             `10 8` | Размеры изображения heatmap.                              |
| `--annot`          | флаг                                           |            `False` | Подписи значений на heatmap.                              |
| `--upper-triangle` | флаг                                           |            `False` | Спрятать нижний треугольник матрицы.                      |
| `--no-cache`       | флаг                                           |            `False` | Отключить кэш `yfinance`.                                 |
| `--log-level`      | `DEBUG` | `INFO` | `WARNING` | `ERROR`         |             `INFO` | Уровень логов.                                            |

---

## Выходные файлы

* **`correlation_matrix.csv`** — квадратная матрица N×N с коэффициентами корреляций.
* **`correlation_pairs.csv`** — таблица пар:

  * `asset_1, asset_2, correlation, n_common_obs`
  * отсортирована по `|correlation|` по убыванию.
* **`correlation_heatmap.png`** — тепловая карта (`RdBu_r`, `vmin=-1`, `vmax=1`, `center=0`).

Пример фрагмента `correlation_pairs.csv`:

```csv
asset_1,asset_2,correlation,n_common_obs
AAPL,MSFT,0.8235,612
SPY,AAPL,0.7810,640
GLD,SPY,-0.2104,598
```

---

## Интерпретация

* **Высокая положительная** корреляция (близко к +1): активы движутся вместе → низкая диверсификация.
* **Высокая отрицательная** (близко к −1): активы движутся в противоположных направлениях → потенциальная хеджирующая пара.
* **Около 0**: линейной связи мало или нет.

> Метод `spearman` устойчивее к выбросам (ранговая корреляция), `kendall` — надёжнее на малых выборках, `pearson` — классическая линейная зависимость.

---

## Тонкости и ограничения

* Разные календари (крипто 24/7, акции — торговые дни) → датасеты выравниваются по **пересечению дат**.
* Для корректных лог-доходностей удаляются **неположительные** цены и нечисловые значения.
* Если после очистки остаток < 2 инструментов — скрипт останавливается с понятным сообщением.
* Для FX используйте нотацию Yahoo: напр. `EURUSD=X`, `USDJPY=X`.
