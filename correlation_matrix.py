#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
correlation_matrix.py

Скрипт строит корреляционную матрицу между заданными инструментами
(акции/ETF/крипто/FX и т.п.), визуализирует её в виде heatmap, выводит
топ-пары по модулю корреляции и сохраняет результаты в файлы.

Запуск (пример):
python correlation_matrix.py \
  --tickers AAPL,MSFT,SPY,TLT,GLD,BTC-USD \
  --start 2023-01-01 --end 2025-11-01 \
  --interval 1d --price-field adjclose \
  --method pearson --min-periods 60 \
  --abs-threshold 0.7 --top-k 8 \
  --annot --upper-triangle --figsize 12 10 \
  --output-dir ./output
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Блок визуализации
import matplotlib.pyplot as plt

# seaborn нужен для heatmap; если не установлен — дадим понятную ошибку
try:
    import seaborn as sns
except Exception as e:
    raise SystemExit(
        "Требуется пакет 'seaborn' для построения heatmap. "
        "Установите: pip install seaborn"
    ) from e

# Источник данных по умолчанию — Yahoo Finance
try:
    import yfinance as yf
except Exception as e:
    raise SystemExit(
        "Требуется пакет 'yfinance' для загрузки данных с Yahoo Finance. "
        "Установите: pip install yfinance"
    ) from e


# ----------------------------- Конфиги/структуры -----------------------------


@dataclass
class CLIConfig:
    tickers: List[str]
    csv_path: Optional[str]
    start: Optional[str]
    end: Optional[str]
    interval: str
    price_field: str
    method: str
    min_periods: int
    abs_threshold: float
    top_k: int
    output_dir: str
    figsize: Tuple[float, float]
    annot: bool
    upper_triangle: bool
    no_cache: bool
    log_level: str


# ----------------------------- Парсинг аргументов ----------------------------


def parse_args() -> CLIConfig:
    parser = argparse.ArgumentParser(
        description="Построение корреляционной матрицы между инструментами.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Список тикеров через запятую (используется, если не указан --csv).",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=str,
        default="",
        help="Путь к локальному CSV. Поддерживаются wide/long форматы (см. README/ТЗ).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="",
        help="Дата начала (YYYY-MM-DD). По умолчанию: 365 дней назад.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="",
        help="Дата конца (YYYY-MM-DD). По умолчанию: сегодня.",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=["1d", "1wk", "1mo"],
        help="Интервал агрегации данных при загрузке.",
    )
    parser.add_argument(
        "--price-field",
        type=str,
        default="adjclose",
        choices=["close", "adjclose", "open", "high", "low"],
        help="Какое поле цены использовать.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pearson",
        choices=["pearson", "spearman", "kendall"],
        help="Метод расчёта корреляции.",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=30,
        help="Минимальное число перекрывающихся наблюдений для пары.",
    )
    parser.add_argument(
        "--abs-threshold",
        type=float,
        default=0.7,
        help="Порог по модулю для вывода «сильных» корреляций.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Количество топ-пар по |корреляции| для вывода.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Директория для сохранения результатов.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[10.0, 8.0],
        help="Размер фигуры heatmap: ширина высота.",
    )
    parser.add_argument(
        "--annot",
        action="store_true",
        help="Подписывать значения корреляций на heatmap.",
    )
    parser.add_argument(
        "--upper-triangle",
        action="store_true",
        help="Маскировать нижний треугольник матрицы (показывать только верхний).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Форсировать свежую загрузку из Yahoo (отключить кэш yfinance).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Уровень логирования.",
    )

    ns = parser.parse_args()

    # Предобработка аргументов и дефолтные даты
    tickers = [t.strip() for t in ns.tickers.split(",") if t.strip()]
    csv_path = ns.csv_path.strip() or None

    # Если не указаны start/end — берём последний год
    if ns.end:
        end = ns.end
    else:
        end = datetime.utcnow().date().isoformat()

    if ns.start:
        start = ns.start
    else:
        start = (datetime.fromisoformat(end) - timedelta(days=365)).date().isoformat()

    cfg = CLIConfig(
        tickers=tickers,
        csv_path=csv_path,
        start=start,
        end=end,
        interval=ns.interval,
        price_field=ns.price_field,
        method=ns.method,
        min_periods=ns.min_periods,
        abs_threshold=ns.abs_threshold,
        top_k=ns.top_k,
        output_dir=ns.output_dir,
        figsize=(float(ns.figsize[0]), float(ns.figsize[1])),
        annot=bool(ns.annot),
        upper_triangle=bool(ns.upper_triangle),
        no_cache=bool(ns.no_cache),
        log_level=ns.log_level,
    )
    return cfg


# ------------------------------- Служебные функции ---------------------------


def setup_logging(level: str) -> None:
    """Настройка логгирования."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(levelname)s | %(message)s",
    )


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_price_field(field: str) -> str:
    """Карта соответствия названий полей между yfinance и нашими аргументами."""
    mapping = {
        "close": "Close",
        "adjclose": "Adj Close",
        "open": "Open",
        "high": "High",
        "low": "Low",
    }
    return mapping[field]


# ---------------------------- Загрузка/подготовка данных ---------------------


def load_prices_from_yahoo(
    tickers: List[str],
    start: Optional[str],
    end: Optional[str],
    interval: str,
    price_field: str,
    no_cache: bool,
) -> pd.DataFrame:
    """
    Загрузка цен из Yahoo Finance по списку тикеров.
    Возвращает DataFrame с индексом-датой и столбцами — по тикерам.
    """
    if not tickers:
        raise ValueError("Не переданы тикеры (--tickers) и не указан CSV (--csv).")

    yf_params = {
        "tickers": tickers,
        "start": start if start else None,
        "end": end if end else None,
        "interval": interval,
        "auto_adjust": False,  # используем явное поле 'Adj Close', если надо
        "progress": False,
        "group_by": "ticker",
        "threads": True,
    }

    # Отключение кэша при необходимости
    if no_cache and hasattr(yf, "cache"):
        try:
            yf.cache.disable()
        except Exception:
            pass

    logging.info(
        f"Загружаем с Yahoo: {len(tickers)} тикеров, {start}..{end}, interval={interval}"
    )
    raw = yf.download(**yf_params)

    if raw.empty:
        raise RuntimeError("Не удалось загрузить данные с Yahoo Finance.")

    col = _normalize_price_field(price_field)

    # Приводим к формату wide: индекс — дата, столбцы — тикеры
    # yfinance при множестве тикеров даёт мультииндекс колонок (Level 0 — тикер, Level 1 — поле)
    if isinstance(raw.columns, pd.MultiIndex):
        data = {}
        for t in tickers:
            if (t, col) in raw.columns:
                s = raw[(t, col)].copy()
                if s.notna().sum() == 0:
                    logging.warning(f"[Yahoo] Тикер {t}: нет данных по полю '{col}' — пропуск.")
                    continue
                data[t] = s
            else:
                logging.warning(f"[Yahoo] Тикер {t}: поле '{col}' отсутствует — пропуск.")
        if not data:
            raise RuntimeError("После фильтрации по полю цены данных не осталось.")
        df = pd.DataFrame(data)
    else:
        # Единичный тикер может вернуть обычный индекс колонок
        if col not in raw.columns:
            raise RuntimeError(f"Поле '{col}' не найдено в загруженных данных.")
        df = raw[[col]].rename(columns={col: tickers[0]})

    # Чистим полностью пустые столбцы и дубли дат
    df = df[sorted(df.columns)]
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index)
    return df


def load_prices_from_csv(path: str, price_field: str) -> pd.DataFrame:
    """
    Загрузка цен из локального CSV.
    Поддерживаются два формата:
      * Wide: Date, T1, T2, ...
      * Long: Date, Ticker, Close (или Price)
    Возвращает DataFrame: индекс — дата, столбцы — тикеры.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV не найден: {path}")

    logging.info(f"Загружаем CSV: {path}")
    df = pd.read_csv(path)
    df_cols = [c.lower() for c in df.columns]

    # Попытка детекта формата
    if "ticker" in df_cols and ("close" in df_cols or "price" in df_cols):
        # Long формат
        # Нормализуем имена колонок
        col_map = {c: c.lower() for c in df.columns}
        df = df.rename(columns=col_map)
        price_col = "close" if "close" in df.columns else "price"
        needed = {"date", "ticker", price_col}
        if not needed.issubset(set(df.columns)):
            raise ValueError("Long CSV должен содержать колонки: Date, Ticker и Close/Price.")
        df["date"] = pd.to_datetime(df["date"])
        pivot = df.pivot_table(index="date", columns="ticker", values=price_col, aggfunc="last")
        pivot = pivot.sort_index()
        return pivot
    else:
        # Wide формат
        if "date" not in df_cols:
            raise ValueError("Wide CSV должен содержать колонку 'Date'.")
        df = df.rename(columns={df.columns[df_cols.index("date")]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        # Оставляем только числовые столбцы (цены)
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError("В Wide CSV не найдено числовых столбцов с ценами.")
        df = df[numeric_cols]
        return df


def clamp_to_period(
    df: pd.DataFrame, start: Optional[str], end: Optional[str]
) -> pd.DataFrame:
    """Обрезаем по датам, если заданы."""
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
    return df


def align_intersection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выравнивание дата-индекса по пересечению (inner join по датам).
    Это важно для сопоставимости рядов с разными календарями (крипто vs акции).
    """
    # Удаляем полностью пустые столбцы
    df = df.dropna(axis=1, how="all")
    # Пересечение по датам: дропаем строки, где все NaN
    df = df.dropna(axis=0, how="all")
    # Запрещаем forward-fill для цен (чтобы не искажать доходности)
    return df


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразуем цены в лог-доходности: r_t = ln(P_t / P_{t-1}).
    Удаляем строки/столбцы, где не удаётся корректно рассчитать.
    """
    # Защита от нулевых/неположительных значений (для логарифма)
    invalid_mask = (prices <= 0) | ~np.isfinite(prices)
    if invalid_mask.any().any():
        logging.warning("Обнаружены неположительные/нечисловые цены — соответствующие точки будут удалены.")
        prices = prices.mask(invalid_mask)

    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all")
    # Оставим столбцы, где есть хотя бы 2 наблюдения для последующей корреляции
    good_cols = [c for c in returns.columns if returns[c].dropna().shape[0] >= 2]
    returns = returns[good_cols]
    return returns


def compute_correlation(
    returns: pd.DataFrame, method: str, min_periods: int
) -> pd.DataFrame:
    """Расчёт корреляционной матрицы с учётом min_periods."""
    # pandas.DataFrame.corr принимает min_periods
    corr = returns.corr(method=method, min_periods=min_periods)
    return corr


def extract_pairs(corr: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Построение таблицы пар i<j: (asset_1, asset_2, correlation, n_common_obs).
    n_common_obs — число перекрывающихся наблюдений по паре (ненулевых/ненан).
    """
    tickers = list(corr.columns)
    pairs = []
    # Предварительно создаём маски валидности для ускорения
    valid_mask = returns.notna()

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            c = corr.loc[t1, t2]
            if pd.isna(c):
                n = int(
                    (valid_mask[t1] & valid_mask[t2]).sum()
                )  # всё равно посчитаем перекрытие
                pairs.append((t1, t2, np.nan, n))
                continue
            # Перекрытие наблюдений
            n = int((valid_mask[t1] & valid_mask[t2]).sum())
            pairs.append((t1, t2, float(c), n))

    df_pairs = pd.DataFrame(
        pairs, columns=["asset_1", "asset_2", "correlation", "n_common_obs"]
    )
    # сортировка по модулю корреляции (NaN в конец)
    df_pairs = df_pairs.sort_values(
        by="correlation", key=lambda s: s.abs(), ascending=False, na_position="last"
    ).reset_index(drop=True)
    return df_pairs


# -------------------------------- Визуализация --------------------------------


def plot_heatmap(
    corr: pd.DataFrame,
    title: str,
    figsize: Tuple[float, float],
    annot: bool,
    upper_triangle: bool,
    out_path: str,
) -> None:
    """Построение и сохранение heatmap корреляций."""
    if corr.empty:
        raise ValueError("Матрица корреляций пуста — нечего визуализировать.")

    plt.figure(figsize=figsize)

    # Маска нижнего треугольника по желанию
    mask = None
    if upper_triangle:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    # Heatmap с дивергентной цветовой картой
    ax = sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        annot=annot,
        fmt=".2f" if annot else "",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
    )
    ax.set_title(title, pad=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"Сохранено изображение: {out_path}")


# --------------------------------- Выходные файлы ------------------------------


def save_outputs(
    corr: pd.DataFrame, pairs: pd.DataFrame, out_dir: str
) -> Dict[str, str]:
    """Сохранение CSV и PNG. Возвращает словарь путей."""
    _ensure_output_dir(out_dir)
    paths: Dict[str, str] = {}
    matrix_csv = os.path.join(out_dir, "correlation_matrix.csv")
    corr.to_csv(matrix_csv, float_format="%.6f")
    paths["matrix_csv"] = matrix_csv

    pairs_csv = os.path.join(out_dir, "correlation_pairs.csv")
    pairs.to_csv(pairs_csv, index=False, float_format="%.6f")
    paths["pairs_csv"] = pairs_csv

    return paths


def print_summary(
    pairs: pd.DataFrame, abs_threshold: float, top_k: int
) -> None:
    """Консольная сводка по топ-парам."""
    if pairs.empty:
        logging.warning("Нет пар для отображения.")
        return

    strong = pairs[pairs["correlation"].abs() >= abs_threshold].copy()
    to_show = strong.head(top_k)

    if to_show.empty:
        print(
            f"Top pairs by |correlation| (threshold={abs_threshold}, top_k={top_k}): нет пар, превышающих порог."
        )
        return

    print(
        f"Top pairs by |correlation| (threshold={abs_threshold}, top_k={top_k}):"
    )
    for i, row in to_show.reset_index(drop=True).iterrows():
        a1, a2, c, n = row["asset_1"], row["asset_2"], row["correlation"], int(row["n_common_obs"])
        sign = "+" if c >= 0 else "-"
        print(f"  {i+1:>2}) {a1:>8} — {a2:<8}: {c: .4f} ({'n=' + str(n)})")


# --------------------------------------- main ---------------------------------


def main() -> None:
    cfg = parse_args()
    setup_logging(cfg.log_level)

    # Проверка входных параметров
    if not cfg.csv_path and not cfg.tickers:
        raise SystemExit("Укажите --tickers или --csv.")

    # Загрузка цен
    if cfg.csv_path:
        prices = load_prices_from_csv(cfg.csv_path, cfg.price_field)
        # Если пользователь также задал start/end — обрежем
        prices = clamp_to_period(prices, cfg.start, cfg.end)
    else:
        prices = load_prices_from_yahoo(
            cfg.tickers,
            cfg.start,
            cfg.end,
            cfg.interval,
            cfg.price_field,
            cfg.no_cache,
        )

    # Предварительная чистка и выравнивание
    prices = align_intersection(prices)

    if prices.shape[1] < 2:
        raise SystemExit(
            f"Недостаточно инструментов после очистки: {prices.shape[1]} (нужно >= 2)."
        )

    logging.info(
        f"Загружено столбцов (тикеров): {prices.shape[1]}; строк (дат): {prices.shape[0]}"
    )

    # Преобразование в лог-доходности
    returns = to_log_returns(prices)

    if returns.shape[1] < 2:
        raise SystemExit(
            "После преобразования в доходности осталось менее 2 валидных инструментов."
        )

    # Расчёт корреляции
    corr = compute_correlation(returns, cfg.method, cfg.min_periods)

    # Извлечение пар
    pairs = extract_pairs(corr, returns)

    # Сводка в консоль
    logging.info(
        f"Используем лог-доходности, method={cfg.method}, min_periods={cfg.min_periods}."
    )
    print_summary(pairs, cfg.abs_threshold, cfg.top_k)

    # Сохранение результатов
    _ensure_output_dir(cfg.output_dir)
    paths = save_outputs(corr, pairs, cfg.output_dir)

    # Заголовок для графика
    title = (
        f"Correlation matrix (method={cfg.method}, "
        f"{returns.index.min().date()}..{returns.index.max().date()}, "
        f"n={returns.dropna().shape[0]})"
    )

    # Построение heatmap
    heatmap_path = os.path.join(cfg.output_dir, "correlation_heatmap.png")
    plot_heatmap(
        corr=corr,
        title=title,
        figsize=cfg.figsize,
        annot=cfg.annot,
        upper_triangle=cfg.upper_triangle,
        out_path=heatmap_path,
    )

    # Финальный вывод
    print("Saved:")
    print(f"  {paths['matrix_csv']}")
    print(f"  {paths['pairs_csv']}")
    print(f"  {heatmap_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОперация прервана пользователем.")