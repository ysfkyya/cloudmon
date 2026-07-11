"""
Özellik mühendisliği modülü.

İki görev için ayrı veri setleri hazırlar:
  • Kapasite tahmini  (regresyon): gelecekteki CPU/Bellek/Disk değerlerini tahmin et
  • Sağlık sınıflandırması (binary): sonraki N saatte risk var mı?

Özellik kategorileri
---------------------
  1. Ham metrikler
  2. Zaman özellikleri (saat, gün, hafta, iş günü vb.)
  3. Hareketli istatistikler (3h / 6h / 12h / 24h / 168h ortalama & standart sapma)
  4. Gecikme (lag) özellikleri (1h / 6h / 24h / 48h / 168h)
  5. Değişim hızı (delta) — 1h, 6h, 24h farkları
  6. Log özellikleri ve bunların hareketli ortalamaları
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    TRAIN_END_DATE, CAPACITY_TARGETS, FORECAST_HORIZONS,
    RISK_HORIZON, THRESHOLDS,
)

# Rolling & lag pencere boyutları
_ROLL_WINDOWS = [3, 6, 12, 24, 168]
_LAG_HOURS    = [1, 6, 24, 48, 168]
_DELTA_HOURS  = [1, 6, 24]

# Temel metrik sütunları — network ve vm_count dahil edildi
_METRIC_COLS = [
    "cpu_usage_percent",
    "memory_usage_percent",
    "disk_usage_percent",
    "api_latency_ms",
    "error_rate_percent",
    "network_in_mbps",
    "network_out_mbps",
    "api_requests_per_hour",
    "vm_count",
]

# Log sayım sütunları — tüm hata tipleri dahil edildi
_LOG_COUNT_COLS = [
    "total_errors",
    "critical_count",
    "timeout_count",
    "db_connection_count",
    "resource_exhausted_count",
    "quota_exceeded_count",
    "auth_failure_count",
    "network_error_count",
    "scheduler_error_count",
    "io_error_count",
]


def build_feature_matrix(metrics_df: pd.DataFrame, log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Metrik + log verilerini birleştirip tam özellik matrisini oluşturur.

    Parametreler
    ------------
    metrics_df : generate_metrics() çıktısı
    log_df     : generate_log_features() çıktısı

    Dönüş
    ------
    Özellikler + orijinal ham sütunlar; timestamp index.
    """
    df = metrics_df.merge(log_df, on="timestamp", how="left").set_index("timestamp")

    # ── Zaman özellikleri ─────────────────────────────────────────────────────
    df["hour_of_day"]     = df.index.hour
    df["day_of_week"]     = df.index.dayofweek
    df["day_of_month"]    = df.index.day
    df["week_of_year"]    = df.index.isocalendar().week.astype(int)
    df["month"]           = df.index.month
    df["is_weekend"]      = (df.index.dayofweek >= 5).astype(int)
    df["is_business_hour"] = (
        (df.index.hour >= 8) & (df.index.hour <= 18) & (df.index.dayofweek < 5)
    ).astype(int)

    # ── Hareketli istatistikler ───────────────────────────────────────────────
    for col in _METRIC_COLS:
        for w in _ROLL_WINDOWS:
            df[f"{col}_rmean_{w}h"] = df[col].rolling(w, min_periods=1).mean()
            if w <= 24:
                df[f"{col}_rstd_{w}h"]  = df[col].rolling(w, min_periods=1).std().fillna(0)

    # ── Gecikme özellikleri ───────────────────────────────────────────────────
    for col in _METRIC_COLS + ["total_errors", "error_log_ratio"]:
        for lag in _LAG_HOURS:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

    # ── Değişim hızı (delta) ─────────────────────────────────────────────────
    _DELTA_COLS = [
        "cpu_usage_percent", "memory_usage_percent", "disk_usage_percent",
        "api_latency_ms", "error_rate_percent", "vm_count",
    ]
    for col in _DELTA_COLS:
        if col in df.columns:
            for d in _DELTA_HOURS:
                df[f"{col}_delta_{d}h"] = df[col].diff(d)

    # ── Log hareketli toplamları ─────────────────────────────────────────────
    for col in _LOG_COUNT_COLS:
        df[f"{col}_rmean_6h"]  = df[col].rolling(6,  min_periods=1).mean()
        df[f"{col}_rsum_24h"]  = df[col].rolling(24, min_periods=1).sum()

    return df


def prepare_capacity_dataset(feature_df: pd.DataFrame):
    """
    Kapasite tahmini için (X_train, X_test, y_train, y_test) döndürür.

    Hedef sütunlar: <kaynak>_future_<horizon>h
    Döndürülen dict yapısı:
        {
          (kaynak, horizon): {
              "X_train": ..., "y_train": ...,
              "X_test":  ..., "y_test":  ...,
              "feature_names": [...],
          }
        }
    """
    df = feature_df.copy()

    # Gelecek değerleri hedef olarak ekle
    for h in FORECAST_HORIZONS:
        for col in CAPACITY_TARGETS:
            df[f"{col}_future_{h}h"] = df[col].shift(-h)

    # Tüm NaN satırları çıkar (lag + gelecek shift nedeniyle ilk/son satırlar)
    df = df.dropna()

    # Özellik sütunları = hedef olmayan ve ham metrik dışındaki sütunlar
    target_cols  = [f"{c}_future_{h}h" for c in CAPACITY_TARGETS for h in FORECAST_HORIZONS]
    feature_cols = [c for c in df.columns if c not in CAPACITY_TARGETS and c not in target_cols]

    train_mask = df.index < pd.Timestamp(TRAIN_END_DATE)
    test_mask  = ~train_mask

    datasets = {}
    for h in FORECAST_HORIZONS:
        for col in CAPACITY_TARGETS:
            tgt = f"{col}_future_{h}h"
            datasets[(col, h)] = {
                "X_train":       df.loc[train_mask, feature_cols].values,
                "y_train":       df.loc[train_mask, tgt].values,
                "X_test":        df.loc[test_mask,  feature_cols].values,
                "y_test":        df.loc[test_mask,  tgt].values,
                "feature_names": feature_cols,
                "timestamps_test": df.index[test_mask],
            }

    return datasets


def prepare_health_dataset(feature_df: pd.DataFrame):
    """
    Sağlık riski sınıflandırması için (X_train, X_test, y_train, y_test) döndürür.

    Etiket: sonraki RISK_HORIZON saatte herhangi bir kritik eşik aşılıyor mu? (0/1)
    """
    df = feature_df.copy()

    # Risk etiketi oluştur
    df["risk_label"] = _create_risk_label(df)

    df = df.dropna()

    target_cols  = [f"{c}_future_{h}h"
                    for c in CAPACITY_TARGETS for h in FORECAST_HORIZONS] + ["risk_label"]
    feature_cols = [c for c in df.columns
                    if c not in CAPACITY_TARGETS and c not in target_cols]

    train_mask = df.index < pd.Timestamp(TRAIN_END_DATE)
    test_mask  = ~train_mask

    return {
        "X_train":         df.loc[train_mask, feature_cols].values,
        "y_train":         df.loc[train_mask, "risk_label"].values,
        "X_test":          df.loc[test_mask,  feature_cols].values,
        "y_test":          df.loc[test_mask,  "risk_label"].values,
        "feature_names":   feature_cols,
        "timestamps_test": df.index[test_mask],
        "y_train_full":    df.loc[train_mask, "risk_label"],  # index'li
        "y_test_full":     df.loc[test_mask,  "risk_label"],
    }


def _create_risk_label(df: pd.DataFrame) -> pd.Series:
    """
    Sonraki RISK_HORIZON saatlik pencerede herhangi bir eşik aşımı varsa 1 döndürür.
    """
    risk = pd.Series(0, index=df.index, dtype=int)
    for h in range(1, RISK_HORIZON + 1):
        exceeded = (
            (df["cpu_usage_percent"].shift(-h)    > THRESHOLDS["cpu_usage_percent"])    |
            (df["memory_usage_percent"].shift(-h)  > THRESHOLDS["memory_usage_percent"]) |
            (df["error_rate_percent"].shift(-h)    > THRESHOLDS["error_rate_percent"])   |
            (df["api_latency_ms"].shift(-h)        > THRESHOLDS["api_latency_ms"])
        ).fillna(False)
        risk |= exceeded.astype(int)
    return risk
