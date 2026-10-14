"""
Özellik mühendisliği modülü.

Kapasite tahmini (regresyon) için veri seti hazırlar.
Sağlık sınıflandırması kaldırıldı; log pipeline analizi bu rolü devraldı.

Özellik kategorileri:
  1. Ham metrikler (vcpu, bellek, disk, gecikme, hata oranı)
  2. Zaman özellikleri (saat, gün, hafta, iş günü)
  3. Hareketli istatistikler (3/6/12/24/168 saat ortalama & std)
  4. Gecikme (lag) özellikleri (1/6/24/48/168 saat)
  5. Değişim hızı (delta) — 1/6/24 saat farkları
  6. Log sayımları ve erken uyarı sinyalleri (hareketli toplamlar)
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    TRAIN_END_DATE, CAPACITY_TARGETS, FORECAST_HORIZONS,
)

_ROLL_WINDOWS = [3, 6, 12, 24, 168]
_LAG_HOURS    = [1, 6, 24, 48, 168]
_DELTA_HOURS  = [1, 6, 24]

_METRIC_COLS = [
    "vcpu_usage_percent",
    "memory_usage_percent",
    "disk_usage_percent",
    "api_latency_ms",
    "error_rate_percent",
    "network_in_mbps",
    "network_out_mbps",
    "api_requests_per_hour",
    "vm_count",
]

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
    # Erken uyarı log sayıları
    "slow_response_count",
    "memory_pressure_count",
    "connection_pool_warn_count",
    "scheduler_degradation_count",
    "quota_pressure_count",
    "disk_io_degradation_count",
    "cascade_indicator_count",
    "total_warning_count",
]


def build_feature_matrix(metrics_df: pd.DataFrame, log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Metrik + log verilerini birleştirip tam özellik matrisini oluşturur.
    """
    df = metrics_df.merge(log_df, on="timestamp", how="left").set_index("timestamp")

    # ── Zaman özellikleri ─────────────────────────────────────────────────────
    df["hour_of_day"]      = df.index.hour
    df["day_of_week"]      = df.index.dayofweek
    df["day_of_month"]     = df.index.day
    df["week_of_year"]     = df.index.isocalendar().week.astype(int)
    df["month"]            = df.index.month
    df["is_weekend"]       = (df.index.dayofweek >= 5).astype(int)
    df["is_business_hour"] = (
        (df.index.hour >= 8) & (df.index.hour <= 18) & (df.index.dayofweek < 5)
    ).astype(int)

    # ── Yeni sütunları dict'te topla, sonra tek concat ──────────────────────
    new_cols: dict = {}

    # Hareketli istatistikler
    for col in _METRIC_COLS:
        s = df[col]
        for w in _ROLL_WINDOWS:
            new_cols[f"{col}_rmean_{w}h"] = s.rolling(w, min_periods=1).mean()
            if w <= 24:
                new_cols[f"{col}_rstd_{w}h"] = s.rolling(w, min_periods=1).std().fillna(0)

    # Gecikme özellikleri
    lag_cols = _METRIC_COLS + ["total_errors", "error_log_ratio", "total_warning_count"]
    for col in lag_cols:
        if col not in df.columns:
            continue
        s = df[col]
        for lag in _LAG_HOURS:
            new_cols[f"{col}_lag_{lag}h"] = s.shift(lag)

    # Değişim hızı (delta)
    _DELTA_COLS = [
        "vcpu_usage_percent", "memory_usage_percent", "disk_usage_percent",
        "api_latency_ms", "error_rate_percent", "vm_count",
        "slow_response_count", "memory_pressure_count", "connection_pool_warn_count",
    ]
    for col in _DELTA_COLS:
        if col not in df.columns:
            continue
        s = df[col]
        for d in _DELTA_HOURS:
            new_cols[f"{col}_delta_{d}h"] = s.diff(d)

    # Log hareketli toplamları
    for col in _LOG_COUNT_COLS:
        if col not in df.columns:
            continue
        s = df[col]
        new_cols[f"{col}_rmean_6h"] = s.rolling(6,  min_periods=1).mean()
        new_cols[f"{col}_rsum_24h"] = s.rolling(24, min_periods=1).sum()
        new_cols[f"{col}_rsum_72h"] = s.rolling(72, min_periods=1).sum()

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Erken uyarı bileşik sinyal
    warn_cols = [c for c in _LOG_COUNT_COLS if "count" in c and c in df.columns
                 and any(w in c for w in ["slow", "memory_pressure", "connection_pool",
                                          "scheduler_degrad", "quota_press",
                                          "disk_io_degrad", "cascade"])]
    if warn_cols:
        ew = df[warn_cols].sum(axis=1)
        df = pd.concat([
            df,
            pd.DataFrame({
                "early_warning_composite": ew,
                "early_warning_rmean_24h": ew.rolling(24, min_periods=1).mean(),
            }, index=df.index),
        ], axis=1)

    return df


def prepare_capacity_dataset(feature_df: pd.DataFrame) -> dict:
    """
    Kapasite tahmini için (X_train, X_test, y_train, y_test) döndürür.

    Hedef sütunlar: <kaynak>_future_<horizon>h
    """
    df = feature_df.copy()

    for h in FORECAST_HORIZONS:
        for col in CAPACITY_TARGETS:
            df[f"{col}_future_{h}h"] = df[col].shift(-h)

    df = df.dropna()

    target_cols  = [f"{c}_future_{h}h" for c in CAPACITY_TARGETS for h in FORECAST_HORIZONS]
    feature_cols = [c for c in df.columns if c not in CAPACITY_TARGETS and c not in target_cols]

    train_mask = df.index < pd.Timestamp(TRAIN_END_DATE)
    test_mask  = ~train_mask

    datasets = {}
    for h in FORECAST_HORIZONS:
        for col in CAPACITY_TARGETS:
            tgt = f"{col}_future_{h}h"
            key = f"{col}_{h}h"
            datasets[key] = {
                "X_train":         df.loc[train_mask, feature_cols].values,
                "y_train":         df.loc[train_mask, tgt].values,
                "X_test":          df.loc[test_mask,  feature_cols].values,
                "y_test":          df.loc[test_mask,  tgt].values,
                "feature_names":   feature_cols,
                "timestamps_test": df.index[test_mask],
            }
    # Eski tuple-key formatını da destekle (capacity_predictor uyumu için)
    for h in FORECAST_HORIZONS:
        for col in CAPACITY_TARGETS:
            datasets[(col, h)] = datasets[f"{col}_{h}h"]

    return datasets
