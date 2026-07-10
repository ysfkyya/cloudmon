"""
main.py — Bulut Altyapısı Proaktif İzleme ve Karar Destek Sistemi

Pipeline adımları
-----------------
1. Veri üretimi   : 6 aylık sentetik metrik + log verisi
2. Özellik müh.   : rolling, lag, delta, zaman ve log özellikleri
3. Kapasite tahm. : CPU / Bellek / Disk için 24h ve 72h regresyon modelleri
4. Sağlık sınıf.  : Binary risk sınıflandırması (sonraki 4 saatte risk var mı?)
5. SHAP analizi   : Açıklanabilirlik ve özellik önem sıralaması
6. Öneriler       : Sistem yöneticisine operasyonel öneri raporu

Kullanım
--------
    python main.py
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Proje kök dizinini Python yoluna ekle
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from config import DATA_DIR, RESULTS_DIR

# Dizin kontrolü
for d in [
    os.path.join(DATA_DIR, "raw"),
    os.path.join(DATA_DIR, "processed", "models"),
    os.path.join(RESULTS_DIR, "figures"),
    os.path.join(RESULTS_DIR, "reports"),
]:
    os.makedirs(d, exist_ok=True)


# ── Modül importları ──────────────────────────────────────────────────────────

from src.data_generation.metrics_generator import generate_metrics, plot_metrics_overview
from src.data_generation.log_generator     import generate_log_features, generate_raw_logs, plot_log_analysis
from src.preprocessing.feature_engineering import (
    build_feature_matrix, prepare_capacity_dataset, prepare_health_dataset,
)
from src.models.capacity_predictor         import train_capacity_models
from src.models.health_classifier          import train_health_models
from src.explainability.shap_explainer     import explain_capacity, explain_health
from src.recommendations.engine            import generate_recommendations


# ── Yardımcı ─────────────────────────────────────────────────────────────────

def _step(msg: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {msg}")
    print(f"{'─' * 60}")


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s:.1f}s"


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main() -> None:
    t_total = time.time()

    print("\n" + "=" * 60)
    print("  BULUT ALT YAPI PROAKTİF İZLEME & KARAR DESTEK SİSTEMİ")
    print("  OpenStack | Prometheus + OpenSearch | ML Pipeline")
    print("=" * 60)

    # ── Adım 1: Veri Üretimi ────────────────────────────────────────────────
    _step("ADIM 1 / 6  →  Sentetik Veri Üretimi")
    t0 = time.time()

    metrics_df = generate_metrics()
    log_df     = generate_log_features(metrics_df)
    raw_logs   = generate_raw_logs(metrics_df, n_samples=5000)

    # Kaydet
    metrics_df.to_csv(os.path.join(DATA_DIR, "raw", "metrics.csv"), index=False)
    log_df.to_csv(    os.path.join(DATA_DIR, "raw", "log_features.csv"), index=False)
    raw_logs.to_csv(  os.path.join(DATA_DIR, "raw", "raw_logs.csv"), index=False)

    print(f"  Metrik veri seti    : {len(metrics_df):,} satır  "
          f"({metrics_df['timestamp'].min().date()} → {metrics_df['timestamp'].max().date()})")
    print(f"  Log özellik seti    : {len(log_df):,} satır")
    print(f"  Ham log örnekleri   : {len(raw_logs):,} kayıt")

    plot_metrics_overview(metrics_df)
    plot_log_analysis(log_df)
    print(f"  [{_elapsed(t0)}]")

    # ── Adım 2: Özellik Mühendisliği ────────────────────────────────────────
    _step("ADIM 2 / 6  →  Özellik Mühendisliği")
    t0 = time.time()

    feature_df = build_feature_matrix(metrics_df, log_df)
    feature_df.to_csv(os.path.join(DATA_DIR, "processed", "features.csv"))

    print(f"  Özellik matrisi     : {feature_df.shape[0]:,} satır × {feature_df.shape[1]} sütun")

    cap_datasets  = prepare_capacity_dataset(feature_df)
    health_dataset = prepare_health_dataset(feature_df)

    # Eğitim / test boyutları
    first_key = list(cap_datasets.keys())[0]
    n_train   = len(cap_datasets[first_key]["X_train"])
    n_test    = len(cap_datasets[first_key]["X_test"])
    print(f"  Kapasite eğitim     : {n_train:,}  |  test: {n_test:,}")
    print(f"  Sağlık eğitim       : {len(health_dataset['X_train']):,}  "
          f"|  test: {len(health_dataset['X_test']):,}")
    print(f"  [{_elapsed(t0)}]")

    # ── Adım 3: Kapasite Tahmin Modelleri ───────────────────────────────────
    _step("ADIM 3 / 6  →  Kapasite Tahmin Modelleri (Regresyon)")
    t0 = time.time()

    capacity_results = train_capacity_models(cap_datasets)
    print(f"  [{_elapsed(t0)}]")

    # ── Adım 4: Sağlık Sınıflandırması ──────────────────────────────────────
    _step("ADIM 4 / 6  →  Sağlık Riski Sınıflandırması (Binary)")
    t0 = time.time()

    health_results = train_health_models(health_dataset)
    print(f"  [{_elapsed(t0)}]")

    # ── Adım 5: SHAP Açıklanabilirlik Analizi ───────────────────────────────
    _step("ADIM 5 / 6  →  SHAP Açıklanabilirlik Analizi")
    t0 = time.time()

    shap_capacity = explain_capacity(capacity_results, cap_datasets)
    shap_health   = explain_health(health_results, health_dataset)
    print(f"  [{_elapsed(t0)}]")

    # ── Adım 6: Operasyonel Öneri Üretimi ───────────────────────────────────
    _step("ADIM 6 / 6  →  Operasyonel Öneri Üretimi")
    t0 = time.time()

    recommendations = generate_recommendations(
        feature_df,
        capacity_results,
        health_results,
        shap_capacity,
        shap_health,
    )
    print(f"  [{_elapsed(t0)}]")

    # ── Özet Raporu ──────────────────────────────────────────────────────────
    _print_summary(capacity_results, health_results, recommendations, t_total)


def _print_summary(cap_res, health_res, recs, t_total):
    print("\n" + "=" * 60)
    print("  PIPELINE TAMAMLANDI")
    print(f"  Toplam süre: {time.time() - t_total:.1f}s")
    print("=" * 60)

    # En iyi kapasite modeli
    print("\n  Kapasite Tahmin Sonuçları (En İyi Model — RMSE Bazlı):")
    from config import CAPACITY_TARGETS, FORECAST_HORIZONS
    _RES_LABELS = {
        "cpu_usage_percent":    "CPU",
        "memory_usage_percent": "Bellek",
        "disk_usage_percent":   "Disk",
    }
    for h in FORECAST_HORIZONS:
        print(f"\n    Ufuk: {h} saat")
        for col in CAPACITY_TARGETS:
            key    = f"{col}_{h}h"
            models = cap_res[key]["models"]
            best_m = min(models.items(), key=lambda x: x[1]["RMSE"])[0]
            m      = models[best_m]
            print(f"    {_RES_LABELS.get(col,col):8s} | {best_m:20s} | "
                  f"MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R²={m['R2']:.4f}")

    # Sağlık sınıflandırma özeti
    print("\n  Sağlık Riski Sınıflandırma Sonuçları:")
    best_h = max(health_res["models"].items(), key=lambda x: x[1]["F1"])[0]
    for mname, mv in health_res["models"].items():
        marker = " ◄ en iyi" if mname == best_h else ""
        print(f"    {mname:22s} | Acc={mv['Accuracy']:.3f}  "
              f"Prec={mv['Precision']:.3f}  Rec={mv['Recall']:.3f}  "
              f"F1={mv['F1']:.3f}  AUC={mv['AUC-ROC']:.3f}{marker}")

    # Öneri özeti
    from config import RESULTS_DIR
    n_crit = sum(1 for r in recs if r["severity"] == "critical")
    n_warn = sum(1 for r in recs if r["severity"] == "warning")
    print(f"\n  Üretilen Öneriler: {len(recs)} toplam  "
          f"({n_crit} kritik, {n_warn} uyarı)")

    print("\n  Sonuçlar:")
    print(f"    Grafikler  → {RESULTS_DIR}/figures/")
    print(f"    Raporlar   → {RESULTS_DIR}/reports/")
    print(f"    Modeller   → data/processed/models/")
    print()


if __name__ == "__main__":
    main()
