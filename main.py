"""
main.py — Bulut Altyapısı Proaktif İzleme ve Karar Destek Sistemi

Pipeline adımları:
  1. Veri üretimi      : 18 aylık sentetik metrik (vCPU 30→1598) + zengin log verisi
  2. Log işleme hattı  : 4 aşamalı log analizi (ayrıştırma → agregasyon → anomali → örüntü)
  3. Özellik müh.      : rolling, lag, delta, zaman ve log özellikleri
  4. Kapasite tahmini  : 7 model × 3 kaynak × 3 ufuk (24h/72h/168h) = 63 regresyon modeli
  5. SHAP analizi      : Açıklanabilirlik ve özellik önem sıralaması
  6. LLM Önerileri     : Metrik + log analizi çıkarımları LLM'e beslenerek proaktif öneriler

vCPU senaryosu: Toplam 2000 vCPU | Ocak 2025: 30 → Temmuz 2026: 1598
                Mevcut trend: ~8 ay sonra tam dolum (Mart 2027)
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from config import DATA_DIR, SYNTHETIC_DIR, RESULTS_DIR

for d in [
    SYNTHETIC_DIR,
    os.path.join(DATA_DIR, "processed", "models"),
    os.path.join(RESULTS_DIR, "figures"),
    os.path.join(RESULTS_DIR, "reports"),
]:
    os.makedirs(d, exist_ok=True)


from src.data_generation.metrics_generator import generate_metrics, plot_metrics_overview
from src.data_generation.log_generator     import generate_log_features, generate_raw_logs, plot_log_analysis
from src.data_generation.log_processor     import run_log_pipeline, plot_log_pipeline_results
from src.preprocessing.feature_engineering import build_feature_matrix, prepare_capacity_dataset
from src.models.capacity_predictor         import train_capacity_models
from src.explainability.shap_explainer     import explain_capacity
from src.recommendations.engine            import generate_recommendations
from src.recommendations.llm_advisor       import LLMAdvisor


def _step(msg: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {msg}")
    print(f"{'─' * 62}")


def _elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


def main() -> None:
    t_total = time.time()

    print("\n" + "=" * 62)
    print("  BULUT ALTYAPI PROAKTİF İZLEME & KARAR DESTEK SİSTEMİ")
    print("  OpenStack | vCPU: 30 → 1598/2000 | Log Pipeline + LLM")
    print("=" * 62)

    # ── Adım 1: Veri Üretimi ────────────────────────────────────────────────
    _step("ADIM 1 / 6  →  Sentetik Veri Üretimi")
    t0 = time.time()

    metrics_df = generate_metrics()
    log_df     = generate_log_features(metrics_df)
    raw_logs   = generate_raw_logs(metrics_df, n_samples=8000)

    metrics_df.to_csv(os.path.join(SYNTHETIC_DIR, "metrics.csv"),      index=False)
    log_df.to_csv(    os.path.join(SYNTHETIC_DIR, "log_features.csv"), index=False)
    raw_logs.to_csv(  os.path.join(SYNTHETIC_DIR, "raw_logs.csv"),     index=False)

    vcpu_start = int(metrics_df["vcpu_used"].iloc[0])
    vcpu_end   = int(metrics_df["vcpu_used"].iloc[-1])

    mem_start_gb = int(metrics_df["memory_used_gb"].iloc[0])
    mem_end_gb   = int(metrics_df["memory_used_gb"].iloc[-1])
    mem_total_gb = int(metrics_df["memory_total_gb"].iloc[0])
    disk_start   = round(float(metrics_df["disk_used_tb"].iloc[0]), 0)
    disk_end     = round(float(metrics_df["disk_used_tb"].iloc[-1]), 0)
    disk_total   = int(metrics_df["disk_total_tb"].iloc[0])

    print(f"  Metrik veri seti    : {len(metrics_df):,} satır  "
          f"({metrics_df['timestamp'].min().date()} → {metrics_df['timestamp'].max().date()})")
    print(f"  vCPU kullanımı      : {vcpu_start} → {vcpu_end} / "
          f"{int(metrics_df['vcpu_total'].iloc[0])} "
          f"(%{metrics_df['vcpu_usage_percent'].iloc[-1]:.1f})")
    print(f"  Bellek kullanımı    : {mem_start_gb} → {mem_end_gb} / {mem_total_gb} GB "
          f"(%{metrics_df['memory_usage_percent'].iloc[-1]:.1f})")
    print(f"  Disk kullanımı      : {disk_start:.0f} → {disk_end:.0f} / {disk_total} TB "
          f"(%{metrics_df['disk_usage_percent'].iloc[-1]:.1f})")
    print(f"  Log özellik seti    : {len(log_df):,} satır")
    print(f"  Ham log kayıtları   : {len(raw_logs):,} satır  "
          f"(WARNING: {(raw_logs['level']=='WARNING').sum():,}  "
          f"ERROR: {(raw_logs['level']=='ERROR').sum():,}  "
          f"CRITICAL: {(raw_logs['level']=='CRITICAL').sum():,})")

    plot_metrics_overview(metrics_df)
    plot_log_analysis(log_df)
    print(f"  [{_elapsed(t0)}]")

    # ── Adım 2: Log İşleme Hattı ────────────────────────────────────────────
    _step("ADIM 2 / 6  →  Log İşleme Hattı (4 Aşamalı Analiz)")
    t0 = time.time()

    log_insights = run_log_pipeline(log_df)
    plot_log_pipeline_results(log_insights)

    anom_count  = len(log_insights.get("anomalies_last_24h", []))
    patt_count  = len(log_insights.get("prefailure_patterns", []))
    trend_count = sum(1 for t in log_insights.get("log_trends_7d", [])
                      if t["direction"] == "increasing")

    print(f"  Tespit edilen anomali  : {anom_count} (son 24 saat)")
    print(f"  Ön-arıza örüntüsü      : {patt_count} çıkarıldı")
    print(f"  Artan log tipi (7 gün) : {trend_count}")
    print(f"  Erken uyarı log (24s)  : {log_insights.get('total_warnings_24h', 0):,}")
    print(f"  Hata log (24s)         : {log_insights.get('total_errors_24h', 0):,}")
    print(f"  [{_elapsed(t0)}]")

    # ── Adım 3: Özellik Mühendisliği ────────────────────────────────────────
    _step("ADIM 3 / 6  →  Özellik Mühendisliği")
    t0 = time.time()

    feature_df = build_feature_matrix(metrics_df, log_df)
    feature_df.to_csv(os.path.join(DATA_DIR, "processed", "features.csv"))

    print(f"  Özellik matrisi     : {feature_df.shape[0]:,} satır × {feature_df.shape[1]} sütun")

    cap_datasets = prepare_capacity_dataset(feature_df)
    first_key    = list(cap_datasets.keys())[0]
    n_train      = len(cap_datasets[first_key]["X_train"])
    n_test       = len(cap_datasets[first_key]["X_test"])
    print(f"  Kapasite eğitim     : {n_train:,}  |  test: {n_test:,}")
    print(f"  [{_elapsed(t0)}]")

    # ── Adım 4: Kapasite Tahmin Modelleri ───────────────────────────────────
    _step("ADIM 4 / 6  →  Kapasite Tahmin Modelleri (Regresyon)")
    t0 = time.time()

    capacity_results = train_capacity_models(cap_datasets)
    print(f"  [{_elapsed(t0)}]")

    # ── Adım 5: SHAP Açıklanabilirlik Analizi ───────────────────────────────
    _step("ADIM 5 / 6  →  SHAP Açıklanabilirlik Analizi")
    t0 = time.time()

    shap_capacity = explain_capacity(capacity_results, cap_datasets)
    print(f"  [{_elapsed(t0)}]")

    # ── Adım 6: Proaktif Öneri Üretimi ──────────────────────────────────────
    _step("ADIM 6 / 6  →  Proaktif Öneri Üretimi (Kural + LLM)")
    t0 = time.time()

    recommendations = generate_recommendations(
        feature_df,
        capacity_results,
        shap_capacity,
        log_insights=log_insights,
    )

    # LLM çağrısı (API anahtarı varsa)
    advisor = LLMAdvisor()
    if advisor.available_providers():
        print(f"\n  LLM Sağlayıcıları: {', '.join(advisor.available_providers())}")
        llm_result = advisor.get_recommendations(
            feature_df,
            shap_capacity,
            capacity_results=capacity_results,
            recommendations=recommendations,
            log_insights=log_insights,
        )
        _save_llm_results(llm_result)
    else:
        print("  LLM: API anahtarı bulunamadı → atlandı "
              "(ANTHROPIC_API_KEY veya OPENAI_API_KEY ayarlayın)")
        llm_result = None

    print(f"  [{_elapsed(t0)}]")

    _print_summary(capacity_results, recommendations, log_insights, t_total)


def _save_llm_results(llm_result: dict) -> None:
    out = os.path.join(RESULTS_DIR, "reports", "llm_recommendations.json")
    safe = {
        "prompt":    llm_result["prompt"],
        "responses": {
            p: {"status": r["status"], "text": r["text"], "error": r["error"]}
            for p, r in llm_result["responses"].items()
        },
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False, indent=2)
    print(f"  → {out}")

    out_txt = os.path.join(RESULTS_DIR, "reports", "llm_recommendations.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("LLM PROAKTİF ÖNERİ RAPORU\n")
        f.write("=" * 62 + "\n\n")
        for provider, resp in llm_result["responses"].items():
            f.write(f"[{provider.upper()}] — {resp['status'].upper()}\n")
            f.write("-" * 40 + "\n")
            if resp["status"] == "ok":
                f.write(resp["text"] + "\n\n")
            else:
                f.write(f"Hata: {resp['error']}\n\n")
    print(f"  → {out_txt}")

    for provider, resp in llm_result["responses"].items():
        if resp["status"] == "ok":
            print(f"\n  ── {provider.upper()} YANITI (özet ilk 400 karakter) ──")
            print(f"  {resp['text'][:400].replace(chr(10), chr(10)+'  ')}...")


def _print_summary(cap_res, recs, log_insights, t_total):
    print("\n" + "=" * 62)
    print("  PİPELİNE TAMAMLANDI")
    print(f"  Toplam süre: {time.time() - t_total:.1f}s")
    print("=" * 62)

    from config import CAPACITY_TARGETS, FORECAST_HORIZONS
    _RES_LABELS = {
        "vcpu_usage_percent":   "vCPU  ",
        "memory_usage_percent": "Bellek",
        "disk_usage_percent":   "Disk  ",
    }

    n_models = len(cap_res) * 7  # 9 hedef × 7 model
    print(f"\n  Kapasite Tahmin Sonuçları (En İyi Model — RMSE Bazlı)  [{n_models} model eğitildi]:")
    for h in FORECAST_HORIZONS:
        print(f"\n    Ufuk: {h} saat")
        for col in CAPACITY_TARGETS:
            key    = f"{col}_{h}h"
            if key not in cap_res:
                continue
            models = cap_res[key]["models"]
            best_m = min(models.items(), key=lambda x: x[1]["RMSE"])[0]
            m      = models[best_m]
            print(f"    {_RES_LABELS.get(col, col):8s} | {best_m:20s} | "
                  f"MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R²={m['R2']:.4f}")

    print("\n  Log Pipeline Özeti:")
    print(f"    Anomali (24s)       : {len(log_insights.get('anomalies_last_24h', []))}")
    print(f"    Ön-arıza örüntüsü   : {len(log_insights.get('prefailure_patterns', []))}")
    print(f"    Cascade olayı       : {len(log_insights.get('cascade_events_recent', []))}")

    svc_issues = [s for s, h in log_insights.get("service_health", {}).items()
                  if h not in ("healthy",)]
    if svc_issues:
        print(f"    Sorunlu servisler   : {', '.join(svc_issues)}")

    n_crit = sum(1 for r in recs if r["severity"] == "critical")
    n_warn = sum(1 for r in recs if r["severity"] == "warning")
    print(f"\n  Üretilen Öneriler: {len(recs)} toplam ({n_crit} kritik, {n_warn} uyarı)")

    print("\n  Çıktılar:")
    print(f"    Grafikler  → {RESULTS_DIR}/figures/")
    print(f"    Raporlar   → {RESULTS_DIR}/reports/")
    print(f"    Modeller   → data/processed/models/")
    print()


if __name__ == "__main__":
    main()
