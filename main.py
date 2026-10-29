"""
main.py — Bulut Altyapısı Proaktif İzleme ve Karar Destek Sistemi

Pipeline adımları:
  1. Veri yükleme        : data/raw/ dizininden CSV okuma
  2. Log işleme hattı    : 4 aşamalı log analizi (ayrıştırma → agregasyon → anomali → örüntü)
  3. Özellik müh.        : rolling, lag, delta, zaman ve log özellikleri
  4. Kapasite tahmini    : 7 model × 3 kaynak × 2 ufuk (24h/72h) = 42 regresyon modeli
  5. SHAP analizi        : Açıklanabilirlik ve özellik önem sıralaması
  6. LLM Önerileri       : Metrik + log analizi çıkarımları LLM'e beslenerek proaktif öneriler
"""

import os
import sys
import time
import json
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from config import DATA_DIR, RAW_DIR, RESULTS_DIR

for d in [
    os.path.join(DATA_DIR, "processed", "models"),
    os.path.join(RESULTS_DIR, "figures"),
    os.path.join(RESULTS_DIR, "reports"),
]:
    os.makedirs(d, exist_ok=True)


from src.log_analysis.log_processor        import run_log_pipeline, plot_log_pipeline_results
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
    print("  OpenStack | ML Pipeline | Log Analizi | LLM")
    print("=" * 62)

    # ── Adım 1: Veri Yükleme ────────────────────────────────────────────────
    _step("ADIM 1 / 6  →  Veri Yükleme")
    t0 = time.time()

    metrics_path  = os.path.join(RAW_DIR, "metrics.csv")
    log_path      = os.path.join(RAW_DIR, "log_features.csv")
    raw_logs_path = os.path.join(RAW_DIR, "raw_logs.csv")

    for p in [metrics_path, log_path, raw_logs_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {p}")

    metrics_df = pd.read_csv(metrics_path,  parse_dates=["timestamp"])
    log_df     = pd.read_csv(log_path,      parse_dates=["timestamp"])
    raw_logs   = pd.read_csv(raw_logs_path)

    metrics_df.to_csv(os.path.join(DATA_DIR, "processed", "metrics_loaded.csv"), index=False)

    print(f"  Metrik veri seti    : {len(metrics_df):,} satır  "
          f"({metrics_df['timestamp'].min().date()} → {metrics_df['timestamp'].max().date()})")
    print(f"  Log özellik seti    : {len(log_df):,} satır")
    print(f"  Ham log kayıtları   : {len(raw_logs):,} satır")
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

    n_models = len(cap_res) * 7
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
