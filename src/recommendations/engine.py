"""
Proaktif öneri motoru — kapasite, log anomalileri ve metrik trend uyarıları.

Öneri kategorileri:
  KAPASİTE : kaynak eşiğine tahmini ulaşma tarihi
  LOG      : log pipeline anomali ve örüntü bulgularına dayalı
  TREND    : haftalık log ve metrik bozulmaları
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import THRESHOLDS, RESULTS_DIR

_CAPACITY_RULES = [
    {
        "resource":      "vcpu_usage_percent",
        "label":         "vCPU Tahsis Oranı",
        "unit":          "%",
        "warn_days":     90,
        "critical_days": 30,
        "suggestions": [
            "Yeni compute node alımı için satın alma sürecini başlatın (temin süresi ~60 gün).",
            "vCPU overcommit oranını gözden geçirin; kullanılmayan instance'ları sonlandırın.",
            "Büyük flavour talepleri için Nova scheduler ağırlıklarını kısıtlayın.",
            "Yoğun kullanım dönemlerinde kullanıcı vCPU kotalarını gözden geçirin.",
        ],
    },
    {
        "resource":      "memory_usage_percent",
        "label":         "Bellek Kullanımı",
        "unit":          "%",
        "warn_days":     60,
        "critical_days": 21,
        "suggestions": [
            "Bellek sızıntısı şüpheli VM'leri ve servisleri inceleyin.",
            "Nova memory_mb_used değerini Keystone proje kotalarıyla karşılaştırın.",
            "Büyük bellek tahsisi yapan VM'leri daha az yoğun node'lara taşıyın.",
        ],
    },
    {
        "resource":      "disk_usage_percent",
        "label":         "Disk Kullanımı",
        "unit":          "%",
        "warn_days":     90,
        "critical_days": 30,
        "suggestions": [
            "Eski Nova instance snapshot'larını ve kullanılmayan Cinder volume'larını temizleyin.",
            "Log rotasyon politikasını gözden geçirin; OpenSearch index'lerini arşivleyin.",
            "Ek Cinder storage backend'i veya NFS mount'u ekleyin.",
        ],
    },
]

_LOG_SUGGESTIONS = {
    "timeout_count":               "Nova API ve Keystone timeout değerlerini artırın; bağlantı havuzu boyutunu büyütün.",
    "db_connection_count":         "MySQL/MariaDB max_connections parametresini artırın; bağlantı havuzu sağlığını izleyin.",
    "resource_exhausted_count":    "Nova scheduler ağırlıklarını ve filtrelerini güncelleyin; compute kaynaklarını dengeleyin.",
    "quota_exceeded_count":        "Keystone proje kotalarını gözden geçirin; yoğun kullanım dönemlerinde kota aşımı yaşanıyor olabilir.",
    "auth_failure_count":          "Keystone token süresi ve revocation listesi yapılandırmasını gözden geçirin.",
    "network_error_count":         "Neutron OVS agent durumunu ve VLAN yapılandırmalarını kontrol edin.",
    "scheduler_error_count":       "Nova compute node'larının servis durumunu ve kaynak raporlamasını doğrulayın.",
    "io_error_count":              "Cinder backend sağlığını ve iSCSI bağlantılarını denetleyin.",
    "slow_response_count":         "API yanıt sürelerini izleyin; connection pool ve thread pool boyutlarını artırmayı değerlendirin.",
    "memory_pressure_count":       "Compute node'larında bellek kullanımını izleyin; OOM killer loglarını kontrol edin.",
    "connection_pool_warn_count":  "DB bağlantı havuzu kapasitesini artırın; oslo.db max_pool_size değerini ayarlayın.",
    "scheduler_degradation_count": "Nova scheduler yanıt sürelerini profilleyin; placement API performansını kontrol edin.",
    "quota_pressure_count":        "Kota limitlerini yoğun kullanım dönemlerine göre önceden gözden geçirin.",
    "disk_io_degradation_count":   "Cinder backend'in I/O latency değerlerini izleyin; storage array queue depth'ini kontrol edin.",
    "cascade_indicator_count":     "Birden fazla servis aynı anda bozuluyor — sistem genelinde bir köklü neden araştırın.",
}

_SEVERITY_LABELS = {"critical": "KRİTİK", "warning": "UYARI", "info": "BİLGİ"}


def generate_recommendations(
    feature_df: pd.DataFrame,
    capacity_results: dict,
    shap_capacity: dict,
    log_insights: dict = None,
) -> list:
    recs   = []
    ts_now = str(feature_df.index[-1])

    recs += _capacity_recommendations(feature_df, capacity_results, shap_capacity, ts_now)
    recs += _log_anomaly_recommendations(log_insights or {}, ts_now)
    recs += _trend_recommendations(feature_df, ts_now)

    order = {"critical": 0, "warning": 1, "info": 2}
    recs  = sorted(recs, key=lambda x: order.get(x["severity"], 9))

    _save_recommendations(recs, feature_df, capacity_results)
    _print_recommendations(recs)

    return recs


# ── Kapasite önerileri ────────────────────────────────────────────────────────

def _capacity_recommendations(feature_df, cap_results, shap_cap, ts_now):
    from src.models.capacity_predictor import predict_days_to_threshold
    recs = []

    for rule in _CAPACITY_RULES:
        col  = rule["resource"]
        info = predict_days_to_threshold(feature_df, cap_results, col)
        days = info.get("days_to_threshold")
        cur  = info.get("current_value", 0)
        thr  = info.get("threshold", 90)
        est  = info.get("estimated_date")
        stable = info.get("stable", False)

        if stable or days is None:
            continue

        severity = (
            "critical" if days <= rule["critical_days"] else
            "warning"  if days <= rule["warn_days"]     else
            None
        )
        if severity is None:
            continue

        shap_key  = next((k for k in shap_cap if col in k and "24h" in k), None)
        top_feat  = ""
        if shap_key and shap_cap.get(shap_key, {}).get("top_features"):
            top_feat = shap_cap[shap_key]["top_features"][0]["feature"]

        daily_trend = info.get("daily_trend", 0)
        months      = info.get("months_to_threshold")
        time_str    = f"~{months:.1f} ay / ~{int(days)} gün" if months else f"~{int(days)} gün"
        detail = (
            f"Mevcut {rule['label']}: {cur:.1f}{rule['unit']} (eşik: {thr}{rule['unit']}). "
            f"30 günlük medyan artış: +{daily_trend:.3f}{rule['unit']}/gün. "
            f"Tahmini eşik tarihi: {est} ({time_str})."
        )
        if top_feat:
            detail += f" En etkili SHAP özelliği: '{top_feat}'."

        recs.append({
            "severity":         severity,
            "category":         "KAPASİTE",
            "title":            f"{rule['label']} — Tahmini Eşik Tarihi: {est}",
            "detail":           detail,
            "actions":          rule["suggestions"],
            "estimated_date":   est,
            "days_remaining":   int(days),
            "months_remaining": round(months, 1) if months else None,
            "timestamp":        ts_now,
        })

    return recs


# ── Log anomali önerileri ─────────────────────────────────────────────────────

def _log_anomaly_recommendations(log_insights: dict, ts_now: str) -> list:
    """Log pipeline çıktısından aksiyon gerektiren durumları öneri olarak üretir."""
    recs = []

    # Anomali tabanlı öneriler
    for anom in log_insights.get("anomalies_last_24h", [])[:5]:
        log_type = anom.get("log_type", "")
        zscore   = anom.get("max_zscore", 0)
        label    = anom.get("label", log_type)
        ratio    = anom.get("ratio", 1)

        severity = "critical" if zscore >= 4.0 or ratio >= 5 else "warning"
        suggestion = _LOG_SUGGESTIONS.get(log_type, "İlgili servis loglarını inceleyin.")

        recs.append({
            "severity":  severity,
            "category":  "LOG ANOMALİSİ",
            "title":     f"{label} — Anormal Artış (z={zscore:.1f}, {ratio:.1f}x normal)",
            "detail":    (
                f"Son 24 saatte '{label}' log sayısı normalin {ratio:.1f} katına çıktı "
                f"(z-skor: {zscore:.1f}). Bu erken uyarı sinyali yaklaşan bir soruna işaret edebilir."
            ),
            "actions": [
                suggestion,
                "Son 6 saatin servis loglarını ayrıntılı inceleyin.",
                "İlgili kapasite metriklerini (vCPU, bellek) beraber değerlendirin.",
            ],
            "timestamp": ts_now,
        })

    # Cascade uyarısı
    cascade = log_insights.get("cascade_events_recent", [])
    if cascade:
        latest = cascade[-1]
        recs.append({
            "severity":  "critical",
            "category":  "CASCADE RİSKİ",
            "title":     f"Çok Servisli Bozulma Tespit Edildi ({latest['start']})",
            "detail":    (
                f"Birden fazla OpenStack servisi aynı anda anomali gösterdi "
                f"({latest['duration_h']:.1f} saat boyunca). "
                f"Bu örüntü sistemik bir sorunun habercisi olabilir."
            ),
            "actions": [
                "Nova, Neutron ve Keystone servis durumlarını eş zamanlı kontrol edin.",
                "OpenStack health-check komutunu çalıştırın: openstack service list.",
                "RabbitMQ kuyruk derinliğini ve DB bağlantı sayısını inceleyin.",
                "Tüm compute ve network node'larında agent durumunu doğrulayın.",
            ],
            "timestamp": ts_now,
        })

    # Ön-arıza örüntüsü uyarısı
    patterns = log_insights.get("prefailure_patterns", [])
    active_patterns = [
        p for p in patterns
        if p.get("is_warning_type") and p.get("confidence", 0) >= 0.5
    ]
    if active_patterns:
        top = active_patterns[0]
        recs.append({
            "severity":  "warning",
            "category":  "ÖN-ARIZA ÖRÜNTÜSÜ",
            "title":     f"Tarihsel Arıza Sinyali: {top['label']}",
            "detail":    (
                f"'{top['label']}' log tipi, geçmiş {top['observed_times']} incidentte "
                f"arızadan ortalama {top['lead_time_hours']:.0f} saat önce "
                f"+{top['avg_increase_pct']:.0f}% artış gösterdi "
                f"(güven: {top['confidence']:.0%}). Mevcut veriler bu örüntüyle örtüşüyor."
            ),
            "actions": [
                f"'{top['label']}' tipindeki logların son trendini inceleyip müdahale planı hazırlayın.",
                "Benzer geçmiş incident'lerin kök neden analizini gözden geçirin.",
                "İlgili OpenStack servislerinde önleyici bakım (restart, config review) yapın.",
            ],
            "timestamp": ts_now,
        })

    return recs


# ── Trend önerileri ───────────────────────────────────────────────────────────

def _trend_recommendations(feature_df, ts_now):
    recs   = []
    window = 168

    trend_checks = [
        ("error_rate_percent",           "Hata Oranı",               "%",    0.5,   1.5),
        ("api_latency_ms",               "API Gecikmesi",            "ms",  50.0, 150.0),
        ("resource_exhausted_count",     "Kaynak Tükenmesi Hatası",  "adet", 20.0,  60.0),
        ("quota_exceeded_count",         "Kota Aşımı Hatası",        "adet",  5.0,  20.0),
        ("timeout_count",                "Timeout Hatası",           "adet", 30.0, 100.0),
        ("slow_response_count",          "Yavaş Yanıt Uyarısı",      "adet", 15.0,  50.0),
        ("connection_pool_warn_count",   "Bağlantı Havuzu Uyarısı",  "adet", 10.0,  40.0),
        ("scheduler_degradation_count",  "Zamanlayıcı Bozulma Uyarısı", "adet", 8.0, 30.0),
    ]

    for col, label, unit, warn_rise, crit_rise in trend_checks:
        if col not in feature_df.columns or len(feature_df) < window * 2:
            continue
        prev_week = feature_df[col].iloc[-(window * 2): -window].mean()
        last_week = feature_df[col].iloc[-window:].mean()
        rise      = last_week - prev_week

        severity = (
            "critical" if rise >= crit_rise else
            "warning"  if rise >= warn_rise  else
            None
        )
        if severity is None:
            continue

        pct_change = (rise / max(prev_week, 0.001)) * 100
        recs.append({
            "severity":  severity,
            "category":  "TREND",
            "title":     f"{label} Son Haftada Belirgin Artış (+%{pct_change:.0f})",
            "detail":    (
                f"Bir önceki haftaya göre ortalama {label} {rise:+.2f} {unit} arttı "
                f"({prev_week:.2f} → {last_week:.2f} {unit}, %{pct_change:.0f} artış)."
            ),
            "actions": [
                f"{label} değişimini inceleyin; ilgili servislerin loglarını kontrol edin.",
                "Değişim yoğun kullanım dönemine denk geliyorsa kullanıcı kota ayarlarını gözden geçirin.",
                "Değişim keskin ise yakın zamanda yapılan yapılandırma değişikliklerini araştırın.",
            ],
            "timestamp": ts_now,
        })

    return recs


# ── Hata özeti (LLM için) ─────────────────────────────────────────────────────

def build_error_summary(feature_df: pd.DataFrame) -> dict:
    if len(feature_df) < 336:
        return {}
    prev_week = feature_df.iloc[-336:-168]
    last_week = feature_df.iloc[-168:]

    error_cols = {
        "timeout_count":              "Timeout",
        "db_connection_count":        "DB Bağlantı",
        "resource_exhausted_count":   "Kaynak Tükenmesi",
        "quota_exceeded_count":       "Kota Aşımı",
        "auth_failure_count":         "Kimlik Doğrulama",
        "network_error_count":        "Ağ Hatası",
        "scheduler_error_count":      "Zamanlayıcı",
        "slow_response_count":        "Yavaş Yanıt",
        "memory_pressure_count":      "Bellek Baskısı",
        "connection_pool_warn_count": "Bağlantı Havuzu",
        "scheduler_degradation_count":"Zamanlayıcı Bozulma",
        "cascade_indicator_count":    "Cascade Göstergesi",
        "critical_count":             "CRITICAL Log",
    }

    summary = {}
    for col, label in error_cols.items():
        if col not in feature_df.columns:
            continue
        prev_sum = int(prev_week[col].sum())
        last_sum = int(last_week[col].sum())
        change   = ((last_sum - prev_sum) / max(prev_sum, 1)) * 100
        summary[label] = {
            "last_7d":    last_sum,
            "prev_7d":    prev_sum,
            "change_pct": round(change, 1),
            "high":       change >= 50 or last_sum > 500,
        }
    return summary


# ── Çıktı fonksiyonları ───────────────────────────────────────────────────────

def _save_recommendations(recs, feature_df, cap_results):
    from src.models.capacity_predictor import predict_days_to_threshold

    cap_summary = {}
    for col in ["vcpu_usage_percent", "memory_usage_percent", "disk_usage_percent"]:
        cap_summary[col] = predict_days_to_threshold(feature_df, cap_results, col)

    err_summary = build_error_summary(feature_df)

    out_json = os.path.join(RESULTS_DIR, "reports", "recommendations.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "recommendations": recs,
            "capacity_summary": cap_summary,
            "error_summary":    err_summary,
        }, f, ensure_ascii=False, indent=2)

    out_txt = os.path.join(RESULTS_DIR, "reports", "recommendations.txt")
    _write_txt_report(out_txt, recs, cap_summary, err_summary)

    print(f"  → {out_json}")
    print(f"  → {out_txt}")


def _write_txt_report(path, recs, cap_summary, err_summary):
    labels = {
        "vcpu_usage_percent":   "vCPU  ",
        "memory_usage_percent": "Bellek",
        "disk_usage_percent":   "Disk  ",
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  BULUT ALT YAPI PROAKTİF İZLEME — OPERASYONEL RAPOR\n")
        f.write(f"  Oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("── KAPASİTE DURUMU ──────────────────────────────────────────────\n")
        for col, info in cap_summary.items():
            lbl  = labels.get(col, col)
            cur  = info.get("current_value", "?")
            thr  = info.get("threshold", "?")
            est  = info.get("estimated_date")
            days   = info.get("days_to_threshold")
            months = info.get("months_to_threshold")
            stb    = info.get("stable", False)
            trend  = info.get("daily_trend", 0)
            if stb or days is None:
                status = "Kararlı trend (>180 gün)"
            else:
                time_str = f"~{months:.1f} ay / ~{int(days)} gün" if months else f"~{int(days)} gün"
                status = f"Tahmini eşik: {est} ({time_str})"
            f.write(f"  {lbl}: %{cur:.1f} → Eşik %{thr} | +{trend:.3f}%/gün | {status}\n")

        f.write("\n── 7 GÜNLÜK LOG ÖZETİ ──────────────────────────────────────────\n")
        for label, data in err_summary.items():
            flag = " !!!" if data.get("high") else ""
            f.write(f"  {label:28s}: {data['last_7d']:6,} kez  "
                    f"({data['change_pct']:+.0f}% değişim){flag}\n")

        f.write("\n── ÖNERİLER ──────────────────────────────────────────────────────\n")
        for i, rec in enumerate(recs, 1):
            sev = _SEVERITY_LABELS.get(rec["severity"], rec["severity"].upper())
            f.write(f"\n[{i}] [{sev}] {rec['category']} — {rec['title']}\n")
            f.write(f"     {rec['detail']}\n")
            f.write("     Aksiyonlar:\n")
            for j, act in enumerate(rec.get("actions", []), 1):
                f.write(f"       {j}. {act}\n")


def _print_recommendations(recs):
    print("\n" + "=" * 70)
    print("  OPERASYONEL ÖNERİLER")
    print("=" * 70)
    icons = {"critical": "[KRİTİK]", "warning": "[UYARI]", "info": "[BİLGİ]"}
    for i, rec in enumerate(recs, 1):
        sev  = _SEVERITY_LABELS.get(rec["severity"], rec["severity"].upper())
        icon = icons.get(rec["severity"], "[BİLGİ]")
        print(f"\n{icon} [{i}] {rec['category']} — {rec['title']}")
        print(f"    {rec['detail']}")
        print("    Aksiyonlar:")
        for j, act in enumerate(rec.get("actions", []), 1):
            print(f"      {j}. {act}")
    print("=" * 70 + "\n")
