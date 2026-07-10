"""
Proaktif öneri motoru.

Model tahminlerini ve SHAP özellik önemlerini yorumlayarak
sistem yöneticisine uygulanabilir operasyonel öneriler sunar.

Öneri kategorileri
------------------
  KAPASITE : kaynakların eşiğe yaklaşması
  SAĞLIK   : risk sınıflandırıcısının yüksek olasılık tespiti
  TREND    : metrik trendlerinde belirgin bozulma
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import THRESHOLDS, RISK_HORIZON, RESULTS_DIR


# ── Eşik tetikleyiciler ve öneri şablonları ───────────────────────────────────

_CAPACITY_RULES = [
    {
        "resource":     "cpu_usage_percent",
        "label":        "CPU Kullanımı",
        "unit":         "%",
        "warn_days":    14,
        "critical_days": 7,
        "suggestions":  [
            "Compute node'larında yük dengelemesi (live migration) uygulayın.",
            "VM flavour sınırlamalarını gözden geçirin; CPU yoğun iş yüklerini izole edin.",
            "Ölçek genişletme (scale-out) için ek compute node alımını planlayın.",
        ],
    },
    {
        "resource":     "memory_usage_percent",
        "label":        "Bellek Kullanımı",
        "unit":         "%",
        "warn_days":    14,
        "critical_days": 7,
        "suggestions":  [
            "Bellek sızıntısı (memory leak) şüpheli VM'leri ve servislerini inceleyin.",
            "Nova memory_mb_used değerini Keystone proje kotalarıyla karşılaştırın.",
            "Büyük bellek tahsisi yapan VM'leri daha az yoğun node'lara taşıyın.",
        ],
    },
    {
        "resource":     "disk_usage_percent",
        "label":        "Disk Kullanımı",
        "unit":         "%",
        "warn_days":    21,
        "critical_days": 10,
        "suggestions":  [
            "Eski Nova instance snapshot'larını ve kullanılmayan Cinder volume'larını temizleyin.",
            "Log rotasyon politikasını gözden geçirin; OpenSearch index'lerini arşivleyin.",
            "Ek Cinder storage backend'i veya NFS mount'u ekleyin.",
        ],
    },
]

_HEALTH_SUGGESTIONS = {
    "timeout":              "Nova API ve Keystone timeout değerlerini artırın; "
                            "bağlantı havuzu (connection pool) boyutunu büyütün.",
    "db_connection":        "MySQL/MariaDB max_connections parametresini artırın; "
                            "bağlantı havuzu sağlığını izleyin.",
    "resource_exhausted":   "Yük dengesizliğini gidermek için Nova scheduler "
                            "ağırlıklarını ve filtrelerini ayarlayın.",
    "auth_failure":         "Keystone token süresi ve revocation listesi yapılandırmasını "
                            "gözden geçirin.",
    "network_error":        "Neutron OVS agent durumunu ve VLAN yapılandırmalarını kontrol edin.",
    "scheduler_error":      "Nova compute node'larının servis durumunu ve "
                            "kaynak raporlamasını doğrulayın.",
    "io_error":             "Cinder backend sağlığını ve iSCSI bağlantılarını denetleyin.",
}

_SEVERITY_LABELS = {
    "critical": "KRITIK",
    "warning":  "UYARI",
    "info":     "BİLGİ",
}


# ── Ana fonksiyonlar ──────────────────────────────────────────────────────────

def generate_recommendations(
    feature_df: pd.DataFrame,
    capacity_results: dict,
    health_results: dict,
    shap_capacity: dict,
    shap_health: dict,
) -> list[dict]:
    """
    Tüm analiz çıktılarını değerlendirerek öneri listesi üretir.

    Dönüş
    ------
    list[dict] — her öneri: severity, category, title, detail, actions, timestamp
    """
    recs  = []
    ts_now = str(feature_df.index[-1])

    # 1) Kapasite tahmin temelli öneriler
    recs += _capacity_recommendations(feature_df, capacity_results, shap_capacity, ts_now)

    # 2) Sağlık riski temelli öneriler
    recs += _health_recommendations(health_results, shap_health, ts_now)

    # 3) Ani trend bozulma uyarıları
    recs += _trend_recommendations(feature_df, ts_now)

    # Önem sırasına göre sırala
    order = {"critical": 0, "warning": 1, "info": 2}
    recs  = sorted(recs, key=lambda x: order.get(x["severity"], 9))

    _save_recommendations(recs)
    _print_recommendations(recs)

    return recs


def _capacity_recommendations(feature_df, cap_results, shap_cap, ts_now):
    from src.models.capacity_predictor import predict_days_to_threshold
    recs = []

    for rule in _CAPACITY_RULES:
        col   = rule["resource"]
        info  = predict_days_to_threshold(feature_df, cap_results, col)
        days  = info.get("days_to_threshold")
        cur   = info.get("current_value", 0)
        thr   = info.get("threshold", 90)

        if days is None:
            continue

        severity = (
            "critical" if days <= rule["critical_days"] else
            "warning"  if days <= rule["warn_days"]     else
            None
        )
        if severity is None:
            continue

        # SHAP'tan en etkili özelliği al
        shap_key = next(
            (k for k in shap_cap if col in k and "24h" in k), None
        )
        top_feat = ""
        if shap_key and shap_cap.get(shap_key, {}).get("top_features"):
            top_feat = shap_cap[shap_key]["top_features"][0]["feature"]

        detail = (
            f"Mevcut {rule['label']}: {cur:.1f}{rule['unit']} "
            f"(eşik: {thr}{rule['unit']}). "
            f"Mevcut trend devam ederse eşiğe ulaşma süresi: yaklaşık {days} gün. "
        )
        if top_feat:
            detail += f"En etkili özellik (SHAP): '{top_feat}'."

        recs.append({
            "severity":  severity,
            "category":  "KAPASITE",
            "title":     f"{rule['label']} {'kritik' if severity == 'critical' else 'uyarı'} eşiğine yaklaşıyor",
            "detail":    detail,
            "actions":   rule["suggestions"],
            "timestamp": ts_now,
        })

    return recs


def _health_recommendations(health_results, shap_health, ts_now):
    recs = []
    best_m  = max(health_results["models"].items(), key=lambda x: x[1]["F1"])[0]
    mv      = health_results["models"][best_m]
    y_proba = mv["y_proba"]
    y_test  = health_results["y_test"]

    # Son test penceresi
    last_prob = float(y_proba[-48:].mean()) if len(y_proba) >= 48 else float(y_proba.mean())

    if last_prob >= 0.65:
        severity = "critical" if last_prob >= 0.80 else "warning"

        # En etkili hata tipi (SHAP'tan)
        top_err_type = ""
        err_features = [
            "timeout", "db_connection", "resource_exhausted",
            "auth_failure", "network_error", "scheduler_error",
        ]
        if shap_health.get("top_features"):
            for feat_info in shap_health["top_features"]:
                feat = feat_info["feature"]
                for et in err_features:
                    if et in feat:
                        top_err_type = et
                        break
                if top_err_type:
                    break

        action_base = [
            f"Sistem risk olasılığı yüksek ({last_prob:.0%}). "
            "Nova, Keystone, Neutron servis durumlarını kontrol edin.",
            "Son 6 saat içindeki CRITICAL log kayıtlarını gözden geçirin.",
        ]
        if top_err_type and top_err_type in _HEALTH_SUGGESTIONS:
            action_base.append(_HEALTH_SUGGESTIONS[top_err_type])

        recs.append({
            "severity":  severity,
            "category":  "SAĞLIK",
            "title":     f"Sistem Risk Uyarısı — Olasılık: {last_prob:.0%} ({best_m})",
            "detail":    (
                f"Son 48 saatlik test penceresinde ortalama risk olasılığı {last_prob:.1%}. "
                f"En etkili bileşen: '{top_err_type or 'belirsiz'}'."
            ),
            "actions":   action_base,
            "timestamp": ts_now,
        })

    return recs


def _trend_recommendations(feature_df, ts_now):
    recs   = []
    window = 168   # 1 hafta

    trend_checks = [
        ("error_rate_percent", "Hata Oranı", "%",    0.5,  1.5),
        ("api_latency_ms",     "API Gecikmesi", "ms", 50.0, 150.0),
    ]

    for col, label, unit, warn_rise, crit_rise in trend_checks:
        if col not in feature_df.columns or len(feature_df) < window * 2:
            continue

        prev_week = feature_df[col].iloc[-(window * 2) : -window].mean()
        last_week = feature_df[col].iloc[-window:].mean()
        rise      = last_week - prev_week

        severity = (
            "critical" if rise >= crit_rise else
            "warning"  if rise >= warn_rise  else
            None
        )
        if severity is None:
            continue

        recs.append({
            "severity":  severity,
            "category":  "TREND",
            "title":     f"{label} Son Haftada Belirgin Artış",
            "detail":    (
                f"Bir önceki haftaya göre ortalama {label} "
                f"{rise:+.2f} {unit} arttı "
                f"({prev_week:.2f} → {last_week:.2f} {unit})."
            ),
            "actions":   [
                f"{label} değişimini inceleyin; ilgili servislerin loglarını kontrol edin.",
                "Değişim keskin ise yakın zamanda yapılan yapılandırma güncellemelerini gözden geçirin.",
            ],
            "timestamp": ts_now,
        })

    return recs


# ── Çıktı fonksiyonları ───────────────────────────────────────────────────────

def _save_recommendations(recs: list) -> None:
    out_json = os.path.join(RESULTS_DIR, "reports", "recommendations.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)

    out_txt = os.path.join(RESULTS_DIR, "reports", "recommendations.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  BULUT ALT YAPI PROAKTİF İZLEME — OPERASYONEL ÖNERİLER\n")
        f.write(f"  Oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 70 + "\n\n")
        for i, rec in enumerate(recs, 1):
            sev_label = _SEVERITY_LABELS.get(rec["severity"], rec["severity"].upper())
            f.write(f"[{i}] [{sev_label}] {rec['category']} — {rec['title']}\n")
            f.write(f"     {rec['detail']}\n")
            f.write("     Önerilen Aksiyonlar:\n")
            for j, act in enumerate(rec["actions"], 1):
                f.write(f"       {j}. {act}\n")
            f.write("\n")

    print(f"  → {out_json}")
    print(f"  → {out_txt}")


def _print_recommendations(recs: list) -> None:
    print("\n" + "=" * 70)
    print("  OPERASYONEL ÖNERİLER")
    print("=" * 70)
    for i, rec in enumerate(recs, 1):
        sev_label = _SEVERITY_LABELS.get(rec["severity"], rec["severity"].upper())
        icon = "🔴" if rec["severity"] == "critical" else ("🟡" if rec["severity"] == "warning" else "🔵")
        print(f"\n{icon} [{i}] [{sev_label}] {rec['category']} — {rec['title']}")
        print(f"    {rec['detail']}")
        print("    Aksiyonlar:")
        for j, act in enumerate(rec["actions"], 1):
            print(f"      {j}. {act}")
    print("=" * 70 + "\n")
