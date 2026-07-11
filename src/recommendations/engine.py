"""
Proaktif öneri motoru — tarih bazlı çıktı + staj uyarıları + LLM besleme.

Öneri kategorileri
------------------
  KAPASİTE : kaynak eşiğine tahmini ulaşma TARİHİ ile
  SAĞLIK   : risk sınıflandırıcısı yüksek olasılık
  TREND    : haftalık hata oranı bozulması
  STAJ     : aktif veya yaklaşan staj dönemi uyarısı
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import THRESHOLDS, RISK_HORIZON, RESULTS_DIR, STAJ_PERIODS, CAPACITY_EXPANSIONS


_CAPACITY_RULES = [
    {
        "resource":      "cpu_usage_percent",
        "label":         "CPU Kullanımı",
        "unit":          "%",
        "warn_days":     60,
        "critical_days": 21,
        "suggestions": [
            "Compute node'larında yük dengelemesi (live migration) uygulayın.",
            "VM flavour sınırlamalarını gözden geçirin; CPU yoğun iş yüklerini izole edin.",
            "Ölçek genişletme (scale-out) için ek compute node alımını planlayın.",
            "Staj dönemine denk geliyorsa kullanıcı kotalarını daraltın.",
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
            "Staj kullanıcılarının disk kotalarını kontrol edin.",
        ],
    },
]

_HEALTH_SUGGESTIONS = {
    "timeout":             "Nova API ve Keystone timeout değerlerini artırın; bağlantı havuzu boyutunu büyütün.",
    "db_connection":       "MySQL/MariaDB max_connections parametresini artırın; bağlantı havuzu sağlığını izleyin.",
    "resource_exhausted":  "Yük dengesizliğini gidermek için Nova scheduler ağırlıklarını ve filtrelerini ayarlayın.",
    "quota_exceeded":      "Keystone proje kotalarını gözden geçirin; staj kullanıcıları kota aşımına yol açıyor olabilir.",
    "auth_failure":        "Keystone token süresi ve revocation listesi yapılandırmasını gözden geçirin.",
    "network_error":       "Neutron OVS agent durumunu ve VLAN yapılandırmalarını kontrol edin.",
    "scheduler_error":     "Nova compute node'larının servis durumunu ve kaynak raporlamasını doğrulayın.",
    "io_error":            "Cinder backend sağlığını ve iSCSI bağlantılarını denetleyin.",
}

_SEVERITY_LABELS = {
    "critical": "KRİTİK",
    "warning":  "UYARI",
    "info":     "BİLGİ",
}


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def generate_recommendations(
    feature_df: pd.DataFrame,
    capacity_results: dict,
    health_results: dict,
    shap_capacity: dict,
    shap_health: dict,
) -> list[dict]:
    recs   = []
    ts_now = str(feature_df.index[-1])

    recs += _capacity_recommendations(feature_df, capacity_results, shap_capacity, ts_now)
    recs += _health_recommendations(health_results, shap_health, ts_now)
    recs += _trend_recommendations(feature_df, ts_now)
    recs += _staj_recommendations(ts_now)

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
        detail = (
            f"Mevcut {rule['label']}: {cur:.1f}{rule['unit']} (eşik: {thr}{rule['unit']}). "
            f"30 günlük medyan artış: +{daily_trend:.3f}{rule['unit']}/gün. "
            f"Tahmini eşik tarihi: {est} (~{int(days)} gün)."
        )
        if top_feat:
            detail += f" En etkili SHAP özelliği: '{top_feat}'."

        recs.append({
            "severity":        severity,
            "category":        "KAPASİTE",
            "title":           f"{rule['label']} — Tahmini Eşik Tarihi: {est}",
            "detail":          detail,
            "actions":         rule["suggestions"],
            "estimated_date":  est,
            "days_remaining":  int(days),
            "timestamp":       ts_now,
        })

    return recs


# ── Sağlık önerileri ─────────────────────────────────────────────────────────

def _health_recommendations(health_results, shap_health, ts_now):
    recs   = []
    best_m = max(health_results["models"].items(), key=lambda x: x[1]["F1"])[0]
    mv     = health_results["models"][best_m]
    y_proba = mv["y_proba"]

    last_prob = float(y_proba[-48:].mean()) if len(y_proba) >= 48 else float(y_proba.mean())

    if last_prob >= 0.65:
        severity = "critical" if last_prob >= 0.80 else "warning"

        err_features = [
            "timeout", "db_connection", "resource_exhausted",
            "quota_exceeded", "auth_failure", "network_error", "scheduler_error",
        ]
        top_err_type = ""
        if shap_health.get("top_features"):
            for feat_info in shap_health["top_features"]:
                for et in err_features:
                    if et in feat_info["feature"]:
                        top_err_type = et
                        break
                if top_err_type:
                    break

        action_base = [
            f"Sistem risk olasılığı yüksek ({last_prob:.0%}). Nova, Keystone, Neutron servis durumlarını kontrol edin.",
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


# ── Trend önerileri ───────────────────────────────────────────────────────────

def _trend_recommendations(feature_df, ts_now):
    recs   = []
    window = 168  # 1 hafta

    trend_checks = [
        ("error_rate_percent",        "Hata Oranı",     "%",   0.5,   1.5),
        ("api_latency_ms",            "API Gecikmesi",  "ms",  50.0,  150.0),
        ("resource_exhausted_count",  "Kaynak Yetersizliği Hatası", "adet", 20.0, 60.0),
        ("quota_exceeded_count",      "Kota Aşımı Hatası",          "adet",  5.0, 20.0),
        ("timeout_count",             "Timeout Hatası", "adet", 30.0,  100.0),
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
                "Değişim staj dönemine denk geliyorsa kullanıcı kota ayarlarını gözden geçirin.",
                "Değişim keskin ise yakın zamanda yapılan yapılandırma güncellemelerini kontrol edin.",
            ],
            "timestamp": ts_now,
        })

    return recs


# ── Staj uyarıları ────────────────────────────────────────────────────────────

def _staj_recommendations(ts_now: str) -> list[dict]:
    recs       = []
    current_ts = pd.Timestamp(ts_now)

    for staj in STAJ_PERIODS:
        start = pd.Timestamp(staj["start"])
        end   = pd.Timestamp(staj["end"])

        if start <= current_ts <= end:
            days_left = max((end - current_ts).days, 0)
            recs.append({
                "severity":  "warning",
                "category":  "STAJ",
                "title":     f"{staj['name']} Aktif — {days_left} Gün Kaldı ({staj['end']})",
                "detail":    (
                    f"Aktif staj dönemi: {staj['start']} → {staj['end']}. "
                    f"Geçmiş veriye göre bu dönemde +{staj['vm_burst']} VM, "
                    f"CPU +{staj['cpu_extra']:.0f}%, Bellek +{staj['mem_extra']:.0f}% "
                    f"yük artışı ve hata oranında +{staj['error_extra']:.1f}% artış beklenir."
                ),
                "actions": [
                    "Nova scheduler ağırlıklarını ve host aggregate'leri kontrol edin.",
                    f"Keystone proje kotalarını doğrulayın — {staj['vm_burst']} ek VM için yeterli kota var mı?",
                    "Quota exceeded ve resource_exhausted hatalarındaki artış staj kaynaklı olabilir; alarm eşiklerini geçici ayarlayın.",
                    "Staj bitiminde oluşturulan VM'lerin temizlenmesi için kullanıcılara bildirim gönderin.",
                ],
                "timestamp": ts_now,
            })

        elif start > current_ts and (start - current_ts).days <= 45:
            days_to = (start - current_ts).days
            recs.append({
                "severity":  "info",
                "category":  "STAJ",
                "title":     f"{staj['name']} {days_to} Gün Sonra Başlıyor ({staj['start']})",
                "detail":    (
                    f"Geçmiş dönem verilerine göre {staj['name']} başladığında "
                    f"~{staj['vm_burst']} ek VM, CPU'da +{staj['cpu_extra']:.0f}%, "
                    f"Bellekte +{staj['mem_extra']:.0f}% artış beklenir. "
                    f"Hazırlık için {days_to} gününüz var."
                ),
                "actions": [
                    "Mevcut compute kapasitesinin staj yükünü kaldırıp kaldıramayacağını hesaplayın.",
                    "Staj kullanıcı hesaplarını ve proje kotalarını staj başlamadan önce yapılandırın.",
                    "Log ve alarm sistemlerinin staj dönemine göre ayarlandığından emin olun.",
                    "Gerekirse yeni donanım alımı için satın alma sürecini başlatın.",
                ],
                "timestamp": ts_now,
            })

    return recs


# ── Hata özeti (LLM ve raporlar için) ────────────────────────────────────────

def build_error_summary(feature_df: pd.DataFrame) -> dict:
    """
    Son 7 gün ile önceki 7 günü karşılaştıran yapılandırılmış hata özeti.
    LLM prompt'una ve dashboard'a beslenir.
    """
    if len(feature_df) < 336:
        return {}

    prev_week = feature_df.iloc[-336:-168]
    last_week = feature_df.iloc[-168:]

    error_cols = {
        "timeout_count":            "Timeout",
        "db_connection_count":      "DB Bağlantı",
        "resource_exhausted_count": "Kaynak Yetersizliği",
        "quota_exceeded_count":     "Kota Aşımı",
        "auth_failure_count":       "Kimlik Doğrulama",
        "network_error_count":      "Ağ Hatası",
        "scheduler_error_count":    "Zamanlayıcı",
        "critical_count":           "CRITICAL Log",
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

def _save_recommendations(recs: list, feature_df: pd.DataFrame, cap_results: dict) -> None:
    from src.models.capacity_predictor import predict_days_to_threshold

    # Kapasite özeti
    cap_summary = {}
    for col in ["cpu_usage_percent", "memory_usage_percent", "disk_usage_percent"]:
        info = predict_days_to_threshold(feature_df, cap_results, col)
        cap_summary[col] = info

    # Hata özeti
    err_summary = build_error_summary(feature_df)

    # JSON
    out_json = os.path.join(RESULTS_DIR, "reports", "recommendations.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "recommendations": recs,
            "capacity_summary": cap_summary,
            "error_summary":    err_summary,
        }, f, ensure_ascii=False, indent=2)

    # TXT raporu
    out_txt = os.path.join(RESULTS_DIR, "reports", "recommendations.txt")
    _write_txt_report(out_txt, recs, cap_summary, err_summary)

    print(f"  → {out_json}")
    print(f"  → {out_txt}")


def _write_txt_report(path, recs, cap_summary, err_summary):
    labels = {
        "cpu_usage_percent":    "CPU   ",
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
            days = info.get("days_to_threshold")
            stb  = info.get("stable", False)
            trend = info.get("daily_trend", 0)

            if stb or days is None:
                status = "Kararlı trend (>180 gün)"
            else:
                status = f"Tahmini eşik: {est} (~{int(days)} gün)"

            f.write(f"  {lbl}: %{cur:.1f} → Eşik %{thr} | +{trend:.3f}%/gün | {status}\n")

        f.write("\n── 7 GÜNLÜK HATA ÖZETİ ─────────────────────────────────────────\n")
        for label, data in err_summary.items():
            flag = " ⚠️" if data.get("high") else ""
            f.write(f"  {label:25s}: {data['last_7d']:6,} kez  "
                    f"({data['change_pct']:+.0f}% önceki haftaya göre){flag}\n")

        f.write("\n── ÖNERİLER ──────────────────────────────────────────────────────\n")
        for i, rec in enumerate(recs, 1):
            sev = _SEVERITY_LABELS.get(rec["severity"], rec["severity"].upper())
            f.write(f"\n[{i}] [{sev}] {rec['category']} — {rec['title']}\n")
            f.write(f"     {rec['detail']}\n")
            f.write("     Aksiyonlar:\n")
            for j, act in enumerate(rec.get("actions", []), 1):
                f.write(f"       {j}. {act}\n")


def _print_recommendations(recs: list) -> None:
    print("\n" + "=" * 70)
    print("  OPERASYONEL ÖNERİLER")
    print("=" * 70)
    icons = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
    for i, rec in enumerate(recs, 1):
        sev  = _SEVERITY_LABELS.get(rec["severity"], rec["severity"].upper())
        icon = icons.get(rec["severity"], "🔵")
        print(f"\n{icon} [{i}] [{sev}] {rec['category']} — {rec['title']}")
        print(f"    {rec['detail']}")
        print("    Aksiyonlar:")
        for j, act in enumerate(rec.get("actions", []), 1):
            print(f"      {j}. {act}")
    print("=" * 70 + "\n")
