"""
Çoklu LLM Proaktif Öneri Modülü.

Metrik analizi + kapasite tahmini + log pipeline içgörüleri (anomaliler,
ön-arıza örüntüleri, trend analizi, servis sağlık durumu) → LLM'e gönderilir.

Her iki taraf da (Claude + GPT-4o) aynı prompt'u alır; yanıtlar karşılaştırılır.
Önemli: Prompt hem metriklerden hem de log analizinden elde edilen çıkarımları içerir.
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import ANTHROPIC_API_KEY as _CFG_ANTHROPIC, OPENAI_API_KEY as _CFG_OPENAI

PROVIDERS = {
    "claude": {
        "label":    "Claude Sonnet (Anthropic)",
        "model_id": "claude-sonnet-4-6",
        "color":    "#7C3AED",
    },
    "openai": {
        "label":    "GPT-4o (OpenAI)",
        "model_id": "gpt-4o",
        "color":    "#10a37f",
    },
}


def build_prompt(context: dict) -> str:
    """
    Zenginleştirilmiş sistem bağlamını LLM prompt'una dönüştürür.
    Hem metrik hemde log analizi çıkarımları dahildir.

    context anahtarları:
      metric_summary      : son 24s ort/maks metrikler
      capacity_dates      : her kaynak için eşik tarihi + günler
      error_summary       : 7 günlük log karşılaştırması
      shap_features       : top SHAP özelliği listesi
      log_anomalies_24h   : son 24s anomali tespitleri
      log_trends_7d       : 7g trend değişimleri
      prefailure_patterns : tarihsel ön-arıza örüntüleri
      cascade_events      : cascade riski dönemleri
      service_health      : servis sağlık durumu (log bazlı)
      recommendations     : kural motoru önerileri
      current_date        : mevcut tarih
    """
    ms      = context.get("metric_summary", {})
    cdates  = context.get("capacity_dates", {})
    err     = context.get("error_summary", {})
    shap    = context.get("shap_features", [])
    recs    = context.get("recommendations", [])
    dt      = context.get("current_date", datetime.now().strftime("%Y-%m-%d"))

    # Log analizi çıkarımları
    log_anom     = context.get("log_anomalies_24h", [])
    log_trends   = context.get("log_trends_7d", [])
    log_patterns = context.get("prefailure_patterns", [])
    cascades     = context.get("cascade_events", [])
    svc_health   = context.get("service_health", {})

    # ── Kapasite satırları ────────────────────────────────────────────────────
    labels_map = {
        "vcpu_usage_percent":   "vCPU  ",
        "memory_usage_percent": "Bellek",
        "disk_usage_percent":   "Disk  ",
    }
    cap_lines = []
    for col, info in cdates.items():
        lbl   = labels_map.get(col, col)
        cur   = info.get("current_value", 0)
        thr   = info.get("threshold", 90)
        trend = info.get("daily_trend", 0)
        if info.get("stable"):
            cap_lines.append(f"  {lbl}: %{cur:.1f} → Kararlı (>180 gün, +{trend:.3f}%/gün)")
        elif info.get("estimated_date"):
            days   = int(info.get("days_to_threshold", 0))
            months = info.get("months_to_threshold")
            time_s = f"~{months:.1f} ay / ~{days} gün" if months else f"~{days} gün"
            cap_lines.append(
                f"  {lbl}: %{cur:.1f} → Eşik %{thr} | Tahmini dolum: "
                f"{info['estimated_date']} ({time_s}) | +{trend:.3f}%/gün"
            )
        else:
            cap_lines.append(f"  {lbl}: %{cur:.1f} → Eşik %{thr} (yetersiz veri)")
    cap_block = "\n".join(cap_lines) or "  (Kapasite verisi yok)"

    # ── Log anomali satırları (son 24 saat) ───────────────────────────────────
    anom_lines = []
    for a in log_anom[:6]:
        warn_tag = "[ERKEN UYARI]" if a.get("is_warning") else "[HATA]"
        anom_lines.append(
            f"  {warn_tag} {a['label']}: {a['current_mean']:.0f} adet/saat "
            f"(normal: {a['baseline']:.0f}) — z-skor: {a['max_zscore']:.1f}"
        )
    anom_block = "\n".join(anom_lines) or "  Son 24 saatte anlamlı anomali tespit edilmedi."

    # ── Log trend satırları (7 gün) ───────────────────────────────────────────
    trend_lines = []
    for t in log_trends[:6]:
        dir_icon = "↑" if t["direction"] == "increasing" else "↓" if t["direction"] == "decreasing" else "→"
        warn_tag = "[⚠]" if t.get("is_warning") else "   "
        trend_lines.append(
            f"  {warn_tag} {t['label']}: {dir_icon} {t['change_pct']:+.0f}% "
            f"({t['prev_7d']:.0f}→{t['last_7d']:.0f} adet)"
        )
    trend_block = "\n".join(trend_lines) or "  Log trendi değişimi yok."

    # ── Ön-arıza örüntüleri ───────────────────────────────────────────────────
    pattern_lines = []
    for p in log_patterns[:5]:
        pattern_lines.append(
            f"  '{p['label']}': Arızadan ~{p['lead_time_hours']:.0f} saat önce "
            f"+{p['avg_increase_pct']:.0f}% artış gösterir "
            f"(güven: {p['confidence']:.0%}, {p['observed_times']} olay)"
        )
    pattern_block = "\n".join(pattern_lines) or "  Yeterli tarihsel örüntü çıkarılamadı."

    # ── Cascade uyarısı ───────────────────────────────────────────────────────
    if cascades:
        c = cascades[-1]
        cascade_block = (
            f"  YAKINDA CASCADE: {c['start']} tarihinde {c['duration_h']:.1f} saatlik "
            f"çok servisli bozulma tespit edildi."
        )
    else:
        cascade_block = "  Yakın dönemde cascade riski tespit edilmedi."

    # ── Servis sağlık durumu ──────────────────────────────────────────────────
    health_map = {"critical": "KRİTİK", "degraded": "BOZUK", "warning": "UYARI", "healthy": "SAĞLIKLI"}
    health_lines = [
        f"  {svc.upper():18s}: {health_map.get(state, state.upper())}"
        for svc, state in svc_health.items()
    ]
    health_block = "\n".join(health_lines) or "  Servis sağlık verisi yok."

    # ── SHAP satırları ────────────────────────────────────────────────────────
    shap_lines = "\n".join(
        f"  {i+1}. {f['feature']} (önem: {f['importance']:.3f})"
        for i, f in enumerate(shap[:8])
    ) or "  (SHAP verisi yok)"

    # ── Hata özeti ────────────────────────────────────────────────────────────
    err_lines = []
    for label, data in err.items():
        flag = " !!! YÜKSEK" if data.get("high") else ""
        err_lines.append(
            f"  {label:28s}: Son7g={data['last_7d']:5,}  "
            f"Önceki7g={data['prev_7d']:5,}  {data['change_pct']:+.0f}%{flag}"
        )
    err_block = "\n".join(err_lines) or "  (Hata verisi yok)"

    # ── Kural motoru özeti ────────────────────────────────────────────────────
    rec_lines = []
    for r in recs[:5]:
        rec_lines.append(f"  [{r.get('severity','?').upper()}] {r.get('category','')}: {r.get('title','')}")
    rec_block = "\n".join(rec_lines) or "  (Otomatik öneri yok)"

    return f"""Sen bir bulut altyapısı (OpenStack) uzman danışmanısın.
Aşağıdaki gerçek zamanlı sistem verileri hem metrik analizinden hem de çok aşamalı
log pipeline analizinden elde edilmiştir. Bu bütünleşik veriyi kullanarak SOMUT,
EYLEMSELve proaktif öneriler sun.

=== MEVCUT TARİH: {dt} ===

━━━ SON 24 SAATLİK METRİK ÖZETİ ━━━
vCPU Kullanımı       : {ms.get('vcpu_used', 0):.0f} / {ms.get('vcpu_total', 2000):.0f}  (%{ms.get('vcpu_mean', 0):.1f} ort, %{ms.get('vcpu_max', 0):.1f} maks)  |  VM: {ms.get('vm_count', 0):.0f}
Bellek Kullanımı     : {ms.get('mem_used_gb', 0):.0f} / {ms.get('mem_total_gb', 8000):.0f} GB  (%{ms.get('mem_mean', 0):.1f} ort, %{ms.get('mem_max', 0):.1f} maks)
Disk Kullanımı       : {ms.get('disk_used_tb', 0):.0f} / {ms.get('disk_total_tb', 500):.0f} TB  (%{ms.get('disk_mean', 0):.1f} ort, %{ms.get('disk_max', 0):.1f} maks)
API Gecikmesi        : Ort {ms.get('latency_mean', 0):.0f} ms  |  Maks {ms.get('latency_max', 0):.0f} ms
Hata Oranı           : Ort %{ms.get('error_mean', 0):.2f}  |  Maks %{ms.get('error_max', 0):.2f}

━━━ KAPASİTE DOLUM TAHMİNİ (ML modeli, 30g medyan trend) ━━━
{cap_block}

━━━ LOG ANALİZİ — SON 24 SAAT ANOMALİLERİ ━━━
(z-skor tabanlı anormal yoğunlaşmalar — [ERKEN UYARI]=problem öncesi sinyal)
{anom_block}

━━━ LOG ANALİZİ — 7 GÜNLÜK TREND DEĞİŞİMİ ━━━
([⚠] = proaktif erken uyarı log tipi)
{trend_block}

━━━ TARİHSEL ÖN-ARIZA ÖRÜNTÜLER (Log Pipeline Çıkarımı) ━━━
(Geçmiş {len(log_patterns)} incident analiz edilerek çıkarılan log imzaları)
{pattern_block}

━━━ CASCADE RİSKİ ━━━
{cascade_block}

━━━ SERVİS SAĞLIK DURUMU (Log Bazlı, Son 24s) ━━━
{health_block}

━━━ 7 GÜNLÜK LOG KARŞILAŞTIRMASI ━━━
{err_block}

━━━ SHAP — KAPASİTE TAHMİNİNDE EN ETKİLİ ÖZELLİKLER ━━━
{shap_lines}

━━━ OTOMATİK KURAL MOTORU ÖNERİLERİ (Özet) ━━━
{rec_block}

━━━ GÖREVİN ━━━
Yukarıdaki metrik VE log analizi verilerini birlikte değerlendirerek:

1. ERKEN UYARI DEĞERLENDİRMESİ: Log anomalileri ve ön-arıza örüntüleri ışığında
   sistemin önümüzdeki 24-72 saatte hangi risk altında olduğunu açıkla. Hangi
   log sinyali ne tür bir soruna işaret ediyor?

2. KAPASİTE PLANLAMA: vCPU/Bellek/Disk için tahmini dolum tarihlerine göre aciliyet
   sıralı önlem öner. "X tarihi Y kaynağı dolacak → şu adımlar alınmalı" formatında.

3. LOG ÖRÜNTÜSÜ ANALİZİ: Tarihsel ön-arıza örüntülerinden (örneğin hangi log tipi
   arızadan kaç saat önce artıyor) çıkarılan en önemli 2-3 içgörüyü paylaş.

4. SERVİS ÖNCELİKLERİ: Bozuk/Kritik servisler için uygulanabilir OpenStack komutu
   veya yapılandırma adımı ver (nova, keystone, neutron, cinder komutları).

5. KAYNAK PLANLAMA: Mevcut trendlere göre kota ve kaynak planlaması için
   spesifik öneriler sun.

Yanıtını TÜRKÇE ver. Teknik jargondan kaçın ama OpenStack servis adlarını kullan.
Madde madde, kısa ve net ol. Toplam yanıt 700 kelimeyi geçmesin."""


def extract_context(
    feature_df: pd.DataFrame,
    shap_capacity: dict,
    capacity_results: dict = None,
    recommendations: list = None,
    log_insights: dict = None,
) -> dict:
    """
    feature_df, model çıktıları, log pipeline sonuçları ve öneri listesinden
    zengin prompt bağlamı hazırlar.
    """
    from config import THRESHOLDS
    from src.models.capacity_predictor import predict_days_to_threshold
    from src.recommendations.engine import build_error_summary

    last_24 = feature_df.tail(24)

    def _safe(col, fn):
        try:
            return float(fn(last_24[col])) if col in last_24.columns else 0.0
        except Exception:
            return 0.0

    metric_summary = {
        "vcpu_mean":      _safe("vcpu_usage_percent",    np.mean),
        "vcpu_max":       _safe("vcpu_usage_percent",    np.max),
        "vcpu_used":      _safe("vcpu_used",             lambda c: c.iloc[-1]),
        "vcpu_total":     _safe("vcpu_total",            lambda c: c.iloc[-1]),
        "mem_mean":       _safe("memory_usage_percent",  np.mean),
        "mem_max":        _safe("memory_usage_percent",  np.max),
        "mem_used_gb":    _safe("memory_used_gb",        lambda c: c.iloc[-1]),
        "mem_total_gb":   _safe("memory_total_gb",       lambda c: c.iloc[-1]),
        "disk_mean":      _safe("disk_usage_percent",    np.mean),
        "disk_max":       _safe("disk_usage_percent",    np.max),
        "disk_used_tb":   _safe("disk_used_tb",          lambda c: c.iloc[-1]),
        "disk_total_tb":  _safe("disk_total_tb",         lambda c: c.iloc[-1]),
        "latency_mean":   _safe("api_latency_ms",        np.mean),
        "latency_max":    _safe("api_latency_ms",        np.max),
        "error_mean":     _safe("error_rate_percent",    np.mean),
        "error_max":      _safe("error_rate_percent",    np.max),
        "vm_count":       _safe("vm_count",              lambda c: c.iloc[-1]),
    }

    # SHAP top özellikler
    shap_features = []
    for key, val in (shap_capacity or {}).items():
        if "24h" in key and val.get("top_features"):
            shap_features = val["top_features"][:8]
            break

    # Kapasite bitiş tarihleri
    cap_results_obj = capacity_results or {}
    capacity_dates  = {}
    for col in ["vcpu_usage_percent", "memory_usage_percent", "disk_usage_percent"]:
        capacity_dates[col] = predict_days_to_threshold(feature_df, cap_results_obj, col)

    # Haftalık hata özeti
    error_summary = build_error_summary(feature_df)

    # Log pipeline içgörüleri
    li = log_insights or {}
    log_anomalies_24h   = li.get("anomalies_last_24h", [])
    log_trends_7d       = li.get("log_trends_7d", [])
    prefailure_patterns = li.get("prefailure_patterns", [])
    cascade_events      = li.get("cascade_events_recent", [])
    service_health      = li.get("service_health", {})

    return {
        "metric_summary":      metric_summary,
        "shap_features":       shap_features,
        "error_summary":       error_summary,
        "capacity_dates":      capacity_dates,
        "recommendations":     recommendations or [],
        "current_date":        str(feature_df.index[-1])[:10],
        "log_anomalies_24h":   log_anomalies_24h,
        "log_trends_7d":       log_trends_7d,
        "prefailure_patterns": prefailure_patterns,
        "cascade_events":      cascade_events,
        "service_health":      service_health,
    }


def _call_claude(prompt: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=PROVIDERS["claude"]["model_id"],
        max_tokens=1800,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _call_openai(prompt: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=PROVIDERS["openai"]["model_id"],
        max_tokens=1800,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


class LLMAdvisor:
    def __init__(self, anthropic_key: str = "", openai_key: str = ""):
        self.keys = {
            "claude": anthropic_key or _CFG_ANTHROPIC or os.environ.get("ANTHROPIC_API_KEY", ""),
            "openai": openai_key    or _CFG_OPENAI    or os.environ.get("OPENAI_API_KEY",    ""),
        }

    def available_providers(self) -> list:
        return [p for p, k in self.keys.items() if k]

    def get_recommendations(
        self,
        feature_df: pd.DataFrame,
        shap_capacity: dict,
        capacity_results: dict = None,
        recommendations: list = None,
        log_insights: dict = None,
    ) -> dict:
        """
        Tüm aktif sağlayıcılara zenginleştirilmiş prompt gönderir.
        Prompt hem metrik hem de log analizi çıkarımlarını içerir.
        """
        context = extract_context(
            feature_df,
            shap_capacity,
            capacity_results or {},
            recommendations  or [],
            log_insights     or {},
        )
        prompt = build_prompt(context)

        results  = {}
        callers  = {"claude": _call_claude, "openai": _call_openai}

        for provider, fn in callers.items():
            key = self.keys.get(provider, "")
            if not key:
                results[provider] = {
                    "status": "skipped",
                    "error":  "API anahtarı girilmedi",
                    "text":   "",
                }
                continue
            try:
                text = fn(prompt, key)
                results[provider] = {"status": "ok", "text": text, "error": ""}
            except Exception as e:
                results[provider] = {"status": "error", "error": str(e), "text": ""}

        return {
            "prompt":    prompt,
            "context":   context,
            "responses": results,
        }
