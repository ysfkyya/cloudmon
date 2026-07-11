"""
Çoklu LLM Proaktif Öneri Modülü.

Metrik özeti + kapasite tahmini (bitiş tarihleri) + SHAP top özellikleri
+ haftalık hata karşılaştırması + staj dönemi bilgisi →
Claude (Anthropic) ve GPT-4o (OpenAI) modeline aynı prompt gönderilir,
yanıtlar yan yana karşılaştırılır.

Kullanım:
    from src.recommendations.llm_advisor import LLMAdvisor
    advisor = LLMAdvisor(anthropic_key="...", openai_key="...")
    result  = advisor.get_recommendations(feature_df, shap_health, shap_capacity,
                                          capacity_results=..., recommendations=...)
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd


# ── Model tanımları ────────────────────────────────────────────────────────────

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


# ── Prompt oluşturucu ─────────────────────────────────────────────────────────

def build_prompt(context: dict) -> str:
    """
    Zenginleştirilmiş sistem bağlamını LLM prompt'una dönüştürür.

    context anahtarları
    -------------------
    metric_summary    : son 24s ort/maks metrikler
    shap_features     : top SHAP özelliği listesi
    error_summary     : haftalık hata karşılaştırması
    capacity_dates    : her kaynak için bitiş tarihi + günler
    staj_status       : aktif/yaklaşan staj dönemleri
    recommendations   : rule-engine'den gelen yapılandırılmış öneriler
    current_date      : mevcut tarih (str)
    """
    ms   = context.get("metric_summary", {})
    shap = context.get("shap_features", [])
    err  = context.get("error_summary", {})
    cdates = context.get("capacity_dates", {})
    staj = context.get("staj_status", {})
    recs = context.get("recommendations", [])
    dt   = context.get("current_date", datetime.now().strftime("%Y-%m-%d"))

    # SHAP satırları
    shap_lines = "\n".join(
        f"  {i+1}. {f['feature']} (önem: {f['importance']:.3f})"
        for i, f in enumerate(shap[:8])
    ) or "  (SHAP verisi yok)"

    # Kapasite bitiş tarihleri
    cap_lines = []
    labels_map = {
        "cpu_usage_percent":    "CPU  ",
        "memory_usage_percent": "Bellek",
        "disk_usage_percent":   "Disk  ",
    }
    for col, info in cdates.items():
        lbl   = labels_map.get(col, col)
        cur   = info.get("current_value", 0)
        thr   = info.get("threshold", 90)
        trend = info.get("daily_trend", 0)
        if info.get("stable"):
            cap_lines.append(f"  {lbl}: %{cur:.1f} → Kararlı trend (>180 gün, +{trend:.3f}%/gün)")
        elif info.get("estimated_date"):
            days = int(info.get("days_to_threshold", 0))
            cap_lines.append(
                f"  {lbl}: %{cur:.1f} → Eşik %{thr} | "
                f"Tahmini dolum: {info['estimated_date']} (~{days} gün) | "
                f"Günlük artış: +{trend:.3f}%"
            )
        else:
            cap_lines.append(f"  {lbl}: %{cur:.1f} → Eşik %{thr} (yetersiz veri)")
    cap_block = "\n".join(cap_lines) or "  (Kapasite verisi yok)"

    # Haftalık hata özeti
    err_lines = []
    for label, data in err.items():
        flag = " ⚠ YÜKSEK" if data.get("high") else ""
        err_lines.append(
            f"  {label:28s}: Son 7g={data['last_7d']:6,}  "
            f"Önceki 7g={data['prev_7d']:6,}  "
            f"Değişim={data['change_pct']:+.0f}%{flag}"
        )
    err_block = "\n".join(err_lines) or "  (Hata verisi yok)"

    # Staj durumu
    if staj.get("active"):
        s = staj["active"]
        staj_block = (
            f"  ⚡ AKTİF STAJ: {s['name']} ({s['start']} – {s['end']}, "
            f"{s.get('days_left', '?')} gün kaldı)\n"
            f"  Beklenen etki: +{s.get('vm_burst', '?')} VM, "
            f"CPU +{s.get('cpu_extra', '?')}%, Bellek +{s.get('mem_extra', '?')}%, "
            f"Hata +{s.get('error_extra', '?')}%"
        )
    elif staj.get("upcoming"):
        s = staj["upcoming"]
        staj_block = (
            f"  📅 YAKLAŞAN STAJ: {s['name']} başlangıç: {s['start']} "
            f"({s.get('days_to', '?')} gün sonra)\n"
            f"  Planlanan etki: +{s.get('vm_burst', '?')} VM, "
            f"CPU +{s.get('cpu_extra', '?')}%, Bellek +{s.get('mem_extra', '?')}%"
        )
    else:
        staj_block = "  Aktif veya yaklaşan staj dönemi yok."

    # Rule-engine önerileri özeti
    rec_lines = []
    for r in recs[:5]:
        rec_lines.append(f"  [{r.get('severity','?').upper()}] {r.get('category','')}: {r.get('title','')}")
    rec_block = "\n".join(rec_lines) or "  (Otomatik öneri yok)"

    return f"""Sen bir bulut altyapısı (OpenStack) uzman danışmanısın.
Aşağıdaki gerçek zamanlı sistem verilerini analiz ederek SOMUT ve EYLEMSELöneriler sun.

=== MEVCUT TARİH: {dt} ===

━━━ SON 24 SAATLİK METRİK ÖZETİ ━━━
CPU Kullanımı       : Ort %{ms.get('cpu_mean', 0):.1f}  |  Maks %{ms.get('cpu_max', 0):.1f}
Bellek Kullanımı    : Ort %{ms.get('mem_mean', 0):.1f}  |  Maks %{ms.get('mem_max', 0):.1f}
Disk Kullanımı      : Ort %{ms.get('disk_mean', 0):.1f}  |  Maks %{ms.get('disk_max', 0):.1f}
API Gecikmesi       : Ort {ms.get('latency_mean', 0):.0f} ms  |  Maks {ms.get('latency_max', 0):.0f} ms
Hata Oranı          : Ort %{ms.get('error_mean', 0):.2f}  |  Maks %{ms.get('error_max', 0):.2f}
Timeout (son 6s)    : {ms.get('timeout_6h', 0):.0f}
DB Bağlantı (son 6s): {ms.get('db_conn_6h', 0):.0f}
Kota Aşımı (son 6s) : {ms.get('quota_6h', 0):.0f}
Kritik Log (son 6s) : {ms.get('critical_6h', 0):.0f}
VM Sayısı           : {ms.get('vm_count', 0):.0f}

━━━ KAPASİTE DOLUM TAHMİNİ (30 günlük medyan trend) ━━━
{cap_block}

━━━ STAJ DÖNEMİ DURUMU ━━━
{staj_block}

━━━ 7 GÜNLÜK HATA KARŞILAŞTIRMASI ━━━
{err_block}

━━━ SHAP — EN ETKİLİ ÖZELLIKLER (sağlık riski modeli) ━━━
{shap_lines}

━━━ OTOMATİK KURAL MOTORU ÖNERİLERİ ━━━
{rec_block}

━━━ GÖREVİN ━━━
Yukarıdaki verileri bütünsel olarak değerlendirerek:

1. KAPASİTE: CPU, Bellek, Disk için tahmini dolum tarihlerine göre aciliyet sıralı
   planlama öner. "Mevcut trend devam ederse X tarihi Y kaynağı dolacak → şu adımlar
   alınmalı" formatında yaz.

2. STAJ HAZIRLIĞI: Aktif/yaklaşan staj dönemini göz önünde bulundurarak kota,
   zamanlayıcı ve alarm eşikleri için spesifik öneriler sun.

3. HATA TRENDLERI: Artan hata türlerine karşı uygulanabilir 2-3 OpenStack komutu
   veya yapılandırma adımı ver.

4. İZLEME ÖNCELİKLERİ: Önümüzdeki 7 gün için hangi 3 metriği/log türünü önce
   izlemek gerektiğini belirt ve neden.

5. IÇGÖRÜ: "Bu sistem genellikle X koşulda Y hatasına giriyor" formatında
   bir örüntü içgörüsü paylaş.

Yanıtını TÜRKÇE ver. Teknik olmayan jargondan kaçın ama OpenStack servis isimlerini
(Nova, Keystone, Neutron, Cinder) kullan. Madde madde, kısa ve net ol.
Toplam yanıt uzunluğu 600 kelimeyi geçmesin."""


# ── Bağlam çıkarıcı ───────────────────────────────────────────────────────────

def extract_context(
    feature_df: pd.DataFrame,
    shap_health: dict,
    shap_capacity: dict = None,
    capacity_results: dict = None,
    recommendations: list = None,
) -> dict:
    """
    feature_df, model çıktıları ve öneri listesinden zengin prompt bağlamı hazırlar.
    """
    from config import THRESHOLDS, STAJ_PERIODS
    from src.models.capacity_predictor import predict_days_to_threshold
    from src.recommendations.engine import build_error_summary

    last_24 = feature_df.tail(24)

    def _safe(col, fn):
        try:
            return float(fn(last_24[col])) if col in last_24.columns else 0.0
        except Exception:
            return 0.0

    metric_summary = {
        "cpu_mean":     _safe("cpu_usage_percent",    np.mean),
        "cpu_max":      _safe("cpu_usage_percent",    np.max),
        "mem_mean":     _safe("memory_usage_percent", np.mean),
        "mem_max":      _safe("memory_usage_percent", np.max),
        "disk_mean":    _safe("disk_usage_percent",   np.mean),
        "disk_max":     _safe("disk_usage_percent",   np.max),
        "latency_mean": _safe("api_latency_ms",       np.mean),
        "latency_max":  _safe("api_latency_ms",       np.max),
        "error_mean":   _safe("error_rate_percent",   np.mean),
        "error_max":    _safe("error_rate_percent",   np.max),
        "timeout_6h":   _safe("timeout_count",        lambda c: c.tail(6).sum()),
        "db_conn_6h":   _safe("db_connection_count",  lambda c: c.tail(6).sum()),
        "quota_6h":     _safe("quota_exceeded_count", lambda c: c.tail(6).sum()),
        "critical_6h":  _safe("critical_count",       lambda c: c.tail(6).sum()),
        "vm_count":     _safe("vm_count",             lambda c: c.iloc[-1]),
    }

    # SHAP özellikler
    shap_features = (shap_health or {}).get("top_features", [])[:8]

    # Kapasite bitiş tarihleri
    cap_results_obj = capacity_results or {}
    capacity_dates  = {}
    for col in ["cpu_usage_percent", "memory_usage_percent", "disk_usage_percent"]:
        capacity_dates[col] = predict_days_to_threshold(feature_df, cap_results_obj, col)

    # Haftalık hata özeti
    error_summary = build_error_summary(feature_df)

    # Staj durumu
    current_ts = feature_df.index[-1]
    staj_status: dict = {}
    for staj in STAJ_PERIODS:
        s_start = pd.Timestamp(staj["start"])
        s_end   = pd.Timestamp(staj["end"])
        if s_start <= current_ts <= s_end:
            staj_status["active"] = {
                **staj,
                "days_left": max((s_end - current_ts).days, 0),
            }
            break
        elif s_start > current_ts and (s_start - current_ts).days <= 45:
            staj_status["upcoming"] = {
                **staj,
                "days_to": (s_start - current_ts).days,
            }

    return {
        "metric_summary":  metric_summary,
        "shap_features":   shap_features,
        "error_summary":   error_summary,
        "capacity_dates":  capacity_dates,
        "staj_status":     staj_status,
        "recommendations": recommendations or [],
        "current_date":    str(current_ts)[:10],
    }


# ── LLM çağrıları ─────────────────────────────────────────────────────────────

def _call_claude(prompt: str, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=PROVIDERS["claude"]["model_id"],
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _call_openai(prompt: str, api_key: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=PROVIDERS["openai"]["model_id"],
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ── Ana sınıf ─────────────────────────────────────────────────────────────────

class LLMAdvisor:
    def __init__(self, anthropic_key: str = "", openai_key: str = ""):
        self.keys = {
            "claude": anthropic_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            "openai": openai_key    or os.environ.get("OPENAI_API_KEY",    ""),
        }

    def available_providers(self) -> list[str]:
        return [p for p, k in self.keys.items() if k]

    def get_recommendations(
        self,
        feature_df: pd.DataFrame,
        shap_health: dict,
        shap_capacity: dict = None,
        capacity_results: dict = None,
        recommendations: list = None,
    ) -> dict:
        """
        Tüm aktif sağlayıcılara zenginleştirilmiş prompt gönderir.
        Döndürür: {"prompt": ..., "context": ..., "responses": {"claude": {...}, "openai": {...}}}
        """
        context = extract_context(
            feature_df,
            shap_health,
            shap_capacity    or {},
            capacity_results or {},
            recommendations  or [],
        )
        prompt = build_prompt(context)

        results = {}
        callers = {"claude": _call_claude, "openai": _call_openai}

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
                results[provider] = {
                    "status": "error",
                    "error":  str(e),
                    "text":   "",
                }

        return {
            "prompt":    prompt,
            "context":   context,
            "responses": results,
        }
