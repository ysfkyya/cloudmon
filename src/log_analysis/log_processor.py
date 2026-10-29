"""
Log işleme hattı — çok aşamalı analiz pipeline'ı.

Tez bağlamı: Problem gelişimini önceden sezmek için loglar dört aşamadan geçirilir.

Aşama 1 — Ayrıştırma & Sınıflandırma:
  Ham loglar tip, servis ve kritiklik bazında sınıflandırılır.
  Erken uyarı tipleri ile hata tipleri ayrıştırılır.

Aşama 2 — Zamansal Agregasyon & Trend Analizi:
  Saatlik sayımlardan 6s/24s/72s hareketli ortalama ve z-skor hesaplanır.
  7 günlük pencerede her log tipinin trend yönü (artış/azalış) belirlenir.

Aşama 3 — Anomali Tespiti:
  Z-skor (σ>2.5) ve patlama tespiti (spike detection) ile anormal log
  yoğunlaşmaları tespit edilir.
  Birden fazla servisin aynı anda bozulması cascade riski olarak işaretlenir.

Aşama 4 — Ön-Arıza Örüntü Çıkarımı:
  Geçmiş incident'lerin öncesindeki log imzaları çıkarılır.
  "Hangi log tipleri, arızadan kaç saat önce, ne oranda artıyor?" sorusu yanıtlanır.
  Bu örüntüler LLM'e zenginleştirilmiş bağlam olarak beslenir.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import RESULTS_DIR, START_DATE, END_DATE

# Log tipleri
_ERROR_TYPES = [
    "timeout_count", "db_connection_count", "resource_exhausted_count",
    "quota_exceeded_count", "auth_failure_count", "network_error_count",
    "scheduler_error_count", "io_error_count",
]

_WARNING_TYPES = [
    "slow_response_count", "memory_pressure_count", "connection_pool_warn_count",
    "scheduler_degradation_count", "quota_pressure_count",
    "disk_io_degradation_count", "cascade_indicator_count",
]

_ALL_LOG_TYPES = _ERROR_TYPES + _WARNING_TYPES

_LABELS = {
    "timeout_count":              "Timeout Hatası",
    "db_connection_count":        "DB Bağlantı Hatası",
    "resource_exhausted_count":   "Kaynak Tükenmesi",
    "quota_exceeded_count":       "Kota Aşımı",
    "auth_failure_count":         "Kimlik Doğrulama Hatası",
    "network_error_count":        "Ağ Hatası",
    "scheduler_error_count":      "Zamanlayıcı Hatası",
    "io_error_count":             "Disk I/O Hatası",
    "slow_response_count":        "Yavaş Yanıt (Erken Uyarı)",
    "memory_pressure_count":      "Bellek Baskısı (Erken Uyarı)",
    "connection_pool_warn_count": "Bağlantı Havuzu Uyarısı (Erken Uyarı)",
    "scheduler_degradation_count":"Zamanlayıcı Bozulması (Erken Uyarı)",
    "quota_pressure_count":       "Kota Baskısı (Erken Uyarı)",
    "disk_io_degradation_count":  "Disk I/O Bozulması (Erken Uyarı)",
    "cascade_indicator_count":    "Cascade Göstergesi (Erken Uyarı)",
}


# ── Aşama 1: Ayrıştırma & Sınıflandırma ─────────────────────────────────────

def stage1_parse_classify(log_feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Saatlik log özellik tablosundan servis sağlık skorları ve tip sınıflandırması üretir.
    Hata logları ile erken uyarı logları ayrı ayrı değerlendirilir.
    """
    df = log_feat_df.copy()

    # Her servis için normalize hata skoru [0..1]
    service_cols = [
        "nova_api_errors", "nova_compute_errors", "nova_scheduler_errors",
        "keystone_errors", "neutron_errors", "cinder_errors",
    ]
    for col in service_cols:
        if col in df.columns:
            max_val = df[col].quantile(0.99) + 1e-6
            df[f"{col}_score"] = (df[col] / max_val).clip(0, 1)

    # Erken uyarı bileşik skoru [0..1]
    warn_cols = [c for c in _WARNING_TYPES if c in df.columns]
    if warn_cols:
        warn_sum = df[warn_cols].sum(axis=1)
        max_warn = warn_sum.quantile(0.99) + 1e-6
        df["early_warning_score"] = (warn_sum / max_warn).clip(0, 1)
    else:
        df["early_warning_score"] = 0.0

    # Genel hata bileşik skoru [0..1]
    err_cols = [c for c in _ERROR_TYPES if c in df.columns]
    if err_cols:
        err_sum = df[err_cols].sum(axis=1)
        max_err = err_sum.quantile(0.99) + 1e-6
        df["error_score"] = (err_sum / max_err).clip(0, 1)
    else:
        df["error_score"] = 0.0

    # Kombinasyon: bileşik log risk skoru
    df["log_risk_score"] = (0.4 * df["early_warning_score"] + 0.6 * df["error_score"]).clip(0, 1)

    return df


# ── Aşama 2: Zamansal Agregasyon & Trend Analizi ─────────────────────────────

def stage2_temporal_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her log tipi için hareketli ortalamaları ve z-skor tabanlı normalizasyonu hesaplar.
    7 günlük pencerede trend yönü belirlenir.
    """
    df = df.copy()
    ts = pd.DatetimeIndex(df["timestamp"])
    df = df.set_index(ts)

    windows = [6, 24, 72]
    for col in _ALL_LOG_TYPES:
        if col not in df.columns:
            continue
        for w in windows:
            df[f"{col}_rmean_{w}h"] = df[col].rolling(w, min_periods=1).mean()

        # Baseline: 7 günlük geçmiş ortalama ve std
        baseline_mean = df[col].rolling(168, min_periods=24).mean()
        baseline_std  = df[col].rolling(168, min_periods=24).std().fillna(1.0) + 1e-6
        df[f"{col}_zscore"] = (df[col] - baseline_mean) / baseline_std

        # 7 günlük trend: son 7 gün vs önceki 7 gün
        last_7d = df[col].rolling(168, min_periods=1).mean()
        prev_7d = df[col].rolling(168, min_periods=1).mean().shift(168)
        df[f"{col}_trend_7d"] = (last_7d - prev_7d) / (prev_7d.abs() + 1e-6) * 100.0

    df = df.reset_index(drop=True)
    return df


# ── Aşama 3: Anomali Tespiti ─────────────────────────────────────────────────

def stage3_anomaly_detection(df: pd.DataFrame) -> dict:
    """
    Z-skor ve patlama tespiti ile anomali dönemlerini bulur.
    Cascade riski: birden fazla log tipinin aynı anda anomali göstermesi.

    Dönüş:
    -------
    {
        "anomalies_last_24h": [...],    # Son 24 saatteki anomaliler
        "anomalies_last_7d":  [...],    # Son 7 gündeki anomaliler
        "cascade_events":     [...],    # Cascade riski olan dönemler
        "anomaly_timeline":   DataFrame # Tam anomali zaman çizelgesi
    }
    """
    ts  = pd.DatetimeIndex(df["timestamp"])
    df2 = df.set_index(ts)

    last_24h = df2.iloc[-24:] if len(df2) >= 24 else df2
    last_7d  = df2.iloc[-168:] if len(df2) >= 168 else df2

    def _extract_anomalies(window_df, window_label, zscore_thr=2.5):
        anomalies = []
        for col in _ALL_LOG_TYPES:
            zc = f"{col}_zscore"
            if zc not in window_df.columns:
                continue
            max_z = window_df[zc].max()
            if max_z >= zscore_thr:
                baseline_col = f"{col}_rmean_72h"
                baseline = (float(window_df[baseline_col].iloc[0])
                            if baseline_col in window_df.columns else 1.0)
                current_mean = float(window_df[col].mean())
                anomalies.append({
                    "log_type":    col,
                    "label":       _LABELS.get(col, col),
                    "is_warning":  col in _WARNING_TYPES,
                    "max_zscore":  round(float(max_z), 2),
                    "current_mean": round(current_mean, 1),
                    "baseline":    round(baseline, 1),
                    "ratio":       round(current_mean / max(baseline, 0.1), 2),
                    "window":      window_label,
                })
        return sorted(anomalies, key=lambda x: x["max_zscore"], reverse=True)

    anomalies_24h = _extract_anomalies(last_24h, "last_24h")
    anomalies_7d  = _extract_anomalies(last_7d,  "last_7d", zscore_thr=2.0)

    # Cascade tespiti: en az 3 farklı log tipi aynı anda z-skoru > 2
    cascade_events = []
    df2["_anomaly_count"] = 0
    for col in _ALL_LOG_TYPES:
        zc = f"{col}_zscore"
        if zc in df2.columns:
            df2["_anomaly_count"] += (df2[zc] > 2.0).astype(int)

    cascade_mask = df2["_anomaly_count"] >= 3
    if cascade_mask.any():
        cascade_periods = []
        in_cascade = False
        start_ts = None
        for t, val in cascade_mask.items():
            if val and not in_cascade:
                in_cascade = True
                start_ts = t
            elif not val and in_cascade:
                in_cascade = False
                cascade_periods.append({"start": start_ts, "end": t})
        if in_cascade and start_ts is not None:
            cascade_periods.append({"start": start_ts, "end": df2.index[-1]})

        for cp in cascade_periods[-5:]:  # son 5 cascade
            duration_h = (cp["end"] - cp["start"]).total_seconds() / 3600
            cascade_events.append({
                "start":      str(cp["start"])[:19],
                "end":        str(cp["end"])[:19],
                "duration_h": round(duration_h, 1),
            })

    # Anomali zaman çizelgesi
    anomaly_timeline = df2[["_anomaly_count"]].copy()

    df2.drop(columns=["_anomaly_count"], inplace=True)

    return {
        "anomalies_last_24h": anomalies_24h,
        "anomalies_last_7d":  anomalies_7d,
        "cascade_events":     cascade_events,
        "anomaly_timeline":   anomaly_timeline,
    }


# ── Aşama 4: Ön-Arıza Örüntü Çıkarımı ───────────────────────────────────────

def stage4_prefailure_patterns(log_df: pd.DataFrame) -> list:
    """
    Geçmiş incident'lerin öncesindeki log imzalarını analiz eder.
    Her incident öncesi buildup döneminde hangi log tipleri anlamlı biçimde
    arttığını istatistiksel olarak belirler.

    Dönüş:
    -------
    [
      {
        "pattern_name":   "slow_response + connection_pool_warn",
        "lead_time_hours": 12,
        "confidence":      0.78,
        "observed_times":  6,
        "log_types":       ["slow_response_count", "connection_pool_warn_count"],
        "avg_increase_pct": 145.0,
      },
      ...
    ]
    """
    ts  = pd.DatetimeIndex(log_df["timestamp"])
    df2 = log_df.set_index(ts)

    patterns = {}  # log_type → {lead_times, increases, count}

    # Yüksek yük dönemlerini veriden otomatik tespit et (total_errors yüksek olan saatler)
    if "total_errors" not in df2.columns:
        return []

    err_series  = df2["total_errors"].fillna(0)
    err_mean    = float(err_series.mean())
    err_std     = float(err_series.std())
    spike_thr   = err_mean + 2.0 * err_std

    # Spike dönemlerini bul
    spike_mask  = err_series > spike_thr
    spike_times = df2.index[spike_mask]

    # Ardışık spike'ları gruplama
    incidents_auto = []
    if len(spike_times) > 0:
        gap = pd.Timedelta(hours=12)
        group_start = spike_times[0]
        group_end   = spike_times[0]
        for t in spike_times[1:]:
            if t - group_end <= gap:
                group_end = t
            else:
                incidents_auto.append((group_start, group_end))
                group_start = t
                group_end   = t
        incidents_auto.append((group_start, group_end))

    buildup_hours = 72
    n_incidents   = max(len(incidents_auto), 1)

    for inc_start, inc_end in incidents_auto:
        buildup_start = inc_start - pd.Timedelta(hours=buildup_hours)
        if buildup_start < df2.index[0]:
            continue

        buildup_window = df2.loc[buildup_start:inc_start]
        ref_end        = buildup_start
        ref_start      = ref_end - pd.Timedelta(hours=buildup_hours)
        ref_window     = df2.loc[ref_start:ref_end]

        if len(buildup_window) < 6 or len(ref_window) < 6:
            continue

        for col in _ALL_LOG_TYPES:
            if col not in df2.columns:
                continue
            ref_mean     = float(ref_window[col].mean())
            buildup_mean = float(buildup_window[col].mean())

            if ref_mean < 0.1:
                continue

            increase_pct = (buildup_mean - ref_mean) / ref_mean * 100.0
            if increase_pct < 30.0:
                continue

            threshold = ref_mean * 1.5
            first_exceed = None
            for t_val, v in buildup_window[col].items():
                if v > threshold:
                    first_exceed = t_val
                    break

            lead_hours = (
                (inc_start - first_exceed).total_seconds() / 3600
                if first_exceed is not None else buildup_hours
            )

            if col not in patterns:
                patterns[col] = {"lead_times": [], "increases": [], "count": 0}
            patterns[col]["lead_times"].append(lead_hours)
            patterns[col]["increases"].append(increase_pct)
            patterns[col]["count"] += 1

    result = []
    for col, data in patterns.items():
        if data["count"] < 2:
            continue
        avg_lead     = float(np.mean(data["lead_times"]))
        avg_increase = float(np.mean(data["increases"]))
        confidence   = min(data["count"] / n_incidents, 1.0)

        result.append({
            "log_type":          col,
            "label":             _LABELS.get(col, col),
            "is_warning_type":   col in _WARNING_TYPES,
            "lead_time_hours":   round(avg_lead, 1),
            "confidence":        round(confidence, 2),
            "observed_times":    data["count"],
            "avg_increase_pct":  round(avg_increase, 1),
        })

    return sorted(result, key=lambda x: (x["confidence"], x["observed_times"]), reverse=True)


# ── Trending log types (son 7 gün) ───────────────────────────────────────────

def compute_log_trends(log_df: pd.DataFrame) -> list:
    """
    Son 7 gün vs önceki 7 gündeki log tip değişimlerini hesaplar.
    LLM prompt'u için özet tablo üretir.
    """
    if len(log_df) < 336:
        return []

    prev_week = log_df.iloc[-336:-168]
    last_week = log_df.iloc[-168:]

    trends = []
    for col in _ALL_LOG_TYPES:
        if col not in log_df.columns:
            continue
        prev_sum = float(prev_week[col].sum())
        last_sum = float(last_week[col].sum())
        if prev_sum < 1.0:
            continue
        change_pct = (last_sum - prev_sum) / prev_sum * 100.0
        trends.append({
            "log_type":   col,
            "label":      _LABELS.get(col, col),
            "is_warning": col in _WARNING_TYPES,
            "last_7d":    round(last_sum),
            "prev_7d":    round(prev_sum),
            "change_pct": round(change_pct, 1),
            "direction":  "increasing" if change_pct > 5 else "decreasing" if change_pct < -5 else "stable",
        })

    return sorted(trends, key=lambda x: x["change_pct"], reverse=True)


# ── Ana pipeline fonksiyonu ───────────────────────────────────────────────────

def run_log_pipeline(log_feat_df: pd.DataFrame) -> dict:
    """
    Dört aşamalı log analiz pipeline'ını çalıştırır.

    Dönüş: LLM ve dashboard'a beslenen yapılandırılmış log içgörüleri.
    """
    print("    → Aşama 1: Ayrıştırma & Sınıflandırma")
    classified = stage1_parse_classify(log_feat_df)

    print("    → Aşama 2: Zamansal Agregasyon & Trend Analizi")
    aggregated = stage2_temporal_aggregation(classified)

    print("    → Aşama 3: Anomali Tespiti")
    anomaly_results = stage3_anomaly_detection(aggregated)

    print("    → Aşama 4: Ön-Arıza Örüntü Çıkarımı")
    prefailure_patterns = stage4_prefailure_patterns(aggregated)

    trends = compute_log_trends(log_feat_df)

    # Son 24s özet istatistikleri
    last_24 = aggregated.tail(24)
    total_warnings_24h = int(
        sum(last_24[c].sum() for c in _WARNING_TYPES if c in last_24.columns)
    )
    total_errors_24h = int(
        last_24["total_errors"].sum() if "total_errors" in last_24.columns else 0
    )
    critical_24h = int(
        last_24["critical_count"].sum() if "critical_count" in last_24.columns else 0
    )

    # Cascade bilgisi
    cascade_events = anomaly_results.get("cascade_events", [])

    # Servis sağlık durumu (log bazlı, son 24 saat)
    service_health = _compute_service_health(last_24)

    insights = {
        "anomalies_last_24h":    anomaly_results["anomalies_last_24h"],
        "anomalies_last_7d":     anomaly_results["anomalies_last_7d"][:5],
        "cascade_events_recent": cascade_events[-3:],
        "prefailure_patterns":   prefailure_patterns[:6],
        "log_trends_7d":         trends[:8],
        "total_warnings_24h":    total_warnings_24h,
        "total_errors_24h":      total_errors_24h,
        "critical_count_24h":    critical_24h,
        "service_health":        service_health,
        "aggregated_df":         aggregated,
    }

    _print_pipeline_summary(insights)

    return insights


def _compute_service_health(last_24: pd.DataFrame) -> dict:
    """Son 24 saatlik log verisiyle servis sağlık durumunu belirler."""
    services = {
        "nova_api":       "nova_api_errors",
        "nova_compute":   "nova_compute_errors",
        "nova_scheduler": "nova_scheduler_errors",
        "keystone":       "keystone_errors",
        "neutron":        "neutron_errors",
        "cinder":         "cinder_errors",
    }
    health = {}
    for svc, col in services.items():
        if col not in last_24.columns:
            health[svc] = "unknown"
            continue
        mean_err = last_24[col].mean()
        if mean_err > 50:
            health[svc] = "critical"
        elif mean_err > 20:
            health[svc] = "degraded"
        elif mean_err > 5:
            health[svc] = "warning"
        else:
            health[svc] = "healthy"
    return health


def _print_pipeline_summary(insights: dict) -> None:
    anom   = insights["anomalies_last_24h"]
    patt   = insights["prefailure_patterns"]
    trends = insights["log_trends_7d"]

    print(f"\n  Log Pipeline Özeti:")
    print(f"    Son 24s anomali  : {len(anom)} tespit  "
          f"({'en yüksek: ' + anom[0]['label'] if anom else '-'})")
    print(f"    Ön-arıza örüntüsü: {len(patt)} örüntü çıkarıldı")

    increasing = [t for t in trends if t["direction"] == "increasing"]
    if increasing:
        top = increasing[0]
        print(f"    7g trend (artış) : {top['label']} +{top['change_pct']:.0f}%")

    cascade = insights["cascade_events_recent"]
    if cascade:
        print(f"    Cascade uyarısı  : {len(cascade)} olay (son: {cascade[-1]['start']})")

    svc_issues = [s for s, h in insights["service_health"].items() if h != "healthy"]
    if svc_issues:
        print(f"    Bozuk servisler  : {', '.join(svc_issues)}")


# ── Görselleştirme ─────────────────────────────────────────────────────────────

def plot_log_pipeline_results(insights: dict) -> None:
    """Pipeline çıktılarını görselleştirir."""
    agg_df  = insights.get("aggregated_df")
    if agg_df is None or len(agg_df) == 0:
        return

    ts = pd.DatetimeIndex(agg_df["timestamp"]) if "timestamp" in agg_df.columns else agg_df.index

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Log Pipeline Analiz Sonuçları — Proaktif İzleme", fontsize=12)

    # ── Panel 1: Log risk skoru ────────────────────────────────────────────────
    ax1 = axes[0][0]
    if "log_risk_score" in agg_df.columns:
        risk_smooth = agg_df["log_risk_score"].rolling(24, min_periods=1).mean()
        ax1.fill_between(ts, risk_smooth, alpha=0.6, color="crimson", label="Log Risk Skoru (24s ort.)")
        high_risk = risk_smooth > 0.5
        ax1.fill_between(ts, risk_smooth, where=high_risk, alpha=0.3, color="orange", label="Yüksek risk (>0.5)")
        ax1.set_ylabel("Risk Skoru [0-1]")
        ax1.set_ylim(0, 1)
    ax1.set_title("Bileşik Log Risk Skoru")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.tick_params(axis="x", rotation=20, labelsize=7)
    ax1.legend(fontsize=8)

    # ── Panel 2: Erken uyarı z-skoru ─────────────────────────────────────────
    ax2 = axes[0][1]
    warn_zcols = [(c, _LABELS.get(c, c)) for c in _WARNING_TYPES[:4]
                  if f"{c}_zscore" in agg_df.columns]
    colors = ["steelblue", "darkorange", "green", "red"]
    for (col, label), color in zip(warn_zcols, colors):
        zc = f"{col}_zscore"
        smooth = agg_df[zc].rolling(24, min_periods=1).mean().clip(-3, 6)
        ax2.plot(ts, smooth, label=label[:20], color=color, linewidth=0.7, alpha=0.85)
    ax2.axhline(2.5, color="red", linestyle="--", linewidth=0.8, label="Anomali eşiği (z=2.5)")
    ax2.set_ylabel("Z-Skoru (24s ort.)")
    ax2.set_title("Erken Uyarı Log Tipleri — Z-Skoru Trendi")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.tick_params(axis="x", rotation=20, labelsize=7)

    # ── Panel 3: Ön-arıza örüntüleri ─────────────────────────────────────────
    ax3 = axes[1][0]
    patterns = insights.get("prefailure_patterns", [])
    if patterns:
        labels_p = [p["label"][:28] for p in patterns[:8]]
        confs    = [p["confidence"] for p in patterns[:8]]
        leads    = [p["lead_time_hours"] for p in patterns[:8]]
        x = np.arange(len(labels_p))
        bars = ax3.barh(x, confs, alpha=0.75, color="steelblue")
        ax3.set_yticks(x)
        ax3.set_yticklabels(labels_p, fontsize=7)
        ax3.set_xlabel("Güven Skoru")
        ax3.set_title("Ön-Arıza Log Örüntüleri (Güven & Önceden Tespit Süresi)")
        for i, (bar, lead) in enumerate(zip(bars, leads)):
            ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{lead:.0f}s önce", va="center", fontsize=7)
    else:
        ax3.text(0.5, 0.5, "Yeterli örüntü tespit edilemedi", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_xlim(0, 1.15)

    # ── Panel 4: 7 günlük log trendi ─────────────────────────────────────────
    ax4 = axes[1][1]
    trends = insights.get("log_trends_7d", [])
    if trends:
        trend_labels = [t["label"][:28] for t in trends[:10]]
        changes      = [t["change_pct"] for t in trends[:10]]
        colors_bar   = ["#d62728" if c > 0 else "#2ca02c" for c in changes]
        y = np.arange(len(trend_labels))
        ax4.barh(y, changes, color=colors_bar, alpha=0.75)
        ax4.set_yticks(y)
        ax4.set_yticklabels(trend_labels, fontsize=7)
        ax4.axvline(0, color="black", linewidth=0.8)
        ax4.set_xlabel("7 Günlük Değişim (%)")
        ax4.set_title("Log Tipi 7 Günlük Trend (kırmızı=artış, yeşil=azalış)")
    else:
        ax4.text(0.5, 0.5, "Yeterli veri yok", ha="center", va="center", transform=ax4.transAxes)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", "log_pipeline_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")
