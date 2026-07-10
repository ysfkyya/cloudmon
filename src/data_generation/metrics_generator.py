"""
Metrik veri üretici.
OpenStack cluster'ından alınan 6 aylık saatlik sistem metriklerini
gerçekçi günlük/haftalık örüntüler, uzun vadeli trendler ve
olay (incident) efektleriyle sentezler.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import START_DATE, END_DATE, FREQ, SEED, INCIDENTS, THRESHOLDS, RESULTS_DIR


# ── Yardımcı sinyal fonksiyonları ────────────────────────────────────────────

def _daily_pattern(hour_of_day: np.ndarray) -> np.ndarray:
    """
    Gün içi yük örüntüsü: 13:00'te zirve, 04:00'te dip.
    Döndürülen değer [0, 1] aralığındadır.
    """
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * (hour_of_day - 4.0) / 24.0))


def _weekly_pattern(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Hafta sonu yükü iş günlerine göre %35 daha düşük."""
    dow = timestamps.dayofweek.values  # 0=Pzt … 6=Paz
    return np.where(dow >= 5, 0.65, 1.0)


def _trend(n: int, start_val: float, end_val: float) -> np.ndarray:
    """n noktada start_val'den end_val'e doğrusal trend."""
    return np.linspace(start_val, end_val, n)


def _incident_intensity(timestamps: pd.DatetimeIndex, inc: dict) -> np.ndarray:
    """
    Bir olayın zaman ekseni üzerindeki yoğunluk profilini vektörize olarak hesaplar.

    Buildup dönemi: karesel artış (max 0.30)
    Aktif dönem   : Gauss çanı (peak'te 1.0)
    """
    buildup_start = pd.Timestamp(inc["start"]) - pd.Timedelta(days=inc["buildup_days"])
    t_start = pd.Timestamp(inc["start"])
    t_peak  = pd.Timestamp(inc["peak"])
    t_end   = pd.Timestamp(inc["end"])

    ts_sec = timestamps.astype("int64").values / 1e9
    bs = buildup_start.timestamp()
    ss = t_start.timestamp()
    ps = t_peak.timestamp()
    es = t_end.timestamp()
    sigma = max((es - ss) / 4.0, 1.0)

    intensity = np.zeros(len(timestamps))

    buildup_mask = (ts_sec >= bs) & (ts_sec < ss)
    if buildup_mask.any():
        buildup_dur = ss - bs
        frac = (ts_sec[buildup_mask] - bs) / buildup_dur
        intensity[buildup_mask] = frac ** 2 * 0.30

    active_mask = (ts_sec >= ss) & (ts_sec <= es)
    if active_mask.any():
        intensity[active_mask] = np.exp(-((ts_sec[active_mask] - ps) ** 2) / (2.0 * sigma ** 2))

    return intensity


# ── Ana üretim fonksiyonu ─────────────────────────────────────────────────────

def generate_metrics() -> pd.DataFrame:
    """
    6 aylık (Ocak–Haziran 2025) saatlik küme düzeyi metrikleri üretir.

    Üretilen sütunlar
    -----------------
    timestamp, cpu_usage_percent, memory_usage_percent, disk_usage_percent,
    api_requests_per_hour, api_latency_ms, error_rate_percent,
    network_in_mbps, network_out_mbps, vm_count
    """
    np.random.seed(SEED)
    timestamps = pd.date_range(START_DATE, END_DATE, freq=FREQ, inclusive="left")
    n = len(timestamps)

    hod    = timestamps.hour.values.astype(float)
    daily  = _daily_pattern(hod)
    weekly = _weekly_pattern(timestamps)
    load   = daily * weekly  # birleşik günlük×haftalık faktör

    # ── CPU (%) ──────────────────────────────────────────────────────────────
    # Temel: gece %20, iş saatlerinde %50'ye kadar
    cpu = 20.0 + 30.0 * load + _trend(n, 0, 5) + np.random.normal(0, 3.5, n)

    # ── Bellek (%) ───────────────────────────────────────────────────────────
    # VM birikmesiyle yavaş artış trendi; 6 ayda +18 puan
    memory = 46.0 + _trend(n, 0, 18) + 5.0 * load + np.random.normal(0, 2.0, n)

    # ── Disk (%) ─────────────────────────────────────────────────────────────
    # Log ve snapshot birikimi: 6 ayda +32 puan; çok az varyasyon
    disk = 40.0 + _trend(n, 0, 32) + np.random.normal(0, 0.5, n)

    # ── API İstekleri (istek/saat) ───────────────────────────────────────────
    api_req = 700.0 + 6500.0 * load + _trend(n, 0, 1200) + np.random.normal(0, 260, n)

    # ── API Gecikmesi (ms) ───────────────────────────────────────────────────
    # CPU ile korele: yüksek CPU → daha fazla gecikme
    latency = 55.0 + 0.65 * cpu + np.random.normal(0, 18, n)

    # ── Hata Oranı (%) ───────────────────────────────────────────────────────
    err_rate = 0.15 + 0.010 * load + np.abs(np.random.normal(0, 0.06, n))

    # ── Ağ (Mbps) ────────────────────────────────────────────────────────────
    net_in  = 85.0  + 370.0 * load + np.random.normal(0, 28, n)
    net_out = 65.0  + 275.0 * load + np.random.normal(0, 22, n)

    # ── VM Sayısı ────────────────────────────────────────────────────────────
    vm_count = 48.0 + _trend(n, 0, 78) + 9.0 * load + np.random.normal(0, 2.5, n)

    df = pd.DataFrame({
        "timestamp":              timestamps,
        "cpu_usage_percent":      cpu,
        "memory_usage_percent":   memory,
        "disk_usage_percent":     disk,
        "api_requests_per_hour":  api_req,
        "api_latency_ms":         latency,
        "error_rate_percent":     err_rate,
        "network_in_mbps":        net_in,
        "network_out_mbps":       net_out,
        "vm_count":               vm_count,
    })

    df = _apply_incidents(df, timestamps)

    # Geçerli aralıklara kırp
    df["cpu_usage_percent"]     = df["cpu_usage_percent"].clip(2, 99)
    df["memory_usage_percent"]  = df["memory_usage_percent"].clip(10, 99)
    df["disk_usage_percent"]    = df["disk_usage_percent"].clip(20, 99)
    df["api_latency_ms"]        = df["api_latency_ms"].clip(10, 15000)
    df["error_rate_percent"]    = df["error_rate_percent"].clip(0, 100)
    df["api_requests_per_hour"] = df["api_requests_per_hour"].clip(50, 200000)
    df["network_in_mbps"]       = df["network_in_mbps"].clip(0, 10000)
    df["network_out_mbps"]      = df["network_out_mbps"].clip(0, 10000)
    df["vm_count"]              = df["vm_count"].clip(1, 600).round().astype(int)

    return df


def _apply_incidents(df: pd.DataFrame, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    for inc in INCIDENTS:
        intensity = _incident_intensity(timestamps, inc)
        df["cpu_usage_percent"]    += inc["cpu_spike"]     * intensity
        df["memory_usage_percent"] += inc["memory_spike"]  * intensity
        df["disk_usage_percent"]   += inc["disk_spike"]    * (intensity > 0.05).astype(float)
        df["api_latency_ms"]       += inc["latency_spike"] * intensity
        df["error_rate_percent"]   += inc["error_spike"]   * intensity
    return df


# ── Görselleştirme ────────────────────────────────────────────────────────────

def plot_metrics_overview(df: pd.DataFrame) -> None:
    """6 metriğin genel görünümünü tek figure'da kaydeder."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        "OpenStack Küme Metrikleri — 6 Aylık Sentetik Veri Seti (2025)",
        fontsize=13, y=1.01,
    )

    panels = [
        ("cpu_usage_percent",    "CPU Kullanımı (%)",       "steelblue"),
        ("memory_usage_percent", "Bellek Kullanımı (%)",    "darkorange"),
        ("disk_usage_percent",   "Disk Kullanımı (%)",      "seagreen"),
        ("api_latency_ms",       "API Gecikmesi (ms)",      "crimson"),
        ("error_rate_percent",   "Hata Oranı (%)",          "purple"),
        ("vm_count",             "VM Sayısı",               "saddlebrown"),
    ]

    for ax, (col, label, color) in zip(axes.flat, panels):
        ax.plot(df["timestamp"], df[col], color=color, linewidth=0.5, alpha=0.85)
        ax.set_title(label, fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

        for inc in INCIDENTS:
            shade_start = pd.Timestamp(inc["start"]) - pd.Timedelta(days=inc["buildup_days"])
            ax.axvspan(shade_start, pd.Timestamp(inc["end"]),
                       alpha=0.10, color="red", linewidth=0)

        if col in THRESHOLDS:
            ax.axhline(
                THRESHOLDS[col], color="red", linestyle="--",
                linewidth=0.9, label=f"Eşik ({THRESHOLDS[col]})",
            )
            ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", "metrics_overview.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")
