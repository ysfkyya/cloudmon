"""
Metrik veri üretici — OpenStack küme metrikleri.

Gerçekçilik katmanları:
  1. Günlük örüntü    : kosinüs (zirve 13:00, dip 04:00)
  2. Haftalık örüntü  : hafta sonu %35 düşük yük
  3. Uzun vadeli trend: VM birikmesi → bellek/disk yavaş artış
  4. Staj dönemleri   : kademeli VM patlaması + yük artışı (trapezoid)
  5. Kapasite alımları: yeni donanım → kullanım yüzdesi ani düşüş (adım)
  6. Incident efektleri: Gauss çanı (buildup → zirve → iyileşme)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    START_DATE, END_DATE, FREQ, SEED,
    INCIDENTS, THRESHOLDS, RESULTS_DIR,
    STAJ_PERIODS, CAPACITY_EXPANSIONS,
)


# ── Yardımcı sinyal fonksiyonları ─────────────────────────────────────────────

def _daily_pattern(hour_of_day: np.ndarray) -> np.ndarray:
    """Gün içi yük: 13:00 zirve, 04:00 dip. Döndürür [0, 1]."""
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * (hour_of_day - 4.0) / 24.0))


def _weekly_pattern(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Hafta sonu yükü iş günlerine göre %35 düşük."""
    return np.where(timestamps.dayofweek.values >= 5, 0.65, 1.0)


def _trend(n: int, start_val: float, end_val: float) -> np.ndarray:
    """n noktada start_val'den end_val'e doğrusal trend."""
    return np.linspace(start_val, end_val, n)


def _staj_intensity(timestamps: pd.DatetimeIndex, staj: dict) -> np.ndarray:
    """
    Staj döneminin yoğunluk profilini hesaplar (trapezoid):
      ramp_days kademeli artış → plato → ramp_days kademeli azalış.
    Döndürür [0, 1] aralığında dizi.
    """
    t_start = pd.Timestamp(staj["start"])
    t_end   = pd.Timestamp(staj["end"])
    ramp_d  = staj.get("ramp_days", 7)
    t_up_end   = t_start + pd.Timedelta(days=ramp_d)
    t_dn_start = t_end   - pd.Timedelta(days=ramp_d)

    ts_f = timestamps.astype(np.int64).values.astype(float) / 1e9
    ss, ue = t_start.timestamp(), t_up_end.timestamp()
    ds, es = t_dn_start.timestamp(), t_end.timestamp()

    intensity = np.zeros(len(timestamps))

    up_mask = (ts_f >= ss) & (ts_f < ue)
    if up_mask.any() and (ue - ss) > 0:
        intensity[up_mask] = (ts_f[up_mask] - ss) / (ue - ss)

    intensity[(ts_f >= ue) & (ts_f < ds)] = 1.0

    dn_mask = (ts_f >= ds) & (ts_f <= es)
    if dn_mask.any() and (es - ds) > 0:
        intensity[dn_mask] = 1.0 - (ts_f[dn_mask] - ds) / (es - ds)

    return intensity


def _incident_intensity(timestamps: pd.DatetimeIndex, inc: dict) -> np.ndarray:
    """
    Incident yoğunluk profili:
      Buildup: karesel artış (max 0.30)
      Aktif  : Gauss çanı (pik'te 1.0)
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
        frac = (ts_sec[buildup_mask] - bs) / max(ss - bs, 1.0)
        intensity[buildup_mask] = frac ** 2 * 0.30

    active_mask = (ts_sec >= ss) & (ts_sec <= es)
    if active_mask.any():
        intensity[active_mask] = np.exp(
            -((ts_sec[active_mask] - ps) ** 2) / (2.0 * sigma ** 2)
        )

    return intensity


def _apply_capacity_expansions(
    arr: np.ndarray,
    timestamps: pd.DatetimeIndex,
    col: str,
) -> np.ndarray:
    """
    Kapasite genişlemesi tarihinden itibaren kullanım yüzdesi kalıcı olarak düşer.
    Gerçek sistemde: yeni donanım devreye girince toplam kapasite artar,
    aynı iş yükü daha küçük yüzde gösterir.
    """
    arr = arr.copy()
    for exp in CAPACITY_EXPANSIONS:
        reduction = exp.get(f"{col}_reduction", 0.0)
        if reduction <= 0:
            continue
        exp_idx = int(np.searchsorted(
            timestamps.astype(np.int64),
            pd.Timestamp(exp["date"]).value,
        ))
        if 0 < exp_idx < len(arr):
            arr[exp_idx:] -= reduction
    return arr


# ── Ana üretim fonksiyonu ─────────────────────────────────────────────────────

def generate_metrics() -> pd.DataFrame:
    """
    18 aylık (Oca 2025–Haz 2026) saatlik küme metrikleri üretir.

    Sütunlar
    --------
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
    load   = daily * weekly

    # ── Staj yoğunlukları (birleşik) ─────────────────────────────────────────
    staj_cpu  = np.zeros(n)
    staj_mem  = np.zeros(n)
    staj_err  = np.zeros(n)
    staj_vm   = np.zeros(n)

    for staj in STAJ_PERIODS:
        sig = _staj_intensity(timestamps, staj)
        staj_cpu += staj["cpu_extra"]   * sig
        staj_mem += staj["mem_extra"]   * sig
        staj_err += staj["error_extra"] * sig
        staj_vm  += staj["vm_burst"]    * sig

    # ── CPU (%) ───────────────────────────────────────────────────────────────
    # Baz: gece ~18%, iş saati ~42%. 18 ayda kademeli artış (+10 puan).
    cpu = (18.0 + 24.0 * load
           + _trend(n, 0, 10)
           + staj_cpu
           + np.random.normal(0, 3.5, n))
    cpu = _apply_capacity_expansions(cpu, timestamps, "cpu")

    # ── Bellek (%) ────────────────────────────────────────────────────────────
    # VM birikmesiyle sürekli artış (+26 puan/18 ay). Staj dönemlerinde ek yük.
    mem = (52.0
           + _trend(n, 0, 26)
           + 6.0 * load
           + staj_mem
           + np.random.normal(0, 2.0, n))
    mem = _apply_capacity_expansions(mem, timestamps, "mem")

    # ── Disk (%) ──────────────────────────────────────────────────────────────
    # Log/snapshot birikimi: sürekli artış (+40 puan/18 ay). Çok az varyasyon.
    disk = (42.0
            + _trend(n, 0, 40)
            + np.random.normal(0, 0.5, n))
    disk = _apply_capacity_expansions(disk, timestamps, "disk")

    # ── API İstekleri (istek/saat) ────────────────────────────────────────────
    api_req = (800.0
               + 7500.0 * load
               + _trend(n, 0, 2000)
               + staj_vm * 8.0
               + np.random.normal(0, 300, n))

    # ── API Gecikmesi (ms) — CPU ile korele ───────────────────────────────────
    latency = 55.0 + 0.65 * cpu + np.random.normal(0, 18, n)

    # ── Hata Oranı (%) ────────────────────────────────────────────────────────
    err_rate = (0.12
                + 0.008 * load
                + staj_err * 0.4
                + np.abs(np.random.normal(0, 0.06, n)))

    # ── Ağ (Mbps) ─────────────────────────────────────────────────────────────
    net_in  = 95.0  + 420.0 * load + staj_vm * 0.5 + np.random.normal(0, 30, n)
    net_out = 72.0  + 310.0 * load + staj_vm * 0.3 + np.random.normal(0, 24, n)

    # ── VM Sayısı ─────────────────────────────────────────────────────────────
    # Baz: 200 VM → 430 VM (18 ayda organik büyüme). Staj dönemlerinde patlama.
    vm_count = (200.0
                + _trend(n, 0, 230)
                + staj_vm
                + 10.0 * load
                + np.random.normal(0, 4.0, n))

    df = pd.DataFrame({
        "timestamp":              timestamps,
        "cpu_usage_percent":      cpu,
        "memory_usage_percent":   mem,
        "disk_usage_percent":     disk,
        "api_requests_per_hour":  api_req,
        "api_latency_ms":         latency,
        "error_rate_percent":     err_rate,
        "network_in_mbps":        net_in,
        "network_out_mbps":       net_out,
        "vm_count":               vm_count,
    })

    df = _apply_incidents(df, timestamps)

    # ── Geçerli aralıklara kırp ───────────────────────────────────────────────
    df["cpu_usage_percent"]     = df["cpu_usage_percent"].clip(2, 99)
    df["memory_usage_percent"]  = df["memory_usage_percent"].clip(10, 99)
    df["disk_usage_percent"]    = df["disk_usage_percent"].clip(20, 99)
    df["api_latency_ms"]        = df["api_latency_ms"].clip(10, 15000)
    df["error_rate_percent"]    = df["error_rate_percent"].clip(0, 100)
    df["api_requests_per_hour"] = df["api_requests_per_hour"].clip(50, 500000)
    df["network_in_mbps"]       = df["network_in_mbps"].clip(0, 10000)
    df["network_out_mbps"]      = df["network_out_mbps"].clip(0, 10000)
    df["vm_count"]              = df["vm_count"].clip(1, 2000).round().astype(int)

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


# ── Görselleştirme ─────────────────────────────────────────────────────────────

def plot_metrics_overview(df: pd.DataFrame) -> None:
    """6 metriğin genel görünümünü tek figure'da kaydeder."""
    n_months = round((pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days / 30)
    date_range_str = (
        f"{pd.Timestamp(START_DATE).strftime('%b %Y')} – "
        f"{pd.Timestamp(END_DATE).strftime('%b %Y')}"
    )

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        f"OpenStack Küme Metrikleri — {n_months} Aylık Sentetik Veri Seti ({date_range_str})",
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
        ax.plot(df["timestamp"], df[col], color=color, linewidth=0.4, alpha=0.85)
        ax.set_title(label, fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        # Staj dönemleri (mavi gölge)
        for staj in STAJ_PERIODS:
            ax.axvspan(pd.Timestamp(staj["start"]), pd.Timestamp(staj["end"]),
                       alpha=0.08, color="royalblue", linewidth=0, label="_staj")

        # Kapasite genişleme tarihleri (yeşil dikey çizgi)
        for exp in CAPACITY_EXPANSIONS:
            ax.axvline(pd.Timestamp(exp["date"]),
                       color="green", linestyle=":", linewidth=0.8, alpha=0.7)

        # Incident dönemleri (kırmızı gölge)
        for inc in INCIDENTS:
            shade_start = pd.Timestamp(inc["start"]) - pd.Timedelta(days=inc["buildup_days"])
            ax.axvspan(shade_start, pd.Timestamp(inc["end"]),
                       alpha=0.09, color="red", linewidth=0)

        if col in THRESHOLDS:
            ax.axhline(THRESHOLDS[col], color="red", linestyle="--",
                       linewidth=0.9, label=f"Eşik ({THRESHOLDS[col]})")
            ax.legend(fontsize=7)

        ax.tick_params(axis="x", rotation=20, labelsize=7)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", "metrics_overview.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")
