"""
Metrik veri üretici — OpenStack küme metrikleri.

vCPU kapasite senaryosu:
  Toplam 2000 vCPU | Ocak 2025: 30 kullanımda → Temmuz 2026: 1598 kullanımda
  Mevcut hızda ~8 ay sonra (Mart 2027) tamamen dolacak.

Gerçekçilik katmanları:
  1. İki fazlı vCPU büyüme trendi (Faz1 hızlı, Faz2 daha yavaş ama hâlâ artış)
  2. Günlük / haftalık örüntü (load sinyali — tüm metrikler bu ile korele)
  3. Staj dönemleri: geçici VM patlaması + tüm kaynaklarda ek yük
  4. Incident efektleri: Gauss buildup → zirve → iyileşme
  5. Kaynak korelasyonu: bellek ve disk vcpu_used'e bağlı trendler
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
    VCPU_TOTAL, VCPU_START, VCPU_END,
    MEMORY_TOTAL_GB, MEMORY_START_GB, MEMORY_END_GB,
    DISK_TOTAL_TB, DISK_START_TB, DISK_END_TB,
    INCIDENTS, THRESHOLDS, RESULTS_DIR,
    STAJ_PERIODS,
)


def _daily_pattern(hour_of_day: np.ndarray) -> np.ndarray:
    """Gün içi yük: 13:00 zirve, 04:00 dip. Döndürür [0, 1]."""
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * (hour_of_day - 4.0) / 24.0))


def _weekly_pattern(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Hafta sonu yükü iş günlerine göre %35 düşük."""
    return np.where(timestamps.dayofweek.values >= 5, 0.65, 1.0)


def _staj_intensity(timestamps: pd.DatetimeIndex, staj: dict) -> np.ndarray:
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


def generate_metrics() -> pd.DataFrame:
    """
    18 aylık saatlik küme metrikleri üretir.

    vCPU hikayesi:
      - Toplam kapasite: 2000 vCPU
      - Ocak 2025: 30 vCPU kullanımda
      - Temmuz 2026: 1598 vCPU kullanımda
      - Mevcut trend: ~8 ay sonra tamamen dolacak

    Sütunlar:
      timestamp,
      vcpu_used, vcpu_total, vcpu_usage_percent,
      memory_used_gb, memory_total_gb, memory_usage_percent,
      disk_used_tb, disk_total_tb, disk_usage_percent,
      api_requests_per_hour, api_latency_ms, error_rate_percent,
      network_in_mbps, network_out_mbps, vm_count
    """
    np.random.seed(SEED)
    timestamps = pd.date_range(START_DATE, END_DATE, freq=FREQ, inclusive="left")
    n = len(timestamps)

    hod    = timestamps.hour.values.astype(float)
    daily  = _daily_pattern(hod)
    weekly = _weekly_pattern(timestamps)
    load   = daily * weekly   # genel yük sinyali [0, 1]

    # ── İki Fazlı Büyüme Trendi (vCPU, bellek, disk için ortak) ─────────────
    # Faz 1 (12 ay): hızlı büyüme   → Faz 2 (6 ay): yavaş ama sürekli artış
    # Bu hızda:
    #   vCPU  : kalan 402 vCPU ≈ 8 ay sonra dolacak
    #   Bellek: kalan ~1280 GB  ≈ 10 ay sonra eşiğe ulaşacak
    #   Disk  : kalan ~110 TB   ≈ 12 ay sonra eşiğe ulaşacak
    phase1_h = 12 * 30 * 24  # 12 ay ≈ 8640 saat
    phase1_h = min(phase1_h, n)

    def _two_phase_trend(start_val, phase1_end_val, end_val):
        arr = np.zeros(n)
        for i in range(n):
            if i < phase1_h:
                arr[i] = start_val + (phase1_end_val - start_val) * (i / phase1_h)
            else:
                arr[i] = phase1_end_val + (end_val - phase1_end_val) * (
                    (i - phase1_h) / max(n - phase1_h, 1)
                )
        return arr

    vcpu_base = _two_phase_trend(VCPU_START, 1290.0, VCPU_END)

    # Bellek büyümesi: 2450 GB → 5200 GB (faz1) → 6720 GB
    mem_base_gb  = _two_phase_trend(MEMORY_START_GB, 5200.0, MEMORY_END_GB)

    # Disk büyümesi: 140 TB → 320 TB (faz1) → 390 TB
    disk_base_tb = _two_phase_trend(DISK_START_TB, 320.0, DISK_END_TB)

    # ── Staj etkileri (vCPU, bellek, disk, hata) ─────────────────────────────
    staj_vcpu = np.zeros(n)
    staj_mem  = np.zeros(n)
    staj_disk = np.zeros(n)
    staj_err  = np.zeros(n)
    staj_vm   = np.zeros(n)

    for staj in STAJ_PERIODS:
        sig = _staj_intensity(timestamps, staj)
        staj_vcpu += staj.get("vcpu_extra", staj.get("vm_burst", 0) * 2) * sig
        staj_mem  += staj["mem_extra"]   * sig
        staj_disk += staj.get("disk_extra", staj["mem_extra"] * 0.5) * sig
        staj_err  += staj["error_extra"] * sig
        staj_vm   += staj["vm_burst"]    * sig

    # ── vCPU kullanımı (tahsis edilen, organik + staj + gürültü) ─────────────
    vcpu_used = (vcpu_base
                 + staj_vcpu * 0.8          # staj süresince ek vCPU tahsisi
                 + 30.0 * load              # günlük yük dalgası (kısa vadeli)
                 + np.random.normal(0, 12, n))
    vcpu_used = np.clip(vcpu_used, 1, VCPU_TOTAL - 1)

    vcpu_usage_percent = vcpu_used / VCPU_TOTAL * 100.0

    # ── Bellek (GB ve %) — iki fazlı trend + staj + yük dalgası ─────────────
    # Faz1: 2450→5200 GB, Faz2: 5200→6720 GB + staj/yük dalgası + gürültü
    memory_used_gb = (
        mem_base_gb
        + staj_mem * 0.9 * (MEMORY_TOTAL_GB / 100.0)   # staj ek bellek (GB cinsinden)
        + 4.0 * load * (MEMORY_TOTAL_GB / 100.0)         # günlük yük dalgası (GB)
        + np.random.normal(0, 60, n)                      # gürültü
    ).clip(MEMORY_START_GB * 0.8, MEMORY_TOTAL_GB * 0.99)

    memory_usage_percent = (memory_used_gb / MEMORY_TOTAL_GB * 100.0).clip(10, 99)

    # ── Disk (TB ve %) — iki fazlı trend + staj + log birikimi ─────────────
    # Faz1: 140→320 TB, Faz2: 320→390 TB + staj + log/snapshot birikimi
    log_accum_tb = np.linspace(0, 8, n)  # 18 ayda +8 TB log/snapshot birikimi
    disk_used_tb = (
        disk_base_tb
        + log_accum_tb
        + staj_disk * 0.7 * (DISK_TOTAL_TB / 100.0)     # staj ek disk
        + np.random.normal(0, 1.5, n)
    ).clip(DISK_START_TB * 0.9, DISK_TOTAL_TB * 0.99)

    disk_usage_percent = (disk_used_tb / DISK_TOTAL_TB * 100.0).clip(15, 99)

    # ── API İstekleri (istek/saat) — VM sayısı ile büyüyen ───────────────────
    api_req = (
        500.0
        + 6000.0 * load
        + (vcpu_used / VCPU_TOTAL) * 3000.0
        + staj_vm * 6.0
        + np.random.normal(0, 250, n)
    )

    # ── API Gecikmesi (ms) — bellek baskısı + yük ────────────────────────────
    db_pressure = np.clip((memory_usage_percent - 65.0) / 35.0, 0, 1)
    net_jitter  = 35.0 * load * np.abs(np.random.normal(0, 1, n))
    api_latency_ms = (
        45.0
        + 0.35 * vcpu_usage_percent    # yüksek vCPU tahsisi → kuyruk gecikmesi
        + 155.0 * db_pressure          # bellek baskısı → DB yavaşlaması
        + net_jitter
        + np.random.normal(0, 16, n)
    )

    # ── Hata Oranı (%) ────────────────────────────────────────────────────────
    resource_pressure = np.clip((vcpu_usage_percent - 50.0) / 50.0, 0, 1)
    error_rate_percent = (
        0.10
        + 0.015 * load
        + 0.8 * resource_pressure      # kaynaklar dolunca hata artar
        + staj_err * 0.35
        + np.abs(np.random.normal(0, 0.08, n))
    )

    # ── Ağ (Mbps) — VM sayısı ile büyüyen ────────────────────────────────────
    net_in  = (80.0
               + 380.0 * load
               + (vcpu_used / VCPU_TOTAL) * 200.0
               + staj_vm * 0.4
               + np.random.normal(0, 25, n))
    net_out = (60.0
               + 280.0 * load
               + (vcpu_used / VCPU_TOTAL) * 150.0
               + staj_vm * 0.25
               + np.random.normal(0, 20, n))

    # ── VM Sayısı (vcpu_used / ortalama flavour vCPU sayısı) ─────────────────
    vm_count = (vcpu_used / 2.0 + np.random.normal(0, 3, n)).clip(1, VCPU_TOTAL // 2)

    df = pd.DataFrame({
        "timestamp":              timestamps,
        # vCPU
        "vcpu_used":              vcpu_used,
        "vcpu_total":             np.full(n, VCPU_TOTAL),
        "vcpu_usage_percent":     vcpu_usage_percent,
        # Bellek
        "memory_used_gb":         memory_used_gb,
        "memory_total_gb":        np.full(n, float(MEMORY_TOTAL_GB)),
        "memory_usage_percent":   memory_usage_percent,
        # Disk
        "disk_used_tb":           disk_used_tb,
        "disk_total_tb":          np.full(n, float(DISK_TOTAL_TB)),
        "disk_usage_percent":     disk_usage_percent,
        # Diğer metrikler
        "api_requests_per_hour":  api_req,
        "api_latency_ms":         api_latency_ms,
        "error_rate_percent":     error_rate_percent,
        "network_in_mbps":        net_in,
        "network_out_mbps":       net_out,
        "vm_count":               vm_count,
    })

    df = _apply_incidents(df, timestamps)

    # ── Geçerli aralıklara kırp ───────────────────────────────────────────────
    df["vcpu_used"]             = df["vcpu_used"].clip(1, VCPU_TOTAL)
    df["vcpu_usage_percent"]    = df["vcpu_usage_percent"].clip(0.5, 99.9)
    df["memory_used_gb"]        = df["memory_used_gb"].clip(100, MEMORY_TOTAL_GB)
    df["memory_usage_percent"]  = df["memory_usage_percent"].clip(1, 99)
    df["disk_used_tb"]          = df["disk_used_tb"].clip(10, DISK_TOTAL_TB)
    df["disk_usage_percent"]    = df["disk_usage_percent"].clip(2, 99)
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
        vcpu_spike     = inc.get("vcpu_spike", 0.0)
        vcpu_spike_pct = vcpu_spike / VCPU_TOTAL * 100.0
        mem_spike_pct  = inc["memory_spike"]   # %
        disk_spike_pct = inc["disk_spike"]      # %

        df["vcpu_used"]            += vcpu_spike     * intensity
        df["vcpu_usage_percent"]   += vcpu_spike_pct * intensity
        df["memory_used_gb"]       += mem_spike_pct  / 100.0 * MEMORY_TOTAL_GB * intensity
        df["memory_usage_percent"] += mem_spike_pct  * intensity
        df["disk_used_tb"]         += disk_spike_pct / 100.0 * DISK_TOTAL_TB * (intensity > 0.05).astype(float)
        df["disk_usage_percent"]   += disk_spike_pct * (intensity > 0.05).astype(float)
        df["api_latency_ms"]       += inc["latency_spike"] * intensity
        df["error_rate_percent"]   += inc["error_spike"]   * intensity
    return df


def plot_metrics_overview(df: pd.DataFrame) -> None:
    """Temel kapasite metriklerini tek figure'da kaydeder."""
    n_months = round((pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days / 30)
    date_range_str = (
        f"{pd.Timestamp(START_DATE).strftime('%b %Y')} – "
        f"{pd.Timestamp(END_DATE).strftime('%b %Y')}"
    )

    mem_start = int(df["memory_used_gb"].iloc[0])
    mem_end   = int(df["memory_used_gb"].iloc[-1])
    disk_start = round(df["disk_used_tb"].iloc[0], 1)
    disk_end   = round(df["disk_used_tb"].iloc[-1], 1)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        f"OpenStack Küme Metrikleri — {n_months} Aylık Sentetik Veri ({date_range_str})\n"
        f"vCPU: {VCPU_START}→{VCPU_END}/{VCPU_TOTAL} | "
        f"Bellek: {mem_start}→{mem_end}/{MEMORY_TOTAL_GB}GB | "
        f"Disk: {disk_start}→{disk_end}/{DISK_TOTAL_TB}TB",
        fontsize=11, y=1.01,
    )

    panels = [
        ("vcpu_used",            f"vCPU Kullanımı (/{VCPU_TOTAL})",             "steelblue"),
        ("memory_used_gb",       f"Bellek Kullanımı (GB / {MEMORY_TOTAL_GB}GB)", "darkorange"),
        ("disk_used_tb",         f"Disk Kullanımı (TB / {DISK_TOTAL_TB}TB)",     "seagreen"),
        ("api_latency_ms",       "API Gecikmesi (ms)",                           "crimson"),
        ("error_rate_percent",   "Hata Oranı (%)",                               "purple"),
        ("vm_count",             "Aktif VM Sayısı",                              "saddlebrown"),
    ]

    for ax, (col, label, color) in zip(axes.flat, panels):
        ax.plot(df["timestamp"], df[col], color=color, linewidth=0.4, alpha=0.85)
        ax.set_title(label, fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        for staj in STAJ_PERIODS:
            ax.axvspan(pd.Timestamp(staj["start"]), pd.Timestamp(staj["end"]),
                       alpha=0.08, color="royalblue", linewidth=0, label="_staj")

        for inc in INCIDENTS:
            shade_start = pd.Timestamp(inc["start"]) - pd.Timedelta(days=inc["buildup_days"])
            ax.axvspan(shade_start, pd.Timestamp(inc["end"]),
                       alpha=0.09, color="red", linewidth=0)

        if col in THRESHOLDS:
            ax.axhline(THRESHOLDS[col], color="red", linestyle="--",
                       linewidth=0.9, label=f"Eşik ({THRESHOLDS[col]})")
            ax.legend(fontsize=7)

        # Projeksiyon etiketleri ekle (vCPU, bellek, disk için)
        if col in ("vcpu_used", "memory_used_gb", "disk_used_tb"):
            last_val = df[col].iloc[-1]
            last_ts  = df["timestamp"].iloc[-1]
            window_h = 24 * 30
            if len(df) > window_h:
                slope = (last_val - df[col].iloc[-window_h]) / window_h
                totals = {
                    "vcpu_used":      VCPU_TOTAL,
                    "memory_used_gb": MEMORY_TOTAL_GB,
                    "disk_used_tb":   DISK_TOTAL_TB,
                }
                total = totals[col]
                for months_ahead in [3, 6]:
                    proj_val = last_val + slope * 24 * 30 * months_ahead
                    proj_ts  = last_ts + pd.Timedelta(days=30 * months_ahead)
                    pct      = proj_val / total * 100
                    ax.annotate(
                        f"+{months_ahead}ay\n%{pct:.0f}",
                        xy=(proj_ts, proj_val),
                        fontsize=6, color="red", alpha=0.7,
                    )

        ax.tick_params(axis="x", rotation=20, labelsize=7)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", "metrics_overview.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")
