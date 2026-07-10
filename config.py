import os

# ── Zaman aralığı ────────────────────────────────────────────────────────────
START_DATE = "2025-09-01"
END_DATE   = "2026-07-01"   # exclusive; 10 tam ay → ~7 296 saatlik nokta
FREQ       = "h"

# ── Genel ayarlar ────────────────────────────────────────────────────────────
SEED      = 42
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── Kritik eşik değerleri ─────────────────────────────────────────────────────
THRESHOLDS = {
    "cpu_usage_percent":    85.0,
    "memory_usage_percent": 88.0,
    "disk_usage_percent":   82.0,
    "api_latency_ms":      1200.0,
    "error_rate_percent":    5.0,
}

# ── ML kapasite tahmini parametreleri ────────────────────────────────────────
FORECAST_HORIZONS = [24, 72]          # tahmin edilecek zaman ufukları (saat)
CAPACITY_TARGETS  = [
    "cpu_usage_percent",
    "memory_usage_percent",
    "disk_usage_percent",
]

# ── Sağlık sınıflandırması parametreleri ──────────────────────────────────────
RISK_HORIZON   = 4       # kaç saat sonraki riski tahmin ediyoruz
TRAIN_END_DATE = "2026-04-01"   # train: Eyl 2025–Mar 2026 (7 ay), test: Nis–Haz 2026 (3 ay)

# ── 10 Aylık Incident (Olay) Tanımları ───────────────────────────────────────
# buildup_days : olaydan kaç gün önce göstergeler bozulmaya başlar
# *_spike      : olay zirvesinde metriğe eklenecek maksimum değer (yüzde veya ms)
INCIDENTS = [
    {
        "name": "Bellek Baskısı — Hesaplama Grubu A",
        "start": "2025-09-20",
        "peak":  "2025-09-22",
        "end":   "2025-09-26",
        "buildup_days": 5,
        "cpu_spike":     12.0,
        "memory_spike":  38.0,
        "disk_spike":     0.0,
        "latency_spike": 420.0,
        "error_spike":    8.5,
    },
    {
        "name": "Disk Kapasitesi Kritik Seviyeye Yaklaşıyor",
        "start": "2025-10-18",
        "peak":  "2025-10-21",
        "end":   "2025-10-25",
        "buildup_days": 7,
        "cpu_spike":     6.0,
        "memory_spike":  8.0,
        "disk_spike":   24.0,
        "latency_spike": 260.0,
        "error_spike":    5.5,
    },
    {
        "name": "Yüksek CPU Yükü Olayı",
        "start": "2025-11-12",
        "peak":  "2025-11-15",
        "end":   "2025-11-19",
        "buildup_days": 3,
        "cpu_spike":    48.0,
        "memory_spike": 18.0,
        "disk_spike":    0.0,
        "latency_spike": 920.0,
        "error_spike":   14.0,
    },
    {
        "name": "Yılsonu Trafik Artışı",
        "start": "2025-12-26",
        "peak":  "2025-12-28",
        "end":   "2026-01-02",
        "buildup_days": 2,
        "cpu_spike":    30.0,
        "memory_spike": 22.0,
        "disk_spike":    5.0,
        "latency_spike": 1100.0,
        "error_spike":   10.0,
    },
    {
        "name": "Veritabanı Bağlantı Fırtınası",
        "start": "2026-01-22",
        "peak":  "2026-01-25",
        "end":   "2026-01-29",
        "buildup_days": 2,
        "cpu_spike":    22.0,
        "memory_spike": 16.0,
        "disk_spike":    0.0,
        "latency_spike": 3400.0,
        "error_spike":   30.0,
    },
    {
        "name": "Ağ Katmanı Kesintisi",
        "start": "2026-02-12",
        "peak":  "2026-02-14",
        "end":   "2026-02-17",
        "buildup_days": 3,
        "cpu_spike":    10.0,
        "memory_spike":  8.0,
        "disk_spike":    0.0,
        "latency_spike": 2600.0,
        "error_spike":   22.0,
    },
    {
        "name": "API Gecikme Bozunumu",
        "start": "2026-03-18",
        "peak":  "2026-03-20",
        "end":   "2026-03-24",
        "buildup_days": 4,
        "cpu_spike":    26.0,
        "memory_spike": 10.0,
        "disk_spike":    0.0,
        "latency_spike": 2200.0,
        "error_spike":   17.0,
    },
    {
        "name": "Zamanlayıcı Kaynak Yetersizliği",
        "start": "2026-04-20",
        "peak":  "2026-04-23",
        "end":   "2026-04-27",
        "buildup_days": 3,
        "cpu_spike":    35.0,
        "memory_spike": 25.0,
        "disk_spike":    8.0,
        "latency_spike": 1500.0,
        "error_spike":   20.0,
    },
    {
        "name": "Çok Kaynaklı Basamaklı Arıza",
        "start": "2026-05-10",
        "peak":  "2026-05-13",
        "end":   "2026-05-17",
        "buildup_days": 5,
        "cpu_spike":    45.0,
        "memory_spike": 34.0,
        "disk_spike":   14.0,
        "latency_spike": 3000.0,
        "error_spike":   35.0,
    },
    {
        "name": "Keystone Kimlik Doğrulama Krizi",
        "start": "2026-06-15",
        "peak":  "2026-06-17",
        "end":   "2026-06-21",
        "buildup_days": 3,
        "cpu_spike":    18.0,
        "memory_spike": 12.0,
        "disk_spike":    0.0,
        "latency_spike": 1800.0,
        "error_spike":   24.0,
    },
]
