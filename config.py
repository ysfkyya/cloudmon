import os

# ── Genel ayarlar ─────────────────────────────────────────────────────────────
SEED          = 42
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
SYNTHETIC_DIR = os.path.join(BASE_DIR, "data", "synthetic")   # Üretilen veri — gerçek veriyle değiştir
RESULTS_DIR   = os.path.join(BASE_DIR, "results")

# ── .env dosyasından ortam değişkenlerini yükle ───────────────────────────────
_env_file = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_file):
    with open(_env_file, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── API Anahtarları ───────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY",    "")

# ── Zaman aralığı ─────────────────────────────────────────────────────────────
START_DATE     = "2025-01-01"
END_DATE       = "2026-07-01"   # 18 ay → ~13 140 saatlik nokta
FREQ           = "h"
TRAIN_END_DATE = "2026-04-01"   # Eğitim: Oca 2025–Mar 2026 (15 ay), Test: Nis–Haz 2026 (3 ay)

# ── Kapasite senaryosu (mutlak değerler) ─────────────────────────────────────
# vCPU  : 2000 toplam | Ocak 2025: 30 → Temmuz 2026: 1598 | ~8 ay sonra doluyor
# Bellek: 8000 GB toplam | ~2450 GB → ~6700 GB | ~10 ay sonra eşiğe ulaşacak
# Disk  : 500 TB toplam  | ~140 TB  → ~390 TB  | ~12 ay sonra eşiğe ulaşacak

VCPU_TOTAL     = 2000    # toplam fiziksel vCPU kapasitesi
VCPU_START     = 30      # Ocak 2025 başı
VCPU_END       = 1598    # Temmuz 2026 sonu

MEMORY_TOTAL_GB = 8000   # toplam küme RAM kapasitesi (GB)
MEMORY_START_GB = 2450   # Ocak 2025 başı (~30.6%)
MEMORY_END_GB   = 6720   # Temmuz 2026 sonu (~84.0%)

DISK_TOTAL_TB   = 500    # toplam depolama kapasitesi (TB)
DISK_START_TB   = 140    # Ocak 2025 başı (~28.0%)
DISK_END_TB     = 390    # Temmuz 2026 sonu (~78.0%)

# ── Kritik eşik değerleri ─────────────────────────────────────────────────────
THRESHOLDS = {
    "vcpu_usage_percent":   90.0,   # vCPU tahsis oranı eşiği
    "memory_usage_percent": 88.0,
    "disk_usage_percent":   85.0,
    "api_latency_ms":      1200.0,
    "error_rate_percent":    5.0,
}

# ── ML kapasite tahmini parametreleri ─────────────────────────────────────────
FORECAST_HORIZONS = [24, 72]               # 1 gün / 3 gün
CAPACITY_TARGETS  = [
    "vcpu_usage_percent",
    "memory_usage_percent",
    "disk_usage_percent",
]

# ── Sağlık riski sınıflandırması ──────────────────────────────────────────────
RISK_HORIZON = 4   # sonraki N saatte herhangi bir eşik aşılacak mı? (binary hedef)

# ── Staj Dönemleri ────────────────────────────────────────────────────────────
STAJ_PERIODS = [
    {
        "name":        "Kış Stajı 2025",
        "start":       "2025-02-01",
        "end":         "2025-03-14",
        "vm_burst":    80,
        "vcpu_extra":  160,
        "mem_extra":   8.0,
        "disk_extra":  4.0,
        "error_extra": 2.5,
        "ramp_days":   7,
    },
    {
        "name":        "Yaz Stajı 2025",
        "start":       "2025-06-01",
        "end":         "2025-09-12",
        "vm_burst":    185,
        "vcpu_extra":  370,
        "mem_extra":   15.0,
        "disk_extra":  8.0,
        "error_extra": 3.8,
        "ramp_days":   14,
    },
    {
        "name":        "Kış Stajı 2025-2026",
        "start":       "2026-01-10",
        "end":         "2026-02-27",
        "vm_burst":    105,
        "vcpu_extra":  210,
        "mem_extra":   10.0,
        "disk_extra":  5.0,
        "error_extra": 3.0,
        "ramp_days":   7,
    },
    {
        "name":        "Yaz Stajı 2026",
        "start":       "2026-06-01",
        "end":         "2026-07-01",
        "vm_burst":    155,
        "vcpu_extra":  310,
        "mem_extra":   13.0,
        "disk_extra":  7.0,
        "error_extra": 3.5,
        "ramp_days":   14,
    },
]

# ── Incident Tanımları (15 olay — 18 ay boyunca) ─────────────────────────────
INCIDENTS = [
    {
        "name":          "Kış Stajı Keystone Kimlik Aşırı Yükü",
        "start":         "2025-02-06",
        "peak":          "2025-02-08",
        "end":           "2025-02-12",
        "buildup_days":  3,
        "vcpu_spike":    80.0,
        "memory_spike":  14.0,
        "disk_spike":     0.0,
        "latency_spike": 680.0,
        "error_spike":    8.0,
    },
    {
        "name":          "Nova Zamanlayıcı Tıkanması",
        "start":         "2025-03-18",
        "peak":          "2025-03-20",
        "end":           "2025-03-23",
        "buildup_days":  2,
        "vcpu_spike":    120.0,
        "memory_spike":   8.0,
        "disk_spike":     0.0,
        "latency_spike": 540.0,
        "error_spike":    6.0,
    },
    {
        "name":          "Disk G/Ç Yavaşlaması",
        "start":         "2025-05-02",
        "peak":          "2025-05-04",
        "end":           "2025-05-07",
        "buildup_days":  4,
        "vcpu_spike":     40.0,
        "memory_spike":   4.0,
        "disk_spike":    12.0,
        "latency_spike": 820.0,
        "error_spike":    7.0,
    },
    {
        "name":          "Yaz Stajı Pik — Kaynak Baskısı",
        "start":         "2025-07-14",
        "peak":          "2025-07-17",
        "end":           "2025-07-21",
        "buildup_days":  3,
        "vcpu_spike":    200.0,
        "memory_spike":  19.0,
        "disk_spike":     4.0,
        "latency_spike": 820.0,
        "error_spike":    9.0,
    },
    {
        "name":          "Yaz Stajı DB Bağlantı Havuzu Tükenmesi",
        "start":         "2025-08-19",
        "peak":          "2025-08-21",
        "end":           "2025-08-25",
        "buildup_days":  2,
        "vcpu_spike":     90.0,
        "memory_spike":  10.0,
        "disk_spike":     0.0,
        "latency_spike": 1900.0,
        "error_spike":   16.0,
    },
    {
        "name":          "Bellek Baskısı — Hesaplama Grubu A",
        "start":         "2025-09-20",
        "peak":          "2025-09-22",
        "end":           "2025-09-26",
        "buildup_days":  5,
        "vcpu_spike":     80.0,
        "memory_spike":  38.0,
        "disk_spike":     0.0,
        "latency_spike": 420.0,
        "error_spike":    8.5,
    },
    {
        "name":          "Disk Kapasitesi Kritik Seviyeye Yaklaşıyor",
        "start":         "2025-10-10",
        "peak":          "2025-10-13",
        "end":           "2025-10-17",
        "buildup_days":  7,
        "vcpu_spike":     40.0,
        "memory_spike":   8.0,
        "disk_spike":    22.0,
        "latency_spike": 260.0,
        "error_spike":    5.5,
    },
    {
        "name":          "Yüksek vCPU Kullanım Olayı",
        "start":         "2025-11-12",
        "peak":          "2025-11-15",
        "end":           "2025-11-19",
        "buildup_days":  3,
        "vcpu_spike":    320.0,
        "memory_spike":  18.0,
        "disk_spike":     0.0,
        "latency_spike": 920.0,
        "error_spike":   14.0,
    },
    {
        "name":          "Yılsonu Trafik Artışı",
        "start":         "2025-12-26",
        "peak":          "2025-12-28",
        "end":           "2026-01-02",
        "buildup_days":  2,
        "vcpu_spike":    200.0,
        "memory_spike":  22.0,
        "disk_spike":     5.0,
        "latency_spike": 1100.0,
        "error_spike":   10.0,
    },
    {
        "name":          "Kış Stajı DB Bağlantı Fırtınası",
        "start":         "2026-01-22",
        "peak":          "2026-01-25",
        "end":           "2026-01-29",
        "buildup_days":  2,
        "vcpu_spike":    140.0,
        "memory_spike":  16.0,
        "disk_spike":     0.0,
        "latency_spike": 3400.0,
        "error_spike":   30.0,
    },
    {
        "name":          "Ağ Katmanı Kesintisi",
        "start":         "2026-02-12",
        "peak":          "2026-02-14",
        "end":           "2026-02-17",
        "buildup_days":  3,
        "vcpu_spike":     70.0,
        "memory_spike":   8.0,
        "disk_spike":     0.0,
        "latency_spike": 2600.0,
        "error_spike":   22.0,
    },
    {
        "name":          "API Gecikme Bozunumu",
        "start":         "2026-03-18",
        "peak":          "2026-03-20",
        "end":           "2026-03-24",
        "buildup_days":  4,
        "vcpu_spike":    170.0,
        "memory_spike":  10.0,
        "disk_spike":     0.0,
        "latency_spike": 2200.0,
        "error_spike":   17.0,
    },
    {
        "name":          "Zamanlayıcı Kaynak Yetersizliği",
        "start":         "2026-04-20",
        "peak":          "2026-04-23",
        "end":           "2026-04-27",
        "buildup_days":  3,
        "vcpu_spike":    230.0,
        "memory_spike":  25.0,
        "disk_spike":     8.0,
        "latency_spike": 1500.0,
        "error_spike":   20.0,
    },
    {
        "name":          "Çok Kaynaklı Basamaklı Arıza",
        "start":         "2026-05-10",
        "peak":          "2026-05-13",
        "end":           "2026-05-17",
        "buildup_days":  5,
        "vcpu_spike":    290.0,
        "memory_spike":  34.0,
        "disk_spike":    14.0,
        "latency_spike": 3000.0,
        "error_spike":   35.0,
    },
    {
        "name":          "Yaz Stajı Keystone Kimlik Doğrulama Krizi",
        "start":         "2026-06-15",
        "peak":          "2026-06-17",
        "end":           "2026-06-21",
        "buildup_days":  3,
        "vcpu_spike":    120.0,
        "memory_spike":  12.0,
        "disk_spike":     0.0,
        "latency_spike": 1800.0,
        "error_spike":   24.0,
    },
]
