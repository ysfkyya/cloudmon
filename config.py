import os

# ── Zaman aralığı ─────────────────────────────────────────────────────────────
START_DATE     = "2025-01-01"
END_DATE       = "2026-07-01"   # 18 ay → ~13 140 saatlik nokta
FREQ           = "h"
TRAIN_END_DATE = "2026-04-01"   # Eğitim: Oca 2025–Mar 2026 (15 ay), Test: Nis–Haz 2026 (3 ay)

# ── Genel ayarlar ─────────────────────────────────────────────────────────────
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

# ── ML kapasite tahmini parametreleri ─────────────────────────────────────────
FORECAST_HORIZONS = [24, 72, 168]          # 1 gün / 3 gün / 1 hafta
CAPACITY_TARGETS  = [
    "cpu_usage_percent",
    "memory_usage_percent",
    "disk_usage_percent",
]

# ── Sağlık sınıflandırması parametreleri ──────────────────────────────────────
RISK_HORIZON = 4   # kaç saat sonraki riski tahmin ediyoruz

# ── Staj Dönemleri ────────────────────────────────────────────────────────────
# vm_burst   : staj pik döneminde sisteme eklenen ortalama ek VM sayısı
# cpu_extra  : CPU kullanımına eklenen yüzde puanı (pik)
# mem_extra  : Bellek kullanımına eklenen yüzde puanı (pik)
# error_extra: Hata oranına eklenen yüzde puanı (pik)
# ramp_days  : Stajın başında/sonunda kademeli artış/azalış günü
STAJ_PERIODS = [
    {
        "name":        "Kış Stajı 2025",
        "start":       "2025-02-01",
        "end":         "2025-03-14",
        "vm_burst":    80,
        "cpu_extra":   11.0,
        "mem_extra":   8.0,
        "error_extra": 2.5,
        "ramp_days":   7,
    },
    {
        "name":        "Yaz Stajı 2025",
        "start":       "2025-06-01",
        "end":         "2025-09-12",
        "vm_burst":    185,
        "cpu_extra":   22.0,
        "mem_extra":   15.0,
        "error_extra": 3.8,
        "ramp_days":   14,
    },
    {
        "name":        "Kış Stajı 2025-2026",
        "start":       "2026-01-10",
        "end":         "2026-02-27",
        "vm_burst":    105,
        "cpu_extra":   13.0,
        "mem_extra":   10.0,
        "error_extra": 3.0,
        "ramp_days":   7,
    },
    {
        "name":        "Yaz Stajı 2026",
        "start":       "2026-06-01",
        "end":         "2026-07-01",
        "vm_burst":    155,
        "cpu_extra":   19.0,
        "mem_extra":   13.0,
        "error_extra": 3.5,
        "ramp_days":   14,
    },
]

# ── Kapasite Genişlemeleri (yeni donanım alımları) ───────────────────────────
# Yeni compute node / depolama eklendiğinde kullanım yüzdesi anlık düşer.
# *_reduction: o metriğin kullanım yüzdesinden düşülecek puan
CAPACITY_EXPANSIONS = [
    {
        "name":          "Disk Depolama Genişlemesi — Mayıs 2025",
        "date":          "2025-05-10",
        "cpu_reduction":  0.0,
        "mem_reduction":  0.0,
        "disk_reduction": 16.0,
    },
    {
        "name":          "Sonbahar 2025 Compute Alımı (8 yeni node)",
        "date":          "2025-10-20",
        "cpu_reduction":  14.0,
        "mem_reduction":  11.0,
        "disk_reduction":  0.0,
    },
    {
        "name":          "İlkbahar 2026 Compute Alımı (5 yeni node)",
        "date":          "2026-03-05",
        "cpu_reduction":   9.0,
        "mem_reduction":   7.0,
        "disk_reduction":  0.0,
    },
]

# ── Incident Tanımları (15 olay — 18 ay boyunca) ─────────────────────────────
# buildup_days: olaydan kaç gün önce göstergeler bozulmaya başlar
INCIDENTS = [
    # ── Oca–Ağu 2025 ──
    {
        "name":          "Kış Stajı Keystone Kimlik Aşırı Yükü",
        "start":         "2025-02-06",
        "peak":          "2025-02-08",
        "end":           "2025-02-12",
        "buildup_days":  3,
        "cpu_spike":     10.0,
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
        "cpu_spike":     18.0,
        "memory_spike":   8.0,
        "disk_spike":     0.0,
        "latency_spike": 540.0,
        "error_spike":    6.0,
    },
    {
        "name":          "Disk G/Ç Yavaşlaması — Depolama Öncesi",
        "start":         "2025-05-02",
        "peak":          "2025-05-04",
        "end":           "2025-05-07",
        "buildup_days":  4,
        "cpu_spike":      6.0,
        "memory_spike":   4.0,
        "disk_spike":     0.0,
        "latency_spike": 820.0,
        "error_spike":    7.0,
    },
    {
        "name":          "Yaz Stajı Pik — CPU Baskısı",
        "start":         "2025-07-14",
        "peak":          "2025-07-17",
        "end":           "2025-07-21",
        "buildup_days":  3,
        "cpu_spike":     30.0,
        "memory_spike":  19.0,
        "disk_spike":     0.0,
        "latency_spike": 820.0,
        "error_spike":    9.0,
    },
    {
        "name":          "Yaz Stajı DB Bağlantı Havuzu Tükenmesi",
        "start":         "2025-08-19",
        "peak":          "2025-08-21",
        "end":           "2025-08-25",
        "buildup_days":  2,
        "cpu_spike":     14.0,
        "memory_spike":  10.0,
        "disk_spike":     0.0,
        "latency_spike": 1900.0,
        "error_spike":   16.0,
    },
    # ── Eyl 2025–Haz 2026 ──
    {
        "name":          "Bellek Baskısı — Hesaplama Grubu A",
        "start":         "2025-09-20",
        "peak":          "2025-09-22",
        "end":           "2025-09-26",
        "buildup_days":  5,
        "cpu_spike":     12.0,
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
        "cpu_spike":      6.0,
        "memory_spike":   8.0,
        "disk_spike":    22.0,
        "latency_spike": 260.0,
        "error_spike":    5.5,
    },
    {
        "name":          "Yüksek CPU Yükü Olayı",
        "start":         "2025-11-12",
        "peak":          "2025-11-15",
        "end":           "2025-11-19",
        "buildup_days":  3,
        "cpu_spike":     48.0,
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
        "cpu_spike":     30.0,
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
        "cpu_spike":     22.0,
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
        "cpu_spike":     10.0,
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
        "cpu_spike":     26.0,
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
        "cpu_spike":     35.0,
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
        "cpu_spike":     45.0,
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
        "cpu_spike":     18.0,
        "memory_spike":  12.0,
        "disk_spike":     0.0,
        "latency_spike": 1800.0,
        "error_spike":   24.0,
    },
]
