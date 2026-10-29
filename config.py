import os

# ── Genel ayarlar ─────────────────────────────────────────────────────────────
SEED        = 42
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

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

