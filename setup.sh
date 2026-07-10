#!/usr/bin/env bash
# ============================================================
#  Bulut Altyapısı Proaktif İzleme — Kurulum Scripti
#  Kullanım: bash setup.sh
#  Sadece ilk kurulumda çalıştırılır (~20 dk)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; exit 1; }

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Bulut Altyapısı Proaktif İzleme — Kurulum          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Python kontrolü
if ! command -v python3 &>/dev/null; then
    error "python3 bulunamadı. Lütfen Python 3.9+ kurun."
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python $PY_VER bulundu"

# Sanal ortam
if [ -d "venv" ]; then
    warn "Sanal ortam zaten mevcut. Atlanıyor."
else
    info "Sanal ortam oluşturuluyor..."
    python3 -m venv venv
fi

# Aktifleştir
source venv/bin/activate
info "Sanal ortam aktif: $(which python3)"

# Bağımlılıklar
info "Bağımlılıklar yükleniyor (requirements.txt)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
info "Tüm bağımlılıklar yüklendi."

# Dizin kontrolü
for d in data/raw data/processed/models results/figures results/reports; do
    mkdir -p "$d"
done

# Veri üretimi ve model eğitimi
echo ""
echo "──────────────────────────────────────────────────────"
echo "  ML Pipeline başlatılıyor (ilk çalıştırma ~15-20 dk)"
echo "──────────────────────────────────────────────────────"
echo ""
python3 main.py

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Kurulum tamamlandı!                                 ║"
echo "║  Dashboard başlatmak için:                           ║"
echo "║    bash start.sh                                     ║"
echo "║  veya belirli port için:                             ║"
echo "║    PORT=8502 bash start.sh                           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
