#!/usr/bin/env bash
# ============================================================
#  Bulut Altyapısı Proaktif İzleme — Dashboard Başlatıcı
#  Kullanım:
#    bash start.sh             # port 8501
#    PORT=8502 bash start.sh   # özel port
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8501}"
VENV="$SCRIPT_DIR/venv"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# Venv kontrolü
if [ ! -d "$VENV" ]; then
    echo -e "${RED}[✗]${NC} Sanal ortam bulunamadı."
    echo "    Önce kurulum yapın: bash setup.sh"
    exit 1
fi

source "$VENV/bin/activate"

# Sonuç dosyası kontrolü
if [ ! -f "results/reports/capacity_results.json" ]; then
    if [ ! -f "data/raw/metrics.csv" ]; then
        echo -e "${RED}[✗]${NC} data/raw/metrics.csv bulunamadı. Veri dosyalarını data/raw/ dizinine koyun."
        exit 1
    fi
    echo -e "${YELLOW}[!]${NC} ML sonuçları bulunamadı. Pipeline çalıştırılıyor..."
    python3 main.py
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Dashboard başlatılıyor                              ║"
echo "║  http://localhost:${PORT}                            ║"
echo "║  Durdurmak için: Ctrl+C                              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

python3 -m streamlit run dashboard.py \
    --server.port "$PORT" \
    --server.headless true \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false
