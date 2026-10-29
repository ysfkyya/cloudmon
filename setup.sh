#!/usr/bin/env bash
# ============================================================
#  Bulut Altyapısı Proaktif İzleme — Kurulum Scripti
#  Kullanım: bash setup.sh
#  Sadece ilk kurulumda çalıştırılır (~15-20 dk)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; exit 1; }
step()  { echo -e "\n${CYAN}──────────────────────────────────────────────────────${NC}"; echo -e "  $*"; echo -e "${CYAN}──────────────────────────────────────────────────────${NC}"; }

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Bulut Altyapısı Proaktif İzleme — Kurulum          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Python kontrolü ───────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    error "python3 bulunamadı. Lütfen Python 3.9+ kurun."
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python $PY_VER bulundu"

# ── API Anahtarları ───────────────────────────────────────────────────────────
step "LLM API Anahtarları (opsiyonel — atlamak için Enter)"

ENV_FILE="$SCRIPT_DIR/.env"
ANTH_KEY=""
OPEN_KEY=""

# Mevcut .env varsa oku
if [ -f "$ENV_FILE" ]; then
    warn ".env dosyası zaten mevcut. Mevcut değerler korunacak (yeniden girmek isterseniz silin: rm .env)"
    ANTH_KEY=$(grep "^ANTHROPIC_API_KEY=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2- || true)
    OPEN_KEY=$(grep  "^OPENAI_API_KEY="    "$ENV_FILE" 2>/dev/null | cut -d'=' -f2- || true)
fi

if [ -z "$ANTH_KEY" ]; then
    echo -n "  Anthropic API anahtarı (Claude Sonnet): "
    read -r ANTH_KEY
fi
if [ -z "$OPEN_KEY" ]; then
    echo -n "  OpenAI API anahtarı   (GPT-4o):        "
    read -r OPEN_KEY
fi

# .env yaz
{
    echo "# CloudMon LLM API Anahtarları — asla commit'leme!"
    [ -n "$ANTH_KEY" ] && echo "ANTHROPIC_API_KEY=$ANTH_KEY"
    [ -n "$OPEN_KEY" ] && echo "OPENAI_API_KEY=$OPEN_KEY"
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"

if [ -n "$ANTH_KEY" ] || [ -n "$OPEN_KEY" ]; then
    info ".env oluşturuldu (LLM entegrasyonu aktif)"
else
    warn "API anahtarı girilmedi — LLM önerileri devre dışı kalacak (dashboard çalışır)"
fi

# ── Sanal ortam ───────────────────────────────────────────────────────────────
if [ -d "venv" ]; then
    warn "Sanal ortam zaten mevcut. Atlanıyor."
else
    info "Sanal ortam oluşturuluyor..."
    python3 -m venv venv
fi

source venv/bin/activate
info "Sanal ortam aktif: $(which python3)"

# ── Bağımlılıklar ─────────────────────────────────────────────────────────────
info "Bağımlılıklar yükleniyor (requirements.txt)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
info "Tüm bağımlılıklar yüklendi."

# ── Dizin yapısı ──────────────────────────────────────────────────────────────
for d in data/raw data/processed/models results/figures results/reports; do
    mkdir -p "$d"
done

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Kurulum tamamlandı!                                 ║"
echo "║                                                      ║"
echo "║  ML pipeline çalıştırmak için:                       ║"
echo "║    source venv/bin/activate && python3 main.py       ║"
echo "║                                                      ║"
echo "║  Dashboard başlatmak için:                           ║"
echo "║    bash start.sh                                     ║"
echo "║  veya belirli port için:                             ║"
echo "║    PORT=8502 bash start.sh                           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
