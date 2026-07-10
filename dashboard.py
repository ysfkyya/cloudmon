"""
dashboard.py — Bulut Altyapısı Proaktif İzleme ve Karar Destek Sistemi
Streamlit Arayüzü  (jinja2 gerektirmez)

Başlatmak için:
    source venv/bin/activate
    python3 -m streamlit run dashboard.py --server.port 8501
"""

import os
import sys
import json
import subprocess

import pandas as pd
import streamlit as st
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# ── Sayfa yapılandırması ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cloud Proaktif İzleme",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS  = os.path.join(BASE, "results")
DATA_DIR = os.path.join(BASE, "data")
FIG_DIR  = os.path.join(RESULTS, "figures")
REP_DIR  = os.path.join(RESULTS, "reports")


# ── Yardımcılar ───────────────────────────────────────────────────────────────

def fig_path(name: str) -> str:
    return os.path.join(FIG_DIR, name)


def load_json(name: str):
    path = os.path.join(REP_DIR, name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def show_image(path: str, caption: str = ""):
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Grafik bulunamadı: {os.path.basename(path)}")


def file_size_kb(path: str) -> str:
    try:
        return f"{os.path.getsize(path) / 1024:.0f} KB"
    except FileNotFoundError:
        return "—"


def results_exist() -> bool:
    return (
        os.path.exists(os.path.join(REP_DIR, "capacity_results.json"))
        and os.path.exists(os.path.join(REP_DIR, "health_results.json"))
    )


def fmt_df(df: pd.DataFrame, fmt: dict) -> pd.DataFrame:
    """jinja2 gerektirmeden sütunları formatlar (string'e çevirir)."""
    df = df.copy()
    for col, spec in fmt.items():
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: spec.format(v) if isinstance(v, (int, float)) else v
            )
    return df


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.rec-critical { background:linear-gradient(90deg,#ffe0e0,#fff5f5);
    border-left:5px solid #e53e3e; padding:14px 18px;
    border-radius:6px; margin-bottom:14px; }
.rec-warning  { background:linear-gradient(90deg,#fff8e1,#fffde7);
    border-left:5px solid #f59e0b; padding:14px 18px;
    border-radius:6px; margin-bottom:14px; }
.rec-info     { background:linear-gradient(90deg,#e3f2fd,#f0f9ff);
    border-left:5px solid #3b82f6; padding:14px 18px;
    border-radius:6px; margin-bottom:14px; }
.box { background:#f8fafc; border-radius:8px; padding:12px 16px;
    font-size:.93rem; color:#374151; margin-bottom:14px;
    border:1px solid #e5e7eb; }
.badge-c { color:#fff; background:#e53e3e; padding:2px 8px;
    border-radius:12px; font-size:.78rem; font-weight:700; }
.badge-w { color:#fff; background:#f59e0b; padding:2px 8px;
    border-radius:12px; font-size:.78rem; font-weight:700; }
.badge-i { color:#fff; background:#3b82f6; padding:2px 8px;
    border-radius:12px; font-size:.78rem; font-weight:700; }
.tag-real { background:#dcfce7; color:#166534; padding:2px 8px;
    border-radius:10px; font-size:.75rem; font-weight:600; }
.tag-sim  { background:#fef9c3; color:#854d0e; padding:2px 8px;
    border-radius:10px; font-size:.75rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("☁️ Proaktif İzleme")
    st.caption("OpenStack · ML Pipeline · SHAP")
    st.divider()

    page = st.radio("Sayfa", [
        "🏠  Ana Sayfa",
        "📊  Veri Genel Görünüm",
        "📈  Kapasite Tahmini",
        "🩺  Sağlık Sınıflandırması",
        "🔍  SHAP Açıklanabilirlik",
        "💡  Operasyonel Öneriler",
    ], label_visibility="collapsed")

    st.divider()
    if st.button("🔄  Pipeline'ı Yeniden Çalıştır", use_container_width=True):
        with st.spinner("Pipeline çalışıyor… (~15 dk)"):
            r = subprocess.run(
                [sys.executable, os.path.join(BASE, "main.py")],
                capture_output=True, text=True,
            )
        if r.returncode == 0:
            st.success("Pipeline tamamlandı!")
            st.rerun()
        else:
            st.error("Hata oluştu.")
            st.code(r.stderr[-2000:])

    if not results_exist():
        st.warning("Sonuçlar bulunamadı.\nPipeline'ı çalıştırın.")

    st.divider()
    if results_exist():
        try:
            m = pd.read_csv(os.path.join(DATA_DIR, "raw", "metrics.csv"),
                            usecols=["timestamp"])
            st.caption(f"📅 {m['timestamp'].iloc[0][:10]}")
            st.caption(f"   → {m['timestamp'].iloc[-1][:10]}")
            st.caption(f"📦 {len(m):,} saatlik nokta")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  ANA SAYFA
# ═══════════════════════════════════════════════════════════════════════════════
if page.startswith("🏠"):
    st.title("Bulut Altyapısı Proaktif İzleme ve Karar Destek Sistemi")
    st.caption("OpenStack · Prometheus / Thanos · OpenSearch · Makine Öğrenmesi · SHAP Açıklanabilirlik")

    # ── Canlı durum kartları ──────────────────────────────────────────────────
    if results_exist():
        try:
            mdf  = pd.read_csv(os.path.join(DATA_DIR, "raw", "metrics.csv"))
            last = mdf.iloc[-1]
            prev = mdf.iloc[-25]
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("CPU",    f"{last['cpu_usage_percent']:.1f}%",
                      f"{last['cpu_usage_percent']-prev['cpu_usage_percent']:+.1f}%")
            c2.metric("Bellek", f"{last['memory_usage_percent']:.1f}%",
                      f"{last['memory_usage_percent']-prev['memory_usage_percent']:+.1f}%")
            c3.metric("Disk",   f"{last['disk_usage_percent']:.1f}%",
                      f"{last['disk_usage_percent']-prev['disk_usage_percent']:+.1f}%")
            c4.metric("API Gecikme", f"{last['api_latency_ms']:.0f} ms",
                      f"{last['api_latency_ms']-prev['api_latency_ms']:+.0f} ms",
                      delta_color="inverse")
            c5.metric("Hata Oranı", f"{last['error_rate_percent']:.2f}%",
                      f"{last['error_rate_percent']-prev['error_rate_percent']:+.2f}%",
                      delta_color="inverse")
            st.caption("↑ Son veri noktasının değerleri; delta = önceki 24 saate göre değişim")
        except Exception as e:
            st.info(f"Metrik kartları yüklenemedi: {e}")

    st.divider()

    # ── Sekmeler ──────────────────────────────────────────────────────────────
    t1,t2,t3,t4,t5 = st.tabs([
        "📌 Sistem Hakkında",
        "🔌 Veri Kaynakları",
        "📦 Veri Boyutları",
        "⚙️  Pipeline Adımları",
        "🔬 Teknik Detaylar",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    with t1:
        st.subheader("Problem")
        st.markdown("""
        <div class="box">
        Günümüzde OpenStack gibi bulut platformlarının izlenmesi büyük ölçüde
        <b>eşik değer temelli alarm sistemlerine</b> dayanmaktadır:
        CPU %85'i geçtiğinde alarm → operatör müdahale eder.
        Bu yaklaşım <b>reaktiftir</b>: sorun zaten oluştuktan sonra harekete geçilir.<br><br>
        Oysa çoğu sistem arızası ani değil, <b>saatler veya günler öncesinde</b>
        başlayan birikim süreçlerinin (buildup) ürünüdür.
        CPU yavaşça artıyor, log hataları seyrekçe artıyor, API gecikmeleri kabarmaya başlıyor —
        ama mevcut sistemler bu sinyalleri erken aşamada görmüyor.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Çözüm")
        st.markdown("""
        <div class="box">
        Bu çalışma iki katmanlı bir yapay zekâ destekli karar destek mekanizması geliştirir:<br><br>
        <b>Katman 1 — Kapasite Tahmini (Regresyon):</b><br>
        Geçmiş kaynak kullanım trendlerini analiz ederek CPU, Bellek ve Disk kullanımını
        <b>24 saat ve 72 saat sonra</b> tahmin eder.
        Sistem yöneticisi kritik eşiğe kaç gün kaldığını önceden bilir.<br><br>
        <b>Katman 2 — Sağlık Riski Sınıflandırması (Binary ML):</b><br>
        Metrik verileri ile servis loglarını aynı analiz çerçevesinde birleştirerek
        <b>"sonraki 4 saatte herhangi bir kritik olay gerçekleşecek mi?"</b>
        sorusunu olasılıkla yanıtlar.<br><br>
        <b>Açıklanabilirlik (SHAP):</b><br>
        Tahminlerin <i>neden</i> üretildiğini Shapley değerleriyle gösterir:
        "Bu risk uyarısı başta <i>error_rate_percent</i> ve <i>timeout_count</i> özelliklerinden kaynaklanıyor."<br><br>
        <b>Öneri Motoru:</b><br>
        Tüm analiz çıktıları somut ve uygulanabilir operasyonel önerilere dönüştürülür.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Sistem Mimarisi")
        st.code("""
┌─────────────────────────────────────────────────────────────────┐
│                       VERİ KAYNAKLARI                           │
│                                                                 │
│  Prometheus / Thanos          OpenSearch                        │
│  ┌─────────────────────┐      ┌──────────────────────────┐     │
│  │ /api/v1/query_range │      │ openstack-logs-* index   │     │
│  │ CPU · Bellek · Disk │      │ Nova · Keystone · Neutron│     │
│  │ API gecikme · Hata  │      │ Cinder log kayıtları     │     │
│  └──────────┬──────────┘      └─────────────┬────────────┘     │
└─────────────┼───────────────────────────────┼────────────────── ┘
              │  saatlik metrik (10 sütun)     │  log sayımları (18 sütun)
              └────────────────┬───────────────┘
                               ▼
              ┌────────────────────────────────┐
              │      ÖZELLİK MÜHENDİSLİĞİ      │
              │  rolling · lag · delta · zaman  │
              │  132 türetilmiş özellik         │
              └────────────┬───────────────────┘
                           │
           ┌───────────────┴──────────────────┐
           ▼                                  ▼
┌──────────────────────┐          ┌──────────────────────────┐
│   KAPASİTE TAHMİNİ   │          │  SAĞLIK SINIFLANDIRMASI  │
│  LR · RF · GBT       │          │  LR · RF · GBT (binary)  │
│  24h / 72h ufku       │          │  4h risk ufku            │
└──────────┬───────────┘          └─────────────┬────────────┘
           │                                    │
           └─────────────────┬──────────────────┘
                             ▼
              ┌───────────────────────────────┐
              │  SHAP · ÖNERİ MOTORU          │
              │  Önem sıralaması + renkli kart │
              └───────────────────────────────┘
        """, language=None)

    # ─────────────────────────────────────────────────────────────────────────
    with t2:
        st.subheader("Veri Kaynakları — Gerçek Sistem vs. Bu Prototip")

        st.markdown("""
        <div class="box">
        ⚠️ <b>Önemli Not:</b>
        Bu akademik prototipte veriler gerçek API'lardan çekilmemektedir.
        Tüm metrik ve log verileri Python ile <b>sentetik olarak üretilmiştir</b>.
        Ancak üretim mantığı gerçek OpenStack davranışını yansıtacak biçimde tasarlanmış,
        gerçekçi günlük/haftalık örüntüler, uzun vadeli trendler ve belgelenmiş incident tipleri modellenmiştir.
        Aşağıda her veri kaynağının gerçek sistemde nasıl çalışacağı ve bu prototipin nasıl simüle ettiği açıklanmaktadır.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📡 Metrik Verileri")
            st.markdown("""
            <div class="box">
            <span class="tag-real">GERÇEK SİSTEMDE</span><br>
            Prometheus HTTP API'dan çekilir:<br>
            <code>GET /api/v1/query_range<br>
            ?query=node_cpu_seconds_total<br>
            &start=2025-09-01T00:00:00Z<br>
            &end=2026-07-01T00:00:00Z<br>
            &step=3600</code><br><br>
            Thanos ile uzun dönem metrikleri saklanır.
            Her compute node için ayrı time series.
            </div>
            <div class="box">
            <span class="tag-sim">BU PROTOTİPTE</span><br>
            <code>src/data_generation/metrics_generator.py</code><br><br>
            <b>Sinyal bileşenleri:</b><br>
            • Günlük örüntü: kosinüs fonksiyonu (zirve 13:00, dip 04:00)<br>
            • Haftalık örüntü: hafta sonu %35 düşük yük<br>
            • Uzun vadeli trend: lineer artış (bellek +18%, disk +32%)<br>
            • 10 tanımlı incident: Gauss yoğunluk profili ile buildup dönemi<br>
            • Gaussian gürültü ile gerçekçilik<br>
            • SEED=42 → tekrarlanabilir sonuçlar
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Üretilen Metrik Sütunları:**")
            st.dataframe(pd.DataFrame([
                {"Sütun": "cpu_usage_percent",   "Birim": "%",    "Aralık": "2–99"},
                {"Sütun": "memory_usage_percent","Birim": "%",    "Aralık": "10–99"},
                {"Sütun": "disk_usage_percent",  "Birim": "%",    "Aralık": "20–99"},
                {"Sütun": "api_requests_per_hour","Birim":"istek/s","Aralık": "50–200K"},
                {"Sütun": "api_latency_ms",      "Birim": "ms",   "Aralık": "10–15,000"},
                {"Sütun": "error_rate_percent",  "Birim": "%",    "Aralık": "0–100"},
                {"Sütun": "network_in_mbps",     "Birim": "Mbps", "Aralık": "0–10,000"},
                {"Sütun": "network_out_mbps",    "Birim": "Mbps", "Aralık": "0–10,000"},
                {"Sütun": "vm_count",            "Birim": "adet", "Aralık": "1–600"},
            ]), width='stretch', hide_index=True)

        with col2:
            st.markdown("#### 📋 Log Verileri")
            st.markdown("""
            <div class="box">
            <span class="tag-real">GERÇEK SİSTEMDE</span><br>
            OpenSearch REST API'dan JSON sorguyla çekilir:<br>
            <code>POST /openstack-logs-*/_search<br>
            {"query":{"range":{"@timestamp":<br>
              {"gte":"now-1h","lte":"now"}}},<br>
             "aggs":{"by_service":{"terms":<br>
              {"field":"service.keyword"}},<br>
             "by_level":{"terms":<br>
              {"field":"level.keyword"}}}}</code><br><br>
            Nova, Keystone, Neutron, Cinder log dosyaları
            Filebeat ile OpenSearch'e gönderilir.
            </div>
            <div class="box">
            <span class="tag-sim">BU PROTOTİPTE</span><br>
            <code>src/data_generation/log_generator.py</code><br><br>
            <b>İki çıktı üretilir:</b><br>
            1. <b>log_features.csv</b> — ML için saatlik agregasyon<br>
               Her saat için servis × hata tipi sayımları.<br>
               Üretim mantığı: metrik verisiyle korele (error_rate, cpu_factor, latency_factor).<br><br>
            2. <b>raw_logs.csv</b> — Görsel doğrulama için gerçekçi log satırları<br>
               Şablon: <code>"Timeout connecting to nova.api after 3421ms"</code><br>
               5,000 örnek satır, incident dönemlerinde daha yoğun.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Üretilen Log Özellik Sütunları:**")
            st.dataframe(pd.DataFrame([
                {"Sütun": "nova_api_errors",          "Açıklama": "Nova API hata sayısı/saat"},
                {"Sütun": "nova_compute_errors",      "Açıklama": "Nova Compute hata sayısı/saat"},
                {"Sütun": "nova_scheduler_errors",    "Açıklama": "Nova Scheduler hata sayısı/saat"},
                {"Sütun": "keystone_errors",          "Açıklama": "Keystone hata sayısı/saat"},
                {"Sütun": "neutron_errors",           "Açıklama": "Neutron hata sayısı/saat"},
                {"Sütun": "cinder_errors",            "Açıklama": "Cinder hata sayısı/saat"},
                {"Sütun": "timeout_count",            "Açıklama": "Bağlantı zaman aşımı sayısı"},
                {"Sütun": "db_connection_count",      "Açıklama": "DB bağlantı hatası sayısı"},
                {"Sütun": "resource_exhausted_count", "Açıklama": "Kaynak yetersizliği sayısı"},
                {"Sütun": "auth_failure_count",       "Açıklama": "Kimlik doğrulama hatası"},
                {"Sütun": "network_error_count",      "Açıklama": "Ağ katmanı hata sayısı"},
                {"Sütun": "scheduler_error_count",    "Açıklama": "Zamanlayıcı hatası"},
                {"Sütun": "io_error_count",           "Açıklama": "I/O hata sayısı"},
                {"Sütun": "critical_count",           "Açıklama": "CRITICAL seviyeli log sayısı"},
                {"Sütun": "total_errors",             "Açıklama": "Toplam hata sayısı/saat"},
                {"Sütun": "total_log_count",          "Açıklama": "Toplam log satırı/saat"},
                {"Sütun": "error_log_ratio",          "Açıklama": "Hata / toplam log oranı"},
            ]), width='stretch', hide_index=True)

        st.divider()
        st.subheader("Gerçek Sisteme Entegrasyon")
        st.markdown("""
        <div class="box">
        Bu prototip gerçek bir sisteme entegre edilmek istendiğinde yalnızca <b>veri üretim katmanı</b> değiştirilir.
        Özellik mühendisliği, ML modelleri, SHAP analizi ve öneri motoru aynı kalır:<br><br>
        <table style="width:100%; border-collapse:collapse; font-size:.88rem;">
        <tr style="background:#f1f5f9;">
          <th style="padding:6px; text-align:left;">Prototip</th>
          <th style="padding:6px; text-align:left;">Gerçek Sistem</th>
        </tr>
        <tr><td style="padding:6px;"><code>metrics_generator.generate_metrics()</code></td>
            <td style="padding:6px;">Prometheus HTTP API → pandas DataFrame</td></tr>
        <tr style="background:#f8fafc;"><td style="padding:6px;"><code>log_generator.generate_log_features()</code></td>
            <td style="padding:6px;">OpenSearch aggregation query → pandas DataFrame</td></tr>
        <tr><td style="padding:6px;"><code>feature_engineering.build_feature_matrix()</code></td>
            <td style="padding:6px;"><b>Değişmez</b> — aynı kod</td></tr>
        <tr style="background:#f8fafc;"><td style="padding:6px;">ML Modelleri, SHAP, Öneriler</td>
            <td style="padding:6px;"><b>Değişmez</b> — aynı kod</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    with t3:
        st.subheader("Veri Boyutları ve Dosya Yapısı")

        if results_exist():
            raw_dir  = os.path.join(DATA_DIR, "raw")
            proc_dir = os.path.join(DATA_DIR, "processed")

            # Dosya tablosu
            files = [
                ("data/raw/metrics.csv",    "Ham metrik verileri",          raw_dir,  "metrics.csv"),
                ("data/raw/log_features.csv","Saatlik log özellik agregasyonu", raw_dir, "log_features.csv"),
                ("data/raw/raw_logs.csv",   "Ham log satırı örnekleri",     raw_dir,  "raw_logs.csv"),
                ("data/processed/features.csv","Türetilmiş özellik matrisi", proc_dir, "features.csv"),
            ]

            rows = []
            for label, desc, d, fname in files:
                path = os.path.join(d, fname)
                try:
                    df_tmp = pd.read_csv(path, nrows=0)
                    ncol   = len(df_tmp.columns)
                    nrow   = sum(1 for _ in open(path)) - 1
                    size   = file_size_kb(path)
                except Exception:
                    ncol, nrow, size = "—", "—", "—"
                rows.append({"Dosya": label, "Açıklama": desc,
                             "Satır": f"{nrow:,}" if isinstance(nrow, int) else nrow,
                             "Sütun": ncol, "Boyut": size})

            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

            # Model dosyaları
            model_dir = os.path.join(proc_dir, "models")
            if os.path.exists(model_dir):
                models = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
                total_mb = sum(
                    os.path.getsize(os.path.join(model_dir, f))
                    for f in models
                ) / (1024**2)
                st.info(f"**Model dosyaları:** `data/processed/models/`  ·  "
                        f"{len(models)} adet .pkl  ·  Toplam ~{total_mb:.1f} MB")

            # Grafik dosyaları
            if os.path.exists(FIG_DIR):
                figs   = [f for f in os.listdir(FIG_DIR) if f.endswith(".png")]
                fig_mb = sum(os.path.getsize(os.path.join(FIG_DIR, f))
                             for f in figs) / (1024**2)
                st.info(f"**Grafik dosyaları:** `results/figures/`  ·  "
                        f"{len(figs)} adet .png  ·  Toplam ~{fig_mb:.1f} MB")

        st.divider()
        st.subheader("Zaman Dilimi ve Olay Takvimi")

        try:
            from config import INCIDENTS, START_DATE, END_DATE, TRAIN_END_DATE
            st.markdown(f"""
            <div class="box">
            <b>Veri aralığı:</b> {START_DATE} → {END_DATE} (10 ay, saatlik)<br>
            <b>Eğitim seti:</b> {START_DATE} → {TRAIN_END_DATE} (7 ay)<br>
            <b>Test seti:</b> {TRAIN_END_DATE} → {END_DATE} (3 ay)<br>
            <b>Toplam incident:</b> {len(INCIDENTS)} adet (10 ay boyunca dağıtılmış)
            </div>
            """, unsafe_allow_html=True)

            inc_df = pd.DataFrame([{
                "Olay":          inc["name"],
                "Başlangıç":    inc["start"],
                "Zirve":        inc["peak"],
                "Bitiş":        inc["end"],
                "Öncü (gün)":   inc["buildup_days"],
                "CPU +":        f"{inc['cpu_spike']:.0f}%",
                "Bellek +":     f"{inc['memory_spike']:.0f}%",
                "Gecikme +":    f"{int(inc['latency_spike'])} ms",
                "Hata +":       f"{inc['error_spike']:.0f}%",
            } for inc in INCIDENTS])
            st.dataframe(inc_df, width='stretch', hide_index=True)
        except Exception as e:
            st.warning(f"Config yüklenemedi: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    with t4:
        st.subheader("Pipeline Adımları — Girdi / İşlem / Çıktı")

        steps_detail = [
            {
                "no": "1️⃣",
                "ad": "Veri Üretimi",
                "girdi": "config.py (tarih aralığı, incident tanımları, SEED=42)",
                "islem": [
                    "metrics_generator.generate_metrics() → saatlik CPU/Bellek/Disk/Gecikme/Hata sinyalleri",
                    "log_generator.generate_log_features() → metrikle korele saatlik log sayımları",
                    "log_generator.generate_raw_logs() → 5,000 gerçekçi log satırı örneği",
                ],
                "cikti": "data/raw/ altında 3 CSV dosyası",
                "sure": "~5 sn",
                "not": "SEED=42 olduğu için her çalıştırmada özdeş veri üretilir. "
                       "Config'i değiştirirseniz (tarih, incident, eşik) farklı veri elde edilir.",
            },
            {
                "no": "2️⃣",
                "ad": "Özellik Mühendisliği",
                "girdi": "metrics.csv + log_features.csv",
                "islem": [
                    "Metrik + log verilerini timestamp üzerinden birleştirir (merge)",
                    "Zaman özellikleri: hour_of_day, day_of_week, is_weekend, is_business_hour vb.",
                    "Hareketli istatistikler: 3h/6h/12h/24h/168h rolling mean ve rolling std",
                    "Gecikme (lag) özellikleri: 1h/6h/24h/48h/168h önceki değerler",
                    "Değişim hızı (delta): 1h/6h/24h farkları",
                    "Log hareketli toplamları: 6h mean ve 24h sum",
                ],
                "cikti": "data/processed/features.csv — 7,272 satır × 132 sütun",
                "sure": "~2 sn",
                "not": "İlk 168 saat lag nedeniyle NaN içerebilir; model eğitiminde dropna() uygulanır.",
            },
            {
                "no": "3️⃣",
                "ad": "Kapasite Tahmini Modelleri",
                "girdi": "features.csv — eğitim: Eyl 2025–Mar 2026 / test: Nis–Haz 2026",
                "islem": [
                    "3 kaynak (CPU, Bellek, Disk) × 2 ufuk (24h, 72h) = 6 hedef değişken",
                    "Her hedef için LinearRegression, RandomForest, GradientBoosting eğitilir",
                    "Toplam 18 model — çapraz doğrulama yok, zamansal bölünmüş test seti",
                    "LR için StandardScaler; RF/GBT için ölçekleme gerekmez",
                ],
                "cikti": "18 .pkl model, capacity_results.json, 2 PNG grafik",
                "sure": "~8 dk (GBT n_estimators=200 nedeniyle)",
                "not": "Disk tahmininde R² negatif: disk monoton artar, "
                       "test dönemindeki ani spike'lar eğitim setinde görülmemiştir.",
            },
            {
                "no": "4️⃣",
                "ad": "Sağlık Riski Sınıflandırması",
                "girdi": "features.csv + risk etiketi (sonraki 4 saate bakış)",
                "islem": [
                    "Risk etiketi: sonraki 4 saatte CPU>85 VEYA Bellek>88 VEYA Hata>%5 VEYA Gecikme>1200ms",
                    "class_weight='balanced' ile sınıf dengesizliği (risk ~%12) ele alınır",
                    "LR (ölçeklenmiş), RF, GBT — aynı zamansal bölünmüş set",
                    "Değerlendirme: Accuracy, Precision, Recall, F1, AUC-ROC",
                ],
                "cikti": "3 .pkl model, health_results.json, 3 PNG grafik",
                "sure": "~30 sn",
                "not": "AUC=1.0 değerleri: risk etiketi doğrudan metriklerden türetildiğinden "
                       "modeller bu sinyalleri mükemmel öğrenebilir — sentetik veri avantajı.",
            },
            {
                "no": "5️⃣",
                "ad": "SHAP Açıklanabilirlik",
                "girdi": "Eğitilmiş modeller + test seti (500 örneklem)",
                "islem": [
                    "TreeExplainer: ağaç tabanlı modeller için verimli Shapley değeri hesabı",
                    "Her kapasite modeli için: bar, beeswarm, dependence grafikleri",
                    "Sağlık modeli için: aynı görselleştirmeler",
                    "CSV tablosu: tüm özellikler için ortalama |SHAP| önem sıralaması",
                ],
                "cikti": "13 PNG grafik, 5 CSV tablo (results/reports/shap_*.csv)",
                "sure": "~40 sn",
                "not": "500 örneklem seçimi: tam test seti yerine örneklem "
                       "hesaplama süresini makul tutar (tam set ~10x yavaş).",
            },
            {
                "no": "6️⃣",
                "ad": "Operasyonel Öneri Motoru",
                "girdi": "Model çıktıları + SHAP sonuçları + son 72/168 saatlik veri",
                "islem": [
                    "Kapasite: son 72 saatin lineer trendi → kritik eşiğe kaç gün?",
                    "Sağlık: son 48 saatlik test penceresinin ortalama risk olasılığı",
                    "Trend: son 7 günün önceki 7 güne göre hata oranı ve gecikme artışı",
                    "SHAP top-1 özelliği → öneriye eklenir ('En etkili özellik: timeout_count')",
                    "Öncelik sıralaması: critical > warning > info",
                ],
                "cikti": "recommendations.json + recommendations.txt",
                "sure": "<1 sn",
                "not": "Öneriler her pipeline çalıştırmasında güncellenir. "
                       "Gerçek sistemde saatlik cron job ile otomatik çalışabilir.",
            },
        ]

        for step in steps_detail:
            with st.expander(f"{step['no']} {step['ad']}  ({step['sure']})", expanded=False):
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown(f"**Girdi:** {step['girdi']}")
                    st.markdown("**İşlem adımları:**")
                    for s in step["islem"]:
                        st.markdown(f"- {s}")
                with c2:
                    st.markdown(f"**Çıktı:** {step['cikti']}")
                    st.markdown(f"""<div class="box">💡 {step['not']}</div>""",
                                unsafe_allow_html=True)

        st.divider()
        st.subheader("'Pipeline'ı Yeniden Çalıştır' Ne Yapar?")
        st.markdown("""
        <div class="box">
        Sidebar'daki <b>🔄 Pipeline'ı Yeniden Çalıştır</b> butonu <code>main.py</code>'yi yeniden çalıştırır.
        <br><br>
        <b>SEED=42 ile aynı veriyi çalıştırırsanız:</b> Tüm sonuçlar birebir aynı olur
        (veri deterministik, modeller aynı hiperparametrelerle eğitilir).<br><br>
        <b>config.py'yi değiştirip çalıştırırsanız fark görürsünüz:</b><br>
        • Tarih aralığını değiştirirseniz → farklı veri boyutu<br>
        • Yeni incident eklerseniz → modeller farklı risk örüntüleri öğrenir<br>
        • THRESHOLDS'u değiştirirseniz → risk etiketleri değişir, sınıflandırma farklılaşır<br>
        • SEED'i değiştirirseniz → farklı gürültü → farklı model performansları
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    with t5:
        st.subheader("Özellik Mühendisliği — 132 Özelliğin Dağılımı")

        feat_cats = pd.DataFrame([
            {"Kategori": "Ham metrik sütunları",       "Sayı": 9,  "Örnek": "cpu_usage_percent, api_latency_ms"},
            {"Kategori": "Ham log özellik sütunları",   "Sayı": 17, "Örnek": "total_errors, timeout_count, nova_api_errors"},
            {"Kategori": "Zaman özellikleri",           "Sayı": 7,  "Örnek": "hour_of_day, day_of_week, is_weekend"},
            {"Kategori": "Rolling mean (3/6/12/24/168h)","Sayı": 25, "Örnek": "cpu_usage_percent_rmean_24h"},
            {"Kategori": "Rolling std (3/6/12/24h)",    "Sayı": 20, "Örnek": "error_rate_percent_rstd_6h"},
            {"Kategori": "Lag özellikleri (1/6/24/48/168h)","Sayı": 35,"Örnek": "memory_usage_percent_lag_168h"},
            {"Kategori": "Delta (değişim hızı) özellikleri","Sayı": 9, "Örnek": "disk_usage_percent_delta_24h"},
            {"Kategori": "Log rolling (mean 6h / sum 24h)","Sayı": 10, "Örnek": "total_errors_rmean_6h"},
        ])
        feat_cats["Toplam"] = feat_cats["Sayı"].cumsum()
        st.dataframe(feat_cats, width='stretch', hide_index=True)
        st.caption("Toplam: 132 özellik. Kapasite tahmininde hedef sütunlar (cpu_future_24h vb.) özellik listesinden hariç tutulur.")

        st.divider()
        st.subheader("Model Hiperparametreleri")
        st.dataframe(pd.DataFrame([
            {"Model":       "LinearRegression",       "Görev": "Kapasite",  "Parametre": "default"},
            {"Model":       "RandomForestRegressor",  "Görev": "Kapasite",  "Parametre": "n_estimators=200, max_depth=12, min_samples_leaf=4"},
            {"Model":       "GradientBoostingRegressor","Görev":"Kapasite", "Parametre": "n_estimators=200, max_depth=5, lr=0.08, subsample=0.85"},
            {"Model":       "LogisticRegression",     "Görev": "Sağlık",   "Parametre": "C=0.5, class_weight=balanced, max_iter=2000"},
            {"Model":       "RandomForestClassifier", "Görev": "Sağlık",   "Parametre": "n_estimators=200, max_depth=12, class_weight=balanced"},
            {"Model":       "GradientBoostingClassifier","Görev":"Sağlık", "Parametre": "n_estimators=200, max_depth=5, lr=0.08, subsample=0.85"},
        ]), width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  VERİ GENEL GÖRÜNÜM
# ═══════════════════════════════════════════════════════════════════════════════
elif page.startswith("📊"):
    st.title("📊 Veri Genel Görünüm")
    st.markdown("""<div class="box">
    Her veri noktası bir <b>saatlik gözlem</b> dönemini temsil eder.
    Kırmızı gölgeli alanlar tanımlı <b>incident dönemleri</b> —
    bu dönemlerin öncesindeki buildup sinyalleri ML modelinin öğreneceği ipuçlarını barındırır.
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Metrik Zaman Serileri", "Log Analizi", "Ham Veri Önizleme"])

    with tab1:
        show_image(fig_path("metrics_overview.png"),
                   "CPU, Bellek, Disk, API Gecikmesi, Hata Oranı, VM Sayısı. "
                   "Kırmızı gölge = incident dönemi. Kesikli yatay çizgi = kritik eşik.")
        st.markdown("""<div class="box">
        <b>Grafiği nasıl okumalı?</b><br>
        • <b>Disk:</b> Monoton artış — log ve snapshot birikimi. Lineer ekstrapolasyon ile ne zaman dolacağı hesaplanır.<br>
        • <b>Bellek:</b> VM sayısı arttıkça yavaş artış trendi (+18% / 10 ay).<br>
        • <b>CPU:</b> Günlük ve haftalık örüntü belirgin; incident dönemlerinde ani spike.<br>
        • <b>Hata Oranı:</b> Normalde <%1, incident dönemlerinde %5–35 aralığına çıkar.
        </div>""", unsafe_allow_html=True)

        try:
            from config import INCIDENTS
            inc_df = pd.DataFrame([{
                "Olay":         inc["name"],
                "Başlangıç":   inc["start"],
                "Bitiş":       inc["end"],
                "Öncü (gün)":  inc["buildup_days"],
                "CPU +":       f"{inc['cpu_spike']:.0f}%",
                "Bellek +":    f"{inc['memory_spike']:.0f}%",
            } for inc in INCIDENTS])
            st.dataframe(inc_df, width='stretch', hide_index=True)
        except Exception:
            pass

    with tab2:
        show_image(fig_path("log_analysis.png"),
                   "Üst: servise göre saatlik hata (stacked alan). Alt: 10 ay toplam hata tipi dağılımı.")
        st.markdown("""<div class="box">
        Log verileri ML için <b>saatlik pencerede agregle</b> edilmiştir.
        Her saat için 6 servisin hata sayısı + 7 hata tipi sayımı + kritik log sayısı hesaplanır.
        Bu özellikler metrik verisiyle timestamp üzerinden birleştirilerek tek özellik matrisine dönüşür.
        </div>""", unsafe_allow_html=True)

    with tab3:
        try:
            mc  = os.path.join(DATA_DIR, "raw", "metrics.csv")
            lc  = os.path.join(DATA_DIR, "raw", "log_features.csv")
            rc  = os.path.join(DATA_DIR, "raw", "raw_logs.csv")
            c1,c2,c3 = st.columns(3)
            c1.metric("Metrik Satırı",   f"{len(pd.read_csv(mc)):,}")
            c2.metric("Log Özellik",     f"{len(pd.read_csv(lc)):,}")
            c3.metric("Ham Log Örneği",  f"{len(pd.read_csv(rc)):,}")
            with st.expander("metrics.csv — ilk 10 satır"):
                st.dataframe(pd.read_csv(mc).head(10), width='stretch')
            with st.expander("log_features.csv — ilk 10 satır"):
                st.dataframe(pd.read_csv(lc).head(10), width='stretch')
            with st.expander("raw_logs.csv — 15 rastgele satır"):
                raw = pd.read_csv(rc)
                st.dataframe(raw.sample(min(15, len(raw)), random_state=1), width='stretch')
        except Exception as e:
            st.warning(f"Veri yüklenemedi: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  KAPASİTE TAHMİNİ
# ═══════════════════════════════════════════════════════════════════════════════
elif page.startswith("📈"):
    st.title("📈 Kapasite Tahmini")
    st.markdown("""<div class="box">
    <b>3 model</b> (LR, RF, GBT) × <b>3 kaynak</b> (CPU, Bellek, Disk) × <b>2 ufuk</b> (24h, 72h) = 18 model.
    Eğitim: Eyl 2025–Mar 2026 · Test: Nis–Haz 2026 (zamansal bölünmüş).
    ★ simgesi: her kaynak–ufuk çifti için en düşük RMSE.
    </div>""", unsafe_allow_html=True)

    cap = load_json("capacity_results.json")
    if not cap:
        st.warning("Kapasite sonuçları bulunamadı.")
        st.stop()

    _res_tr = {
        "cpu_usage_percent":    "CPU (%)",
        "memory_usage_percent": "Bellek (%)",
        "disk_usage_percent":   "Disk (%)",
    }

    rows = []
    for key, val in cap.items():
        for mname, mv in val["models"].items():
            rows.append({
                "Kaynak": _res_tr.get(val["resource"], val["resource"]),
                "Ufuk":   f"{val['horizon_h']}s",
                "Model":  mname,
                "MAE":    round(mv["MAE"],  3),
                "RMSE":   round(mv["RMSE"], 3),
                "R²":     round(mv["R2"],   4),
                "MAPE":   round(mv.get("MAPE", 0), 2),
            })
    df_res = pd.DataFrame(rows)

    tab24, tab72 = st.tabs(["24 Saatlik Ufuk", "72 Saatlik Ufuk"])

    for tab, horizon in [(tab24, "24s"), (tab72, "72s")]:
        with tab:
            sub = df_res[df_res["Ufuk"] == horizon].drop(columns="Ufuk").copy()
            best_set = set()
            for kaynak in sub["Kaynak"].unique():
                idx = sub[sub["Kaynak"] == kaynak]["RMSE"].idxmin()
                best_set.add(idx)
            sub[""] = sub.index.map(lambda i: "★ EN İYİ" if i in best_set else "")
            st.dataframe(
                sub,
                column_config={
                    "MAE":  st.column_config.NumberColumn("MAE",  format="%.3f"),
                    "RMSE": st.column_config.NumberColumn("RMSE", format="%.3f"),
                    "R²":   st.column_config.NumberColumn("R²",   format="%.4f"),
                    "MAPE": st.column_config.NumberColumn("MAPE", format="%.2f"),
                    "":     st.column_config.TextColumn("",       width="small"),
                },
                width='stretch', hide_index=True,
            )

    st.markdown("""<div class="box">
    <b>MAE</b>: Ortalama Mutlak Hata (puan) ·
    <b>RMSE</b>: Kök Ortalama Kare Hata (büyük sapmaları cezalandırır) ·
    <b>R²</b>: 1=mükemmel, 0=ortalama, negatif=ortalamadan kötü ·
    <b>MAPE</b>: Yüzde bazlı hata<br>
    Disk R²<0: monoton artış + test setindeki spike'lar eğitim setinde görülmemiş → lineer ekstrapolasyon tamamlayıcı olarak kullanılır.
    </div>""", unsafe_allow_html=True)

    show_image(fig_path("capacity_forecasts.png"),
               "Gerçek (mavi) ve tahmin (kırmızı kesikli). Her satır bir kaynak; sol 24h, sağ 72h.")
    show_image(fig_path("capacity_model_comparison.png"),
               "Her kaynak için üç modelin RMSE karşılaştırması.")


# ═══════════════════════════════════════════════════════════════════════════════
#  SAĞLIK SINIFLANDIRMASI
# ═══════════════════════════════════════════════════════════════════════════════
elif page.startswith("🩺"):
    st.title("🩺 Sağlık Riski Sınıflandırması")
    st.markdown("""<div class="box">
    <b>Soru:</b> Sonraki 4 saatte herhangi bir kritik eşik aşılacak mı? (0=Güvenli, 1=Riskli)<br>
    Risk oranı ~%12 → sınıf dengesizliği <code>class_weight='balanced'</code> ile ele alınır.
    Birincil değerlendirme metriği: <b>F1 skoru</b>.
    </div>""", unsafe_allow_html=True)

    health = load_json("health_results.json")
    if not health:
        st.warning("Sağlık sonuçları bulunamadı.")
        st.stop()

    best_f1 = max(v["F1"] for v in health["models"].values())
    rows = []
    for mname, mv in health["models"].items():
        rows.append({
            "Model":     mname,
            "Accuracy":  round(mv["Accuracy"],  3),
            "Precision": round(mv["Precision"], 3),
            "Recall":    round(mv["Recall"],    3),
            "F1 Skoru":  round(mv["F1"],        3),
            "AUC-ROC":   round(mv["AUC-ROC"],   3),
            "":          "★ EN İYİ" if mv["F1"] == best_f1 else "",
        })
    df_h = pd.DataFrame(rows)
    st.dataframe(
        df_h,
        column_config={
            "Accuracy":  st.column_config.NumberColumn(format="%.3f"),
            "Precision": st.column_config.NumberColumn(format="%.3f"),
            "Recall":    st.column_config.NumberColumn(format="%.3f"),
            "F1 Skoru":  st.column_config.NumberColumn(format="%.3f"),
            "AUC-ROC":   st.column_config.NumberColumn(format="%.3f"),
            "":          st.column_config.TextColumn("", width="small"),
        },
        width='stretch', hide_index=True,
    )

    st.markdown("""<div class="box">
    <b>Precision</b>: "Riskli" dediğinde ne kadar haklı? (az yanlış alarm) ·
    <b>Recall</b>: Gerçek risklerin kaçı yakalandı? (az kaçan incident) ·
    <b>F1</b>: İkisinin dengesi · <b>AUC-ROC</b>: Tüm eşiklerde ayrıştırma gücü
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        show_image(fig_path("health_roc_curves.png"), "Eğri sola-yukarı = daha iyi. Kesikli = rastgele (AUC=0.5).")
    with c2:
        show_image(fig_path("health_confusion_matrices.png"), "Satır=gerçek, sütun=tahmin.")

    show_image(fig_path("health_risk_timeline.png"),
               "Üst: gerçek risk etiketleri. Alt: en iyi modelin her saat için risk olasılığı (0–1).")
    st.markdown("""<div class="box">
    İyi bir model risk olasılığını incident başlamadan <b>birkaç saat öncesinde</b> yükseltir —
    bu proaktif izlemenin temel göstergesidir. Olasılık 0.5 eşiğini aşınca "riskli" kararı üretilir.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SHAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page.startswith("🔍"):
    st.title("🔍 SHAP Açıklanabilirlik Analizi")
    st.markdown("""<div class="box">
    <b>SHAP (SHapley Additive exPlanations)</b>: Oyun teorisinden türetilen Shapley değerleriyle
    her özelliğin modelin tahminini ortalama beklentiden ne yönde ve ne kadar sapdırdığı hesaplanır.
    Ağaç tabanlı modeller için <code>TreeExplainer</code> kullanılır (verimli, model-içi hesaplama).
    Test setinden 500 örneklem üzerinde çalışır.
    </div>""", unsafe_allow_html=True)

    tab_cpu, tab_mem, tab_disk, tab_sag = st.tabs([
        "CPU Tahmini", "Bellek Tahmini", "Disk Tahmini", "Sağlık Riski",
    ])

    def _shap_sec(res_key, horizon, title, yorum):
        show_image(fig_path(f"shap_capacity_bar_{res_key}_{horizon}h.png"),
                   f"{title} — En Önemli 20 Özellik (bar)")
        show_image(fig_path(f"shap_capacity_beeswarm_{res_key}_{horizon}h.png"),
                   "Her nokta=test örneği. Renk=özellik değeri (kırmızı=yüksek). "
                   "Yatay=SHAP katkısı (sağ=tahmin artar).")
        show_image(fig_path(f"shap_capacity_dep_{res_key}_{horizon}h.png"),
                   "X=özellik değeri, Y=SHAP katkısı. Pozitif SHAP → tahmin artar.")
        csv_p = os.path.join(REP_DIR, f"shap_capacity_{res_key}_{horizon}h.csv")
        if os.path.exists(csv_p):
            with st.expander("SHAP Özellik Önem Tablosu (tüm 132 özellik)"):
                df_s = pd.read_csv(csv_p)
                df_s["importance"] = df_s["importance"].round(4)
                st.dataframe(df_s.head(30), width='stretch', hide_index=True)
        st.markdown(f'<div class="box">{yorum}</div>', unsafe_allow_html=True)

    with tab_cpu:
        h = st.radio("Ufuk", [24, 72], horizontal=True, key="sh_cpu")
        _shap_sec("cpu_usage_percent", h, f"CPU — {h}s ufku",
                  "CPU tahmini için <b>kısa vadeli hareketli ortalama</b> (cpu_rmean_3h, cpu_lag_1h) "
                  "en yüksek SHAP değerini alır: CPU kullanımı güçlü süreklilik gösterir. "
                  "Incident dönemlerinde <b>log hata sayımları ve gecikme metrikleri</b> devreye girer.")

    with tab_mem:
        h2 = st.radio("Ufuk", [24, 72], horizontal=True, key="sh_mem")
        _shap_sec("memory_usage_percent", h2, f"Bellek — {h2}s ufku",
                  "Bellek yavaş artış trendine sahip olduğundan <b>week_of_year</b> ve "
                  "<b>168h hareketli ortalama</b> önemli rol oynar. "
                  "VM yaratma patlamalarında <b>vm_count lag</b> özellikleri de belirleyici.")

    with tab_disk:
        _shap_sec("disk_usage_percent", 72, "Disk — 72s ufku",
                  "<b>disk_rmean_168h</b> ve <b>week_of_year</b> temel belirleyici: "
                  "disk monoton artar. Negatif R² disk tahmininin zorluğunu gösterir; "
                  "lineer ekstrapolasyon bu modeli tamamlamak için kullanılır.")

    with tab_sag:
        show_image(fig_path("shap_health_bar.png"), "Sağlık Riski — En Önemli 20 Özellik")
        show_image(fig_path("shap_health_beeswarm.png"),
                   "Kırmızı=yüksek özellik değeri risk olasılığını artırıyor; mavi=azaltıyor.")
        show_image(fig_path("shap_health_dep.png"), "En etkili 2 özelliğin bağımlılık grafiği.")
        csv_h = os.path.join(REP_DIR, "shap_health.csv")
        if os.path.exists(csv_h):
            with st.expander("SHAP Özellik Önem Tablosu — Sağlık Riski"):
                df_sh = pd.read_csv(csv_h)
                df_sh["importance"] = df_sh["importance"].round(4)
                st.dataframe(df_sh.head(30), width='stretch', hide_index=True)
        st.markdown("""<div class="box">
        Sağlık riskinde en belirleyici özellik <b>error_rate_percent</b> ve log türevleri
        (<b>timeout_count</b>, <b>db_connection_count</b>). Bu, risk tespitinin
        yalnızca kaynak metriğiyle değil, <b>servis log verisiyle birlikte</b>
        yapılması gerektiğinin somut SHAP kanıtıdır.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ÖNERİLER
# ═══════════════════════════════════════════════════════════════════════════════
elif page.startswith("💡"):
    st.title("💡 Operasyonel Öneriler")
    st.markdown("""<div class="box">
    Model çıktıları + SHAP top özelliği → kural motoru → önceliklendirilmiş öneriler.<br>
    <b>3 kaynak:</b> Kapasite trendi (72s lineer ekstrapolasyon) ·
    Sağlık riski (48s pencere ortalaması) · Trend bozulması (7g/7g karşılaştırma).
    </div>""", unsafe_allow_html=True)

    recs = load_json("recommendations.json")
    if not recs:
        st.warning("Öneri dosyası bulunamadı.")
        st.stop()

    n_c = sum(1 for r in recs if r.get("severity") == "critical")
    n_w = sum(1 for r in recs if r.get("severity") == "warning")
    n_i = sum(1 for r in recs if r.get("severity") == "info")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Toplam", len(recs))
    c2.metric("🔴 Kritik", n_c)
    c3.metric("🟡 Uyarı",  n_w)
    c4.metric("🔵 Bilgi",  n_i)
    st.divider()

    _sm = {
        "critical": ("🔴","KRITIK","rec-critical","badge-c"),
        "warning":  ("🟡","UYARI", "rec-warning", "badge-w"),
        "info":     ("🔵","BİLGİ","rec-info",    "badge-i"),
    }
    _cat = {"KAPASITE":"Kaynak Kapasitesi","SAĞLIK":"Servis Sağlığı","TREND":"Metrik Trendi"}

    for i, rec in enumerate(recs, 1):
        sev = rec.get("severity","info")
        icon, label, css, badge = _sm.get(sev, ("🔵","BİLGİ","rec-info","badge-i"))
        acts = "".join(f"<li>{a}</li>" for a in rec.get("actions",[]))
        st.markdown(f"""
        <div class="{css}">
          <div style="margin-bottom:6px;">
            {icon} <span class="{badge}">{label}</span>
            <span style="font-size:.82rem;color:#6b7280;"> #{i} · {_cat.get(rec.get('category',''),rec.get('category',''))}</span>
          </div>
          <div style="font-weight:600;font-size:1.05rem;margin-bottom:6px;">{rec.get('title','')}</div>
          <div style="font-size:.92rem;color:#374151;margin-bottom:8px;">{rec.get('detail','')}</div>
          <b>Önerilen Aksiyonlar:</b>
          <ol style="margin:4px 0 0 18px;padding:0;">{acts}</ol>
          <div style="font-size:.78rem;color:#9ca3af;margin-top:8px;">
            Oluşturulma: {rec.get('timestamp','')}
          </div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    txt_path = os.path.join(REP_DIR, "recommendations.txt")
    if os.path.exists(txt_path):
        with st.expander("Metin Raporu"):
            st.code(open(txt_path, encoding="utf-8").read(), language=None)
        with open(txt_path, "rb") as f:
            st.download_button("📥 Raporu İndir (.txt)", f, file_name="recommendations.txt")
