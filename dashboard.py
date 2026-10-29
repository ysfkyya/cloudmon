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
    return os.path.exists(os.path.join(REP_DIR, "capacity_results.json"))


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
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("☁️ Proaktif İzleme")
    st.caption("OpenStack · ML Pipeline · SHAP")
    st.divider()

    page = st.radio("Sayfa", [
        "🏠  Ana Sayfa",
        "📖  Proje Hakkında",
        "📊  Veri Genel Görünüm",
        "📈  Kapasite Tahmini",
        "🩺  Sağlık Sınıflandırması",
        "🔍  SHAP Açıklanabilirlik",
        "💡  Operasyonel Öneriler",
        "🤖  LLM Öneri Karşılaştırması",
    ], label_visibility="collapsed")

    st.divider()
    if st.button("🔄  Pipeline'ı Yeniden Çalıştır", use_container_width=True):
        with st.spinner("Pipeline çalışıyor…"):
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
#  PROJe HAKKINDA — README.md görüntüleyici
# ═══════════════════════════════════════════════════════════════════════════════
if page.startswith("📖"):
    import re
    readme_path = os.path.join(BASE, "README.md")

    st.markdown("""
    <style>
    .readme-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 50%, #1a4a7a 100%);
        padding: 32px 36px; border-radius: 12px; margin-bottom: 20px; color: white;
    }
    .readme-header h1 { color: white; margin: 0 0 8px 0; font-size: 1.8rem; }
    .readme-header p  { color: #c9dff5; margin: 0; font-size: 1rem; }
    .readme-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.35);
        color: white; padding: 3px 12px; border-radius: 20px;
        font-size: .80rem; margin: 6px 4px 0 0;
    }
    .paper-card {
        background: #f8fafc; border-radius: 10px; padding: 16px 18px;
        margin-bottom: 14px; border-left: 4px solid #3b82f6;
        border: 1px solid #e2e8f0;
    }
    .paper-card.aiops    { border-left-color: #7c3aed; }
    .paper-card.anomaly  { border-left-color: #dc2626; }
    .paper-card.capacity { border-left-color: #059669; }
    .paper-card.xai      { border-left-color: #d97706; }
    .paper-card.log      { border-left-color: #0891b2; }
    .paper-card.llm      { border-left-color: #7c3aed; }
    .paper-card.openstack{ border-left-color: #be185d; }
    .paper-title  { font-weight: 700; font-size: .97rem; color: #1e293b; margin-bottom: 4px; }
    .paper-meta   { font-size: .80rem; color: #64748b; margin-bottom: 8px; }
    .paper-body   { font-size: .88rem; color: #374151; }
    .paper-rel    { margin-top: 8px; font-size: .83rem; background: #eff6ff;
                    border-radius: 6px; padding: 6px 10px; color: #1d4ed8; }
    .contrib-card {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border-radius: 10px; padding: 18px 20px; margin-bottom: 14px;
        border-left: 5px solid #16a34a;
    }
    .contrib-title { font-weight: 700; font-size: 1.05rem; color: #14532d; margin-bottom: 8px; }
    .contrib-body  { font-size: .90rem; color: #374151; }
    .rq-card {
        background: #fefce8; border-radius: 8px; padding: 12px 16px;
        margin-bottom: 10px; border-left: 4px solid #ca8a04;
        font-size: .90rem; color: #374151;
    }
    </style>
    <div class="readme-header">
      <h1>☁️ CloudMon</h1>
      <p>Bulut Altyapısı Proaktif İzleme ve Karar Destek Sistemi</p>
      <div style="margin-top:12px;">
        <span class="readme-badge">🐍 Python 3.10+</span>
        <span class="readme-badge">🤖 ML Pipeline</span>
        <span class="readme-badge">📊 SHAP</span>
        <span class="readme-badge">☁️ OpenStack</span>
        <span class="readme-badge">🔮 LLM Entegrasyonu</span>
        <span class="readme-badge">📈 27 Model</span>
        <span class="readme-badge">📚 6 Araştırma Alanı</span>
        <span class="readme-badge">🏆 6 Özgün Katkı</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_readme, tab_lit, tab_contrib = st.tabs([
        "📋 Proje Dokümantasyonu",
        "📚 Literatür Taraması",
        "🏆 Özgün Katkılar",
    ])

    # ── Sekme 1: README ──────────────────────────────────────────────────────────
    with tab_readme:
        if os.path.exists(readme_path):
            with open(readme_path, encoding="utf-8") as f:
                readme_content = f.read()
            readme_rendered = re.sub(
                r"```mermaid\n(.*?)```",
                lambda m: "```\n" + m.group(1) + "```",
                readme_content, flags=re.DOTALL,
            )
            with st.expander("📋 İçindekiler", expanded=False):
                toc_items = [
                    "🎯 Motivasyon ve Problem Tanımı", "🏗️ Sistem Mimarisi",
                    "🔄 Pipeline Akışı", "📥 Veri Katmanı",
                    "⚙️ Özellik Mühendisliği", "📈 Kapasite Tahmini",
                    "🩺 Sağlık Riski Sınıflandırması", "🔍 SHAP Açıklanabilirlik",
                    "💡 Öneri Motoru", "🤖 LLM Entegrasyonu",
                    "📊 Model Performans Sonuçları", "🗂️ Proje Yapısı",
                    "🚀 Kurulum ve Çalıştırma", "📚 Literatür Taraması",
                    "🏆 Özgün Katkılar",
                ]
                cols_toc = st.columns(2)
                for i, label in enumerate(toc_items):
                    cols_toc[i % 2].markdown(f"- {label}")
            st.markdown(readme_rendered)
            st.divider()
            with open(readme_path, "rb") as f_dl:
                st.download_button(
                    "📥 README.md İndir", data=f_dl,
                    file_name="CloudMon_README.md", mime="text/markdown",
                    use_container_width=True,
                )
        else:
            st.error("README.md dosyası bulunamadı.")

    # ── Sekme 2: Literatür Taraması ──────────────────────────────────────────────
    with tab_lit:
        st.markdown("""<div class="box">
        CloudMon altı farklı araştırma alanının kesişiminde konumlanır.
        Aşağıda her alan için temel akademik çalışmalar ve bu çalışmaların
        CloudMon tasarımıyla ilişkisi açıklanmaktadır.
        </div>""", unsafe_allow_html=True)

        PAPERS = {
            "🟣 AIOps — Yapay Zekâ Destekli BT Operasyonları": {
                "color_cls": "aiops",
                "desc": "AIOps, büyük ölçekli BT sistemlerinin log, metrik ve olay verilerini makine "
                        "öğrenmesiyle analiz ederek operasyonel zekâ üretmesini sağlayan disiplindir. "
                        "2017'de Gartner tarafından kavramsallaştırılmış, 2019'dan itibaren akademik "
                        "literatürde yerini sağlamlaştırmıştır.",
                "papers": [
                    {
                        "title": "AIOps: Real-World Challenges and Research Innovations",
                        "meta":  "Dang et al. · ICSE 2019 · Microsoft Research",
                        "body":  "AIOps'un endüstriyel zorlukları ve açık araştırma soruları; "
                                 "noise reduction, anomaly detection ve root cause localization "
                                 "problemi Microsoft ölçeğinde tanımlanmıştır.",
                        "rel":   "CloudMon öneri motorunu motive eden 'noise reduction' ve "
                                 "'root cause localization' problemleri bu çalışmada ilk kez sistematik biçimde ele alınmıştır.",
                    },
                    {
                        "title": "Outage Prediction and Diagnosis for Cloud Service Systems",
                        "meta":  "Chen et al. · WWW 2019 · Microsoft Azure",
                        "body":  "Azure bulut servis kesintilerini tahmin eden hibrit ML yaklaşımı; "
                                 "metrik + log birleşimi ile iki aşamalı (tahmin + tanı) mimari.",
                        "rel":   "CloudMon'un iki-katmanlı mimarisine (kapasite tahmini + sağlık "
                                 "sınıflandırması) doğrudan ilham kaynağı olmuştur.",
                    },
                    {
                        "title": "Anomaly Detection and Failure Root Cause Analysis in Cloud Applications: A Survey",
                        "meta":  "Soldani & Brogi · ACM Computing Surveys 2022",
                        "body":  "Bulut-tabanlı mikroservis sistemlerinde 40+ anomali tespit yöntemi "
                                 "karşılaştırması; metrik tabanlı, log tabanlı ve izleme tabanlı "
                                 "yaklaşımların güçlü ve zayıf yönleri.",
                        "rel":   "CloudMon'un log+metrik fusion yaklaşımının alan içindeki "
                                 "konumlanması için temel referans tarama çalışmasıdır.",
                    },
                ],
            },
            "🔴 Anomali Tespiti ve Erken Uyarı": {
                "color_cls": "anomaly",
                "desc": "Bu alandaki çalışmalar bir sistemin 'normal' durumundan sapmasını tespit "
                        "etmeye odaklanır. CloudMon'un sağlık sınıflandırıcısı bu literatürün "
                        "doğrudan bir mirasçısıdır.",
                "papers": [
                    {
                        "title": "Anomaly Detection: A Survey",
                        "meta":  "Chandola, Banerjee & Kumar · ACM Computing Surveys 2009",
                        "body":  "Anomali tespitinin temel sınıflandırması: nokta anomali, bağlamsal "
                                 "anomali, kolektif anomali. İstatistiksel, mesafe ve sınıflandırma "
                                 "tabanlı yöntemlerin kapsamlı karşılaştırması.",
                        "rel":   "CloudMon'un risk etiketi 'bağlamsal anomali' kategorisine girer: "
                                 "anlık eşik değil, 4 saatlik zaman penceresi içinde koşullu risk tespiti.",
                    },
                    {
                        "title": "DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning",
                        "meta":  "Du et al. · CCS 2017 · UC Davis",
                        "body":  "LSTM ile sistem loglarında sıra anomalisi tespiti. Log şablonlarını "
                                 "otomatik öğrenir, anormal log geçişlerini gerçek zamanlı tespit eder.",
                        "rel":   "CloudMon log verilerini sıra değil sayım olarak işler; DeepLog'un "
                                 "template gereksinimi yerine agregatör yaklaşım benimsenerek "
                                 "yapısal esneklik ve yorumlanabilirlik kazanılmıştır.",
                    },
                    {
                        "title": "A Survey on Automated Log Analysis for Reliability Engineering",
                        "meta":  "He et al. · ACM Computing Surveys 2021",
                        "body":  "Log parsing, anomaly detection, failure diagnosis ve prediction "
                                 "üzerine 200+ makale taraması. Aggregation tabanlı ve template "
                                 "tabanlı yaklaşımların karşılaştırması.",
                        "rel":   "CloudMon'un log_features.csv tasarımı bu taramanın "
                                 "'aggregation-based' yaklaşım sınıfına girmektedir: saatlik sayım "
                                 "agregasyonu, template değişikliklerine doğal olarak dayanıklıdır.",
                    },
                ],
            },
            "🟢 Bulut Kapasite Planlaması ve İş Yükü Tahmini": {
                "color_cls": "capacity",
                "desc": "Kapasite planlaması, uzun vadeli kaynak gereksinimini öngörerek doğru "
                        "zamanda genişleme kararı verme disiplinidir. Bu alan CloudMon'un en "
                        "doğrudan katkı sunduğu alandır.",
                "papers": [
                    {
                        "title": "Workload Prediction Using ARIMA Model and Its Impact to Resource Provisioning and Cost Parameter in Cloud Computing Environments",
                        "meta":  "Calheiros et al. · IEEE Transactions on Parallel and Distributed Systems 2015",
                        "body":  "ARIMA ile CloudSim simülasyonunda kaynak tahsisi optimizasyonu. "
                                 "Mevsimsel örüntülerin tespiti ve öngörülü ölçekleme stratejisi.",
                        "rel":   "CloudMon tree-based ML tercih eder: ARIMA stasyonerlik varsayımını "
                                 "gerektirir; OpenStack iş yükleri yoğun kullanım dönemleri + donanım alımları "
                                 "+ ani yük artışları içerdiğinden ARIMA tek başına yetersiz kalır.",
                    },
                    {
                        "title": "A Review of Auto-Scaling Techniques for Elastic Applications in Cloud Environments",
                        "meta":  "Lorido-Botran et al. · Journal of Grid Computing 2014",
                        "body":  "Reaktif, öngörülü ve hibrit elastik ölçekleme stratejileri. "
                                 "Öngörülü ölçekleme için gerekli tahmin modeli gereksinimleri analizi.",
                        "rel":   "CloudMon'un 'eşiğe kaç gün' çıktısı doğrudan öngörülü ölçekleme "
                                 "kararlarını besler. Bu çalışma planlama ufkunun neden önemli "
                                 "olduğunu teorik olarak ortaya koyar.",
                    },
                    {
                        "title": "An Empirical Study of Deep Learning Models for Workload Prediction in Cloud Data Centers",
                        "meta":  "Chen et al. · IEEE ICDCS 2021",
                        "body":  "LSTM, GRU, Transformer gibi derin öğrenme modellerinin iş yükü "
                                 "tahmininde kapsamlı karşılaştırması. Veri boyutu ve model "
                                 "karmaşıklığı arasındaki denge analizi.",
                        "rel":   "CloudMon Gradient Boosting tercih eder: ~13 000 örnekte derin "
                                 "öğrenme overfitting riskiyle karşılaşır ve yorumlanabilirlik "
                                 "avantajı kaybolur. Bu çalışma tercih gerekçesini destekler.",
                    },
                ],
            },
            "🟡 Açıklanabilir Yapay Zekâ (XAI)": {
                "color_cls": "xai",
                "desc": "Makine öğrenmesi modelleri 'kara kutu' olduğunda operatör güvenmez ve "
                        "aksiyona geçmez. XAI araştırmaları bu güven açığını kapatmaya çalışır.",
                "papers": [
                    {
                        "title": "A Unified Approach to Interpreting Model Predictions",
                        "meta":  "Lundberg & Lee · NeurIPS 2017 · University of Washington",
                        "body":  "SHAP (SHapley Additive exPlanations) yönteminin önerilmesi. "
                                 "Oyun teorisindeki Shapley değeri kavramı makine öğrenmesi "
                                 "açıklanabilirliğine uyarlanmıştır. Tutarlılık ve yerellik "
                                 "aksiyomlarını aynı anda sağlayan ilk yöntem.",
                        "rel":   "CloudMon SHAP'ı doğrudan ve merkezi biçimde kullanır. "
                                 "Bu çalışma tüm açıklanabilirlik katmanının temel referansıdır.",
                    },
                    {
                        "title": "From Local Explanations to Global Understanding with Explainable AI for Trees",
                        "meta":  "Lundberg et al. · Nature Machine Intelligence 2020",
                        "body":  "TreeExplainer: ağaç tabanlı modeller için O(TLD) verimli "
                                 "SHAP hesabı. Polynomial yerine polinom-altı zaman karmaşıklığı "
                                 "sayesinde büyük ağaçlarda uygulanabilir hale gelmiştir.",
                        "rel":   "CloudMon TreeExplainer tercih eder. 500 test örneği ile "
                                 "makul sürede (≈40 sn) tamamlanması bu verimlilik sayesindedir.",
                    },
                    {
                        "title": "DARPA's Explainable Artificial Intelligence (XAI) Program",
                        "meta":  "Gunning & Aha · AI Magazine 2019 · DARPA",
                        "body":  "XAI'nin askeri, sağlık ve kamu sektörü uygulamalarında "
                                 "güven ve şeffaflık gereksinimleri. İnsan-makine güven döngüsü "
                                 "modeli ve değerlendirme çerçevesi.",
                        "rel":   "CloudMon aynı güven gereksinimini operasyonel BT'ye taşır: "
                                 "'neden alarm?' sorusu yanıtlanmadan operatör aksiyona geçmez. "
                                 "Bu çalışma CloudMon'un varoluş gerekçesini meşrulaştıran "
                                 "üst-düzey referanstır.",
                    },
                ],
            },
            "🔵 LLM ile BT Operasyonları": {
                "color_cls": "llm",
                "desc": "Büyük dil modellerinin SRE (Site Reliability Engineering) ve BT "
                        "yönetimine entegrasyonu 2022'den itibaren hız kazanmıştır. "
                        "CloudMon bu alanda en güncel çalışmalarla paraleldir.",
                "papers": [
                    {
                        "title": "Recommending Root-Cause and Mitigation Steps for Cloud Incidents using Large Language Models",
                        "meta":  "Ahmed et al. · ICSE 2023 · Microsoft",
                        "body":  "GPT-4 ile Azure incident'larının kök neden analizi ve "
                                 "düzeltme adımı önerisi. İnsan uzman önerilerine kıyasla "
                                 "%22 F1 artışı elde edilmiştir.",
                        "rel":   "CloudMon aynı hedefi taşır ancak yapılandırılmış bağlam "
                                 "(SHAP + kural motoru çıktısı + log içgörüleri) ile prompt kalitesini "
                                 "artırır; LLM'e ham metrik değil yorumlanmış bağlam verilir.",
                    },
                    {
                        "title": "LILAC: Log Parsing using LLMs with Adaptive Parsing Cache",
                        "meta":  "Jiang et al. · ICSE 2024",
                        "body":  "GPT-4 ile sıfır-atış log ayrıştırma ve adaptif önbellek "
                                 "mekanizması. Önceden etiket gerektirmeyen dinamik log parsing.",
                        "rel":   "CloudMon'un LLM katmanı bu alandaki en son trendi yansıtır. "
                                 "Log içeriğini doğrudan LLM'e göndermek yerine agregat özellikler "
                                 "gönderilir — token verimliliği ve tutarlılık için.",
                    },
                    {
                        "title": "Owl: A Large Language Model for IT Operations",
                        "meta":  "Guo et al. · arXiv 2024",
                        "body":  "BT operasyonları için ince ayar yapılmış özel LLM. "
                                 "Log + metrik + komut geçmişi ile eğitim. Genel amaçlı "
                                 "LLM'lere kıyasla alana özgü performans artışı.",
                        "rel":   "CloudMon genel amaçlı LLM'leri (Claude, GPT-4o) kullanır. "
                                 "Bu çalışmayla zıtlık kurar: özel fine-tuning gerektirmeden "
                                 "yapılandırılmış bağlamla operasyonel değer üretilebildiğini gösterir.",
                    },
                    {
                        "title": "GPT-4 is Not Yet a Human Expert in Cybersecurity: A Benchmarking Study",
                        "meta":  "Wang et al. · arXiv 2023",
                        "body":  "LLM'lerin güvenlik ve altyapı uzman kararlarındaki "
                                 "sınırlılıklarının nicel analizi. Bağlam kalitesinin LLM "
                                 "performansına etkisi.",
                        "rel":   "CloudMon bu sınırlılığı kabul ederek LLM'i kural motorunun "
                                 "üzerine bir katman olarak konumlar, tek karar vericisi yapmaz. "
                                 "Hibrit yaklaşım bu çalışmanın bulgularıyla örtüşür.",
                    },
                ],
            },
            "🔵 OpenStack ve Bulut Altyapısı İzleme": {
                "color_cls": "openstack",
                "desc": "OpenStack'e özgü izleme ve yönetim çalışmaları — CloudMon'un "
                        "doğrudan uygulama alanı.",
                "papers": [
                    {
                        "title": "Orchestration of Monitoring for IaaS Cloud Platforms: OpenStack Case Study",
                        "meta":  "Alam et al. · IEEE CLOUD 2017",
                        "body":  "Prometheus + OpenStack entegrasyonunun sistematik analizi. "
                                 "Compute, network ve storage kaynaklarının izleme mimarisi.",
                        "rel":   "CloudMon veri kaynağı olarak aynı Prometheus + OpenSearch "
                                 "mimarisini benimser. Bu çalışma gerçek entegrasyon için "
                                 "referans mimari sağlar.",
                    },
                    {
                        "title": "Detecting and Localizing End-to-End Performance Degradation for Cellular Networks",
                        "meta":  "Chiang et al. · IEEE INFOCOM 2017",
                        "body":  "Ağ performans bozunumunun yapısal analizi, bozunuma katkıda "
                                 "bulunan servis bileşenlerinin tespiti. Kısmen OpenStack'e "
                                 "uyarlanabilir yöntemler.",
                        "rel":   "CloudMon API latency bozunumunu trend analizi olarak modellemekte "
                                 "bu çalışmadan ilham alır. 'Bozunuma neden katkıda bulunanlar?' "
                                 "sorusu SHAP ile yanıtlanmaktadır.",
                    },
                    {
                        "title": "Drain: An Online Log Parsing Approach with Fixed Depth Tree",
                        "meta":  "He et al. · IEEE ICDM 2017",
                        "body":  "Gerçek zamanlı log ayrıştırma için sabit derinlikli ağaç "
                                 "yapısı. 1 ms/satır performansı ile OpenStack gibi yüksek "
                                 "hacimli log ortamlarında kullanılabilir.",
                        "rel":   "CloudMon log parsing yerine saatlik sayım agregasyonunu tercih "
                                 "eder — template güncellenmesi gerektirmeyen daha basit "
                                 "ve yorumlanabilir yaklaşım.",
                    },
                ],
            },
        }

        for area_title, area_data in PAPERS.items():
            st.markdown(f"### {area_title}")
            st.markdown(f'<div class="box">{area_data["desc"]}</div>',
                        unsafe_allow_html=True)
            for p in area_data["papers"]:
                st.markdown(f"""
                <div class="paper-card {area_data['color_cls']}">
                  <div class="paper-title">📄 {p['title']}</div>
                  <div class="paper-meta">{p['meta']}</div>
                  <div class="paper-body">{p['body']}</div>
                  <div class="paper-rel">🔗 <b>CloudMon ile İlişki:</b> {p['rel']}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("")

    # ── Sekme 3: Özgün Katkılar ──────────────────────────────────────────────────
    with tab_contrib:
        st.markdown("""<div class="box">
        Mevcut literatür incelendiğinde CloudMon'un altı temel alanda özgün katkı sunduğu görülmektedir.
        Aşağıdaki karşılaştırma tablosu ve katkı açıklamaları bu konumlandırmayı netleştirir.
        </div>""", unsafe_allow_html=True)

        st.subheader("Konumlandırma Karşılaştırması")
        pos_df = pd.DataFrame([
            {"Boyut": "Analiz kapsamı",       "Geleneksel Yaklaşım": "Tek katman (anomali VEYA kapasite)",  "CloudMon": "6 katman entegre pipeline"},
            {"Boyut": "Veri kaynağı",         "Geleneksel Yaklaşım": "Metrik VEYA log (ayrı ayrı)",         "CloudMon": "Metrik + log zaman hizalaması → tek model"},
            {"Boyut": "Açıklanabilirlik",     "Geleneksel Yaklaşım": "Opsiyonel, sonradan eklenen",          "CloudMon": "Zorunlu; SHAP çıktısı aksiyona dönüştürülmüş"},
            {"Boyut": "Alan bilgisi",         "Geleneksel Yaklaşım": "Domain-agnostic tasarım",              "CloudMon": "Staj takvimi + donanım alımları entegre"},
            {"Boyut": "LLM kullanımı",        "Geleneksel Yaklaşım": "Yok veya tek model",                  "CloudMon": "İki model (Claude + GPT-4o) karşılaştırmalı"},
            {"Boyut": "Veri kaynağı",          "Geleneksel Yaklaşım": "Kara kutu veya gizli veri",            "CloudMon": "Prometheus + OpenSearch entegrasyonu, data/raw/ üzerinden"},
            {"Boyut": "Hedef çıktı",          "Geleneksel Yaklaşım": "Anomali etiketi veya tahmin değeri",  "CloudMon": "Önceliklendirilmiş aksiyon listesi"},
            {"Boyut": "Tahmin ufku",          "Geleneksel Yaklaşım": "Tek ufuk (genellikle kısa vadeli)",   "CloudMon": "24h / 72h / 168h çoklu ufuk"},
        ])
        st.dataframe(pos_df, width="stretch", hide_index=True)

        st.divider()
        st.subheader("Özgün Katkılar")

        CONTRIBS = [
            {
                "no": "1",
                "title": "Uçtan Uca Entegre Pipeline",
                "body": """Literatürdeki çalışmaların büyük çoğunluğu yalnızca <b>tek bir soruya</b> odaklanır:
                anomali tespiti <i>veya</i> kapasite tahmini <i>veya</i> açıklanabilirlik.
                CloudMon bu altı katmanı tek bir çalıştırılabilir pipeline'da birleştirir:<br><br>
                <code>Veri Üretimi → Özellik Mühendisliği → Kapasite Tahmini →
                Sağlık Sınıflandırması → SHAP → Öneri Motoru → LLM</code><br><br>
                Her katmanın çıktısı bir sonraki katmanın girdisini besler. Modüler yapı
                sayesinde gerçek sisteme entegrasyonda yalnızca veri üretim katmanı değiştirilir.""",
            },
            {
                "no": "2",
                "title": "Log + Metrik Geçici Hizalaması",
                "body": """Çoğu çalışma ya yalnızca metrikleri ya da yalnızca logları kullanır.
                CloudMon ikisini <b>zaman damgası üzerinden tek özellik matrisinde birleştirir</b>.<br><br>
                Bu birleştirmenin etkisi SHAP analizinde net biçimde görülür: sağlık riski
                modelinde en yüksek katkı sağlayan özellikler arasında hem <code>error_rate_percent</code>
                (metrik) hem de <code>timeout_count_rmean_6h</code> (log türevi) birlikte yer almaktadır.
                Metrik eşiği henüz aşılmadan log sinyali yükselmiş olabilir — bu fusion'ın
                proaktif izleme açısından değerini kanıtlar.""",
            },
            {
                "no": "3",
                "title": "SHAP Çıktısının Aksiyona Dönüştürülmesi",
                "body": """Literatürdeki XAI çalışmaları genellikle açıklamada durur:
                <i>"Bu özellik modelin tahminini +0.31 artırdı."</i><br><br>
                CloudMon açıklamayı aksiyona bağlar:<br>
                <i>"Bu uyarı özellikle <code>timeout_count_rmean_6h</code> yüksekliğinden kaynaklanıyor.
                → Nova API timeout eşiğini kontrol edin
                → DB bağlantı havuzunu genişletin
                → Son 6 saatin Cinder loglarını inceleyin"</i><br><br>
                Bu bağlantı, XAIOps (Explainable AI for Operations) alanında mevcut literatürde
                nadiren görülen ancak sahada kritik öneme sahip bir köprüdür.""",
            },
            {
                "no": "4",
                "title": "Alan Bilgisinin Modele Entegrasyonu",
                "body": """Genel bulut izleme çalışmaları <b>domain-agnostic</b> tasarlanır:
                organizasyonel örüntüler veya planlı yük artışları modele dahil edilmez.<br><br>
                CloudMon OpenStack'e özgü servis hiyerarşisini (Nova, Keystone, Neutron, Cinder)
                ve kaynak kota yapısını hem özellik mühendisliğine hem öneri motoruna
                hem de LLM bağlamına doğrudan entegre eder.<br><br>
                <i>"Nova scheduler hatalarında artış tespit edildi — placement API performansını
                kontrol edin, compute node kaynaklarını dengeleyin."</i><br><br>
                Literatürde bu düzeyde OpenStack-aware tasarım oldukça nadirdir.""",
            },
            {
                "no": "5",
                "title": "Çift LLM Karşılaştırma Çerçevesi",
                "body": """Mevcut çalışmalar tek bir LLM ile değerlendirme yapar (Ahmed et al.: yalnızca GPT-4,
                Guo et al.: yalnızca özel model). CloudMon aynı zenginleştirilmiş bağlamı
                <b>Claude Sonnet (Anthropic)</b> ve <b>GPT-4o (OpenAI)</b>'ya eş zamanlı gönderir
                ve yanıtları yan yana karşılaştırır.<br><br>
                Bu çerçeve hem model davranış farklılıklarını hem de prompt mühendisliği kalitesini
                değerlendirmeye olanak tanır. Operasyonel bir sistemde hangi LLM'in daha
                tutarlı sonuçlar ürettiğini ampirik olarak ölçmek için doğrudan kullanılabilir.""",
            },
            {
                "no": "6",
                "title": "Modüler ve Genişletilebilir Pipeline Mimarisi",
                "body": """CloudMon'un veri katmanı ile analiz katmanı birbirinden tamamen bağımsızdır.
                data/raw/ dizinine doğru formatta veri eklendiğinde tüm pipeline otomatik çalışır:<br><br>
                • <b>Veri bağımsızlığı:</b> Prometheus, OpenSearch veya başka bir kaynak — format uyumu yeterli<br>
                • <b>Tekrarlanabilirlik:</b> SEED=42, deterministik model eğitimi<br>
                • <b>Yapılandırılabilirlik:</b> config.py üzerinden tarih aralığı, eşik değerleri, tahmin ufukları<br>
                • <b>Ölçeklenebilirlik:</b> Yeni kaynak tipi veya servis eklemek için yalnızca feature_engineering.py güncellenir<br><br>
                Bu tasarım farklı OpenStack kurulumlarına minimum değişiklikle entegre olmayı sağlar.""",
            },
        ]

        for c in CONTRIBS:
            st.markdown(f"""
            <div class="contrib-card">
              <div class="contrib-title">🏆 Katkı {c['no']} — {c['title']}</div>
              <div class="contrib-body">{c['body']}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("Açık Araştırma Soruları")
        rqs = [
            ("Kavram kayması (Concept Drift)",
             "Staj dönemlerine ve donanım alımlarına insan müdahalesi olmadan uyum sağlayan "
             "sürekli öğrenme mekanizmaları nasıl tasarlanır?"),
            ("Çok kiracılı ortamlar (Multi-Tenancy)",
             "Tek bir model birden fazla OpenStack tenant için genellenebilir mi, "
             "yoksa her tenant kendi modelini mi gerektiriyor?"),
            ("LLM güvenilirliği",
             "Aynı bağlama farklı LLM'lerin farklı öneriler üretmesi operasyonel "
             "güvenilirlik açısından nasıl ele alınmalı? Hangi yanıt seçilmeli?"),
            ("Gerçek zamanlı SHAP",
             "TreeExplainer'ı saatlik cron job ile çalıştırmak hesaplama açısından "
             "sürdürülebilir mi? FastSHAP gibi yaklaşık yöntemler bu sorunu çözer mi?"),
            ("Öneri kalite ölçütü",
             "Üretilen önerilerin kalitesini nicel olarak ölçmek için hangi metrikler "
             "kullanılabilir? İnsan değerlendirmesi ile otomatik metrik arasındaki korelasyon?"),
        ]
        for i, (rq_title, rq_body) in enumerate(rqs, 1):
            st.markdown(f"""
            <div class="rq-card">
              <b>❓ SA{i} — {rq_title}</b><br>
              {rq_body}
            </div>""", unsafe_allow_html=True)


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
            c1.metric("vCPU",   f"{last['vcpu_usage_percent']:.1f}%",
                      f"{last['vcpu_usage_percent']-prev['vcpu_usage_percent']:+.1f}%")
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
        st.subheader("Veri Kaynakları")

        st.markdown("""
        <div class="box">
        Sistem verileri <b>data/raw/</b> dizininden okunmaktadır.
        Metrik verisi Prometheus/Thanos'tan, log verisi OpenSearch'ten saatlik agreglenerek bu formata dönüştürülmüştür.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📡 Metrik Verileri")
            st.markdown("""
            <div class="box">
            <b>Kaynak:</b> Prometheus / Thanos<br><br>
            <code>GET /api/v1/query_range</code><br>
            Saatlik resolüsyon, tüm compute node'ları agregasyonu.<br><br>
            <b>Kapsam:</b> vCPU, Bellek, Disk kullanım yüzdeleri,
            API gecikmesi, hata oranı, ağ trafiği, VM sayısı.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Metrik Sütunları:**")
            st.dataframe(pd.DataFrame([
                {"Sütun": "vcpu_usage_percent",   "Birim": "%",     "Açıklama": "vCPU tahsis oranı"},
                {"Sütun": "memory_usage_percent", "Birim": "%",     "Açıklama": "Bellek kullanım oranı"},
                {"Sütun": "disk_usage_percent",   "Birim": "%",     "Açıklama": "Disk kullanım oranı"},
                {"Sütun": "api_requests_per_hour","Birim": "istek/s","Açıklama": "Saatlik API istek sayısı"},
                {"Sütun": "api_latency_ms",       "Birim": "ms",    "Açıklama": "API yanıt süresi"},
                {"Sütun": "error_rate_percent",   "Birim": "%",     "Açıklama": "Hata oranı"},
                {"Sütun": "network_in_mbps",      "Birim": "Mbps",  "Açıklama": "Gelen ağ trafiği"},
                {"Sütun": "network_out_mbps",     "Birim": "Mbps",  "Açıklama": "Giden ağ trafiği"},
                {"Sütun": "vm_count",             "Birim": "adet",  "Açıklama": "Aktif VM sayısı"},
            ]), width='stretch', hide_index=True)

        with col2:
            st.markdown("#### 📋 Log Verileri")
            st.markdown("""
            <div class="box">
            <b>Kaynak:</b> OpenSearch<br><br>
            <code>POST /openstack-logs-*/_search</code><br>
            Nova, Keystone, Neutron, Cinder log dosyaları
            Filebeat ile OpenSearch'e iletilmekte, saatlik pencerede agreglenmektedir.<br><br>
            <b>İki dosya:</b><br>
            1. <b>log_features.csv</b> — ML için saatlik log sayımları<br>
            2. <b>raw_logs.csv</b> — Ham log satırı örnekleri
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Log Özellik Sütunları:**")
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

    # ─────────────────────────────────────────────────────────────────────────
    with t3:
        st.subheader("Veri Boyutları ve Dosya Yapısı")

        if results_exist():
            raw_dir  = os.path.join(DATA_DIR, "raw")
            proc_dir = os.path.join(DATA_DIR, "processed")

            # Dosya tablosu
            files = [
                ("data/raw/metrics.csv",     "Metrik verileri (vCPU / Bellek / Disk)",  raw_dir,  "metrics.csv"),
                ("data/raw/log_features.csv","Saatlik log özellik agregasyonu",          raw_dir,  "log_features.csv"),
                ("data/raw/raw_logs.csv",    "Ham log satırı örnekleri",                raw_dir,  "raw_logs.csv"),
                ("data/processed/features.csv",    "Türetilmiş özellik matrisi",              proc_dir, "features.csv"),
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
        st.subheader("Zaman Dilimi")

        try:
            from config import START_DATE, END_DATE, TRAIN_END_DATE
            st.markdown(f"""
            <div class="box">
            <b>Veri aralığı:</b> {START_DATE} → {END_DATE} (saatlik çözünürlük)<br>
            <b>Eğitim seti:</b> {START_DATE} → {TRAIN_END_DATE}<br>
            <b>Test seti:</b> {TRAIN_END_DATE} → {END_DATE}
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Config yüklenemedi: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    with t4:
        st.subheader("Pipeline Adımları — Girdi / İşlem / Çıktı")

        steps_detail = [
            {
                "no": "1️⃣",
                "ad": "Veri Yükleme",
                "girdi": "data/raw/ dizini (metrics.csv, log_features.csv, raw_logs.csv)",
                "islem": [
                    "metrics.csv okunur → saatlik vCPU/Bellek/Disk/Gecikme/Hata sütunları",
                    "log_features.csv okunur → saatlik servis log sayımları",
                    "raw_logs.csv okunur → ham log satırı örnekleri",
                ],
                "cikti": "Pandas DataFrame'lere yüklenen metrik ve log verileri",
                "sure": "<1 sn",
                "not": "Veriler data/raw/ dizininde bulunmalıdır. "
                       "Prometheus/OpenSearch entegrasyonunda bu adım API çağrısıyla değiştirilir.",
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
                "girdi": "features.csv — eğitim: Oca 2025–Mar 2026 (15 ay) / test: Nis–Haz 2026 (3 ay)",
                "islem": [
                    "3 kaynak (vCPU, Bellek, Disk) × 3 ufuk (24h, 72h, 168h) = 9 hedef değişken",
                    "Her hedef için 7 model: LR, RF, GBT, XGBoost, LightGBM, CatBoost, MLP",
                    "Toplam 63 model — çapraz doğrulama yok, zamansal bölünmüş test seti",
                    "LR ve MLP için StandardScaler; ağaç tabanlı modeller ham özellik kullanır",
                    "En düşük RMSE → 'yıldızlı' model → gelecek tahmini (kaç ay/gün kaldı)",
                ],
                "cikti": "63 .pkl model, capacity_results.json, 3 PNG grafik (karşılaştırma / tahmin / artık)",
                "sure": "~15-20 dk (7 model × 9 hedef, GBT/XGBoost/CatBoost n_estimators=200)",
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
                       "modeller bu sinyalleri mükemmel öğrenebilir — operasyonel veri kalitesi.",
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
        <b>Ne yapar:</b> data/raw/ dizinindeki mevcut veriyi okur, özellik mühendisliği uygular,
        tüm ML modellerini yeniden eğitir, SHAP analizini ve öneri motorunu çalıştırır.<br><br>
        <b>config.py'yi değiştirirseniz:</b><br>
        • THRESHOLDS'u değiştirirseniz → risk etiketleri değişir<br>
        • FORECAST_HORIZONS'u değiştirirseniz → farklı tahmin ufukları<br>
        • TRAIN_END_DATE'i değiştirirseniz → eğitim/test bölünmesi değişir
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
    Kırmızı gölgeli alanlar yüksek hata oranı veya kaynak baskısının yaşandığı dönemlerdir —
    bu dönemlerin öncesindeki buildup sinyalleri ML modelinin öğreneceği ipuçlarını barındırır.
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Metrik Zaman Serileri", "Log Analizi", "Ham Veri Önizleme"])

    with tab1:
        show_image(fig_path("metrics_overview.png"),
                   "CPU, Bellek, Disk, API Gecikmesi, Hata Oranı, VM Sayısı. "
                   "Kırmızı gölge = yüksek yük dönemi. Kesikli yatay çizgi = kritik eşik.")
        st.markdown("""<div class="box">
        <b>Grafiği nasıl okumalı?</b><br>
        • <b>Disk:</b> Monoton artış — log ve snapshot birikimi. Lineer ekstrapolasyon ile ne zaman dolacağı hesaplanır.<br>
        • <b>Bellek:</b> VM sayısı arttıkça yavaş artış trendi.<br>
        • <b>CPU:</b> Günlük ve haftalık örüntü belirgin; yoğun dönemlerde ani spike.<br>
        • <b>Hata Oranı:</b> Normalde düşük seviyelerde, yük artışlarında belirgin yükseliş gösterir.
        </div>""", unsafe_allow_html=True)

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
    from config import FORECAST_HORIZONS, TRAIN_END_DATE, END_DATE, START_DATE
    st.title("📈 Kapasite Tahmini")
    n_algo     = 7  # LR, RF, GBT, XGBoost, LightGBM, CatBoost, MLP
    test_period = f"{pd.Timestamp(TRAIN_END_DATE).strftime('%b %Y')} – {pd.Timestamp(END_DATE).strftime('%b %Y')}"
    horizon_str = " / ".join(f"{h}h" for h in FORECAST_HORIZONS)
    st.markdown(f"""<div class="box">
    <b>{n_algo} model</b> (LR, RF, GBT, XGBoost, LightGBM, CatBoost, MLP) × <b>3 kaynak</b> (vCPU, Bellek, Disk) × <b>{len(FORECAST_HORIZONS)} ufuk</b> ({horizon_str}) = <b>{n_algo * 3 * len(FORECAST_HORIZONS)} model</b>.<br>
    Eğitim: {pd.Timestamp(START_DATE).strftime('%b %Y')} – {pd.Timestamp(TRAIN_END_DATE).strftime('%b %Y')} (15 ay) · Test: {test_period} (3 ay, zamansal bölünmüş).<br>
    ★ simgesi: her kaynak–ufuk çifti için en düşük RMSE'ye sahip model. Gelecek tahmini bu modelle yapılır.
    </div>""", unsafe_allow_html=True)

    cap = load_json("capacity_results.json")
    if not cap:
        st.warning("Kapasite sonuçları bulunamadı.")
        st.stop()

    _res_tr = {
        "vcpu_usage_percent":   "vCPU (%)",
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

    outer_tab1, outer_tab2 = st.tabs(["📊 Model Karşılaştırma", "📅 Kapasite Dolum Tarihleri"])

    with outer_tab1:
        horizons_in_results = sorted(set(int(v["horizon_h"]) for v in cap.values()))
        horizon_tabs = st.tabs([f"{h} Saatlik Ufuk" for h in horizons_in_results])

        for tab, horizon in zip(horizon_tabs, horizons_in_results):
            with tab:
                horizon_key = f"{horizon}s"
                sub = df_res[df_res["Ufuk"] == horizon_key].drop(columns="Ufuk").copy()
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
                   "Gerçek (mavi) ve tahmin (kırmızı kesikli). Her satır bir kaynak.")
        show_image(fig_path("capacity_model_comparison.png"),
                   "Her kaynak için üç modelin RMSE karşılaştırması.")
        show_image(fig_path("capacity_residuals.png"),
                   "Hata dağılımı histogramı: sıfır hata (kırmızı kesikli) etrafında simetrik dağılım iyi kalibrasyonu gösterir. Turuncu noktalı çizgi ort. hatayı gösterir.")

    with outer_tab2:
        st.markdown("""<div class="box">
        Her kaynak için <b>en düşük RMSE</b>'ye sahip 24s ufuklu ML modeli seçilir ve iteratif
        ileriye tahmin yapılır. Tam 1.5 yıllık veri geçmişi kullanılır — kırpılmaz —
        böylece disk gibi yavaş büyüyen kaynaklar da doğru büyüme trendi korunur. Maks. 10 yıl ileri bakılır.<br>
        <b>Konfigüre Eşik:</b> vCPU %90 · Bellek %88 · Disk %85 &nbsp;|&nbsp;
        <b>%100 Tarihi:</b> Fiziksel kapasitenin tamamen dolacağı tahmini an
        </div>""", unsafe_allow_html=True)

        with st.spinner("ML modelleri ile eşik tarihleri hesaplanıyor…"):
            try:
                sys.path.insert(0, BASE)
                from src.preprocessing.feature_engineering import build_feature_matrix
                from src.models.capacity_predictor import predict_days_to_threshold

                _metrics_path = os.path.join(DATA_DIR, "raw", "metrics.csv")
                _log_path     = os.path.join(DATA_DIR, "raw", "log_features.csv")
                if not os.path.exists(_metrics_path) or not os.path.exists(_log_path):
                    st.warning("Veri dosyaları bulunamadı. Önce pipeline'ı çalıştırın.")
                    st.stop()

                _mdf = pd.read_csv(_metrics_path, parse_dates=["timestamp"])
                _ldf = pd.read_csv(_log_path,     parse_dates=["timestamp"])
                _fdf = build_feature_matrix(_mdf, _ldf)
                if "timestamp" in _fdf.columns:
                    _fdf = _fdf.set_index("timestamp")

                _res_cfg = [
                    ("vcpu_usage_percent",   "vCPU",   90.0, "#e53e3e"),
                    ("memory_usage_percent", "Bellek", 88.0, "#f59e0b"),
                    ("disk_usage_percent",   "Disk",   85.0, "#3b82f6"),
                ]

                def _fmt_thr(info):
                    est  = info.get("estimated_date")
                    days = info.get("days_to_threshold")
                    stb  = info.get("stable", False)
                    if stb or est is None:
                        return "Kararlı (>10 yıl)", "—"
                    d = int(days) if days else 0
                    m = round(d / 30.44, 1)
                    return est, f"~{m} ay (~{d} gün)"

                _computed = []
                for _col_key, _lbl, _cfg_thr, _clr in _res_cfg:
                    _i_cfg = predict_days_to_threshold(_fdf, cap, col=_col_key)
                    _i_100 = predict_days_to_threshold(_fdf, cap, col=_col_key, thr=100.0)
                    _cfg_date, _cfg_rem   = _fmt_thr(_i_cfg)
                    _d100_date, _d100_rem = _fmt_thr(_i_100)
                    _cur  = _i_cfg.get("current_value", "?")
                    _meth = _i_cfg.get("method", "")
                    _computed.append((_col_key, _lbl, _cfg_thr, _clr,
                                      _cur, _cfg_date, _cfg_rem, _d100_date, _d100_rem, _meth))

                # ── Tablo ────────────────────────────────────────────────────────
                _thr_rows = []
                for _, _lbl, _cfg_thr, _, _cur, _cfg_date, _cfg_rem, _d100_date, _d100_rem, _meth in _computed:
                    _thr_rows.append({
                        "Kaynak":             _lbl,
                        "Mevcut (%)":         round(float(_cur), 1) if isinstance(_cur, (int, float)) else _cur,
                        "Konfigüre Eşik (%)": int(_cfg_thr),
                        "Eşik Tarihi":        _cfg_date,
                        "Eşiğe Kalan":        _cfg_rem,
                        "%100 Tarihi":        _d100_date,
                        "%100 Kalan":         _d100_rem,
                        "Yöntem":             _meth,
                    })
                st.dataframe(pd.DataFrame(_thr_rows), width='stretch', hide_index=True)

                # ── Kartlar ───────────────────────────────────────────────────────
                st.divider()
                _card_cols = st.columns(3)
                for _cw, (_col_key, _lbl, _cfg_thr, _clr,
                           _cur, _cfg_date, _cfg_rem, _d100_date, _d100_rem, _meth) in zip(_card_cols, _computed):
                    _cur_str = f"%{round(float(_cur), 1)}" if isinstance(_cur, (int, float)) else str(_cur)
                    with _cw:
                        st.markdown(
                            f'<div style="border-left:4px solid {_clr};padding:12px 14px;'
                            f'border-radius:6px;background:#f9fafb;">'
                            f'<div style="font-weight:700;font-size:1.05rem;color:{_clr};margin-bottom:10px;">{_lbl}</div>'
                            f'<div style="font-size:.88rem;margin-bottom:6px;"><b>Mevcut:</b> {_cur_str}</div>'
                            f'<div style="font-size:.88rem;margin-bottom:2px;">'
                            f'<b>📅 Eşik (%{int(_cfg_thr)}):</b> '
                            f'<span style="color:#dc2626;font-weight:600;">{_cfg_date}</span></div>'
                            f'<div style="font-size:.8rem;color:#6b7280;margin:0 0 8px 10px;">{_cfg_rem}</div>'
                            f'<div style="font-size:.88rem;margin-bottom:2px;">'
                            f'<b>📅 Tam Dolum (%100):</b> '
                            f'<span style="color:#7c3aed;font-weight:600;">{_d100_date}</span></div>'
                            f'<div style="font-size:.8rem;color:#6b7280;margin:0 0 8px 10px;">{_d100_rem}</div>'
                            f'<div style="font-size:.75rem;color:#9ca3af;">Yöntem: {_meth}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            except Exception as e:
                st.error(f"Hesaplama hatası: {e}")
                import traceback
                with st.expander("Hata detayı"):
                    st.code(traceback.format_exc(), language="python")


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
    show_image(fig_path("health_calibration.png"),
               "Reliability Diagram: iyi kalibre modelde tahmin edilen olasılık gerçek oranla örtüşür (diyagonal). Kesikli çizgi = mükemmel kalibrasyon.")
    st.markdown("""<div class="box">
    İyi bir model risk olasılığını incident başlamadan <b>birkaç saat öncesinde</b> yükseltir —
    bu proaktif izlemenin temel göstergesidir. Olasılık 0.5 eşiğini aşınca "riskli" kararı üretilir.<br>
    <b>Reliability Diagram</b>: Modelin "0.8 olasılık" dediğinde gerçekten ~%80 oranında riskli mi?
    Eğri diyagonale yakınsa model iyi kalibre demektir.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SHAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page.startswith("🔍"):
    st.title("🔍 SHAP Açıklanabilirlik Analizi")
    st.markdown("""<div class="box">
    <b>SHAP (SHapley Additive exPlanations)</b>: Oyun teorisinden türetilen Shapley değerleriyle
    her özelliğin modelin tahminini ortalama beklentiden ne yönde ve ne kadar sapdırdığı hesaplanır.
    Ağaç tabanlı modeller için <code>TreeExplainer</code>, doğrusal modeller için <code>LinearExplainer</code> kullanılır.
    Test setinden 500 örneklem üzerinde çalışır.
    </div>""", unsafe_allow_html=True)

    from config import FORECAST_HORIZONS
    tab_cpu, tab_mem, tab_disk, tab_sag = st.tabs([
        "CPU Tahmini", "Bellek Tahmini", "Disk Tahmini", "Sağlık Riski",
    ])
    _shap_horizons = FORECAST_HORIZONS

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
        h = st.radio("Ufuk", _shap_horizons, horizontal=True, key="sh_cpu")
        _shap_sec("vcpu_usage_percent", h, f"vCPU — {h}s ufku",
                  "vCPU tahmini için <b>kısa vadeli hareketli ortalama</b> (vcpu_rmean_3h, vcpu_lag_1h) "
                  "en yüksek SHAP değerini alır: vCPU kullanımı güçlü süreklilik gösterir. "
                  "Incident dönemlerinde <b>log hata sayımları ve gecikme metrikleri</b> devreye girer.")

    with tab_mem:
        h2 = st.radio("Ufuk", _shap_horizons, horizontal=True, key="sh_mem")
        _shap_sec("memory_usage_percent", h2, f"Bellek — {h2}s ufku",
                  "Bellek yavaş artış trendine sahip olduğundan <b>week_of_year</b> ve "
                  "<b>168h hareketli ortalama</b> önemli rol oynar. "
                  "VM yaratma patlamalarında <b>vm_count lag</b> özellikleri de belirleyici.")

    with tab_disk:
        h3 = st.radio("Ufuk", _shap_horizons, horizontal=True, key="sh_disk")
        _shap_sec("disk_usage_percent", h3, f"Disk — {h3}s ufku",
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

    rec_data = load_json("recommendations.json")
    if not rec_data:
        st.warning("Öneri dosyası bulunamadı.")
        st.stop()

    # Yeni format: {"recommendations": [...], "capacity_summary": {...}, "error_summary": {...}}
    recs     = rec_data.get("recommendations", rec_data) if isinstance(rec_data, dict) else rec_data
    cap_sum  = rec_data.get("capacity_summary", {}) if isinstance(rec_data, dict) else {}
    err_sum  = rec_data.get("error_summary",    {}) if isinstance(rec_data, dict) else {}

    n_c = sum(1 for r in recs if r.get("severity") == "critical")
    n_w = sum(1 for r in recs if r.get("severity") == "warning")
    n_i = sum(1 for r in recs if r.get("severity") == "info")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Toplam", len(recs))
    c2.metric("🔴 Kritik", n_c)
    c3.metric("🟡 Uyarı",  n_w)
    c4.metric("🔵 Bilgi",  n_i)
    st.divider()

    # ── Kapasite bitiş tarihleri ───────────────────────────────────────────────
    if cap_sum:
        st.subheader("Kapasite Dolum Tahmini")
        _cap_labels = {
            "vcpu_usage_percent":   ("vCPU", "#e53e3e"),
            "memory_usage_percent": ("Bellek", "#f59e0b"),
            "disk_usage_percent":   ("Disk", "#3b82f6"),
        }
        cc1, cc2, cc3 = st.columns(3)
        for col_w, (col_key, (lbl, clr)) in zip([cc1, cc2, cc3], _cap_labels.items()):
            info   = cap_sum.get(col_key, {})
            cur    = info.get("current_value", "?")
            est    = info.get("estimated_date")
            days   = info.get("days_to_threshold")
            months = info.get("months_to_threshold")
            stb    = info.get("stable", False)
            trend  = info.get("daily_trend", 0)
            meth   = info.get("method", "")
            with col_w:
                if stb or est is None:
                    col_w.metric(f"📅 {lbl} Bitiş", "Kararlı", f"+{trend:.3f}%/gün")
                else:
                    time_str = f"~{months:.1f} ay (~{int(days)} gün)" if months else f"~{int(days)} gün"
                    col_w.metric(f"📅 {lbl} Bitiş", est, f"{time_str} · %{cur:.1f} şu an")
                if meth:
                    st.caption(f"Yöntem: {meth}")
        st.divider()

    # ── Haftalık hata özeti ────────────────────────────────────────────────────
    if err_sum:
        st.subheader("7 Günlük Hata Karşılaştırması")
        err_rows = []
        for label, data in err_sum.items():
            err_rows.append({
                "Hata Tipi":    label,
                "Son 7 Gün":   f"{data['last_7d']:,}",
                "Önceki 7 Gün": f"{data['prev_7d']:,}",
                "Değişim":      f"{data['change_pct']:+.0f}%",
                "Durum":        "⚠️ YÜKSEK" if data.get("high") else "Normal",
            })
        st.dataframe(pd.DataFrame(err_rows), width='stretch', hide_index=True)
        st.divider()

    _sm = {
        "critical": ("🔴","KRİTİK","rec-critical","badge-c"),
        "warning":  ("🟡","UYARI", "rec-warning", "badge-w"),
        "info":     ("🔵","BİLGİ", "rec-info",    "badge-i"),
    }
    _cat = {
        "KAPASİTE": "Kapasite",
        "SAĞLIK":   "Servis Sağlığı",
        "TREND":    "Metrik Trendi",
        "STAJ":     "Staj Dönemi",
    }

    for i, rec in enumerate(recs, 1):
        sev = rec.get("severity","info")
        icon, label, css, badge = _sm.get(sev, ("🔵","BİLGİ","rec-info","badge-i"))
        acts = "".join(f"<li>{a}</li>" for a in rec.get("actions",[]))
        cat_label = _cat.get(rec.get("category",""), rec.get("category",""))
        est_line  = ""
        if rec.get("estimated_date"):
            _d = rec.get("days_remaining")
            _m = rec.get("months_remaining")
            _t = f" (~{_m:.1f} ay / {_d} gün)" if (_m is not None and _d is not None) else (f" (~{_d} gün)" if _d is not None else "")
            est_line = f'<span style="color:#e53e3e;font-size:.85rem;"> 📅 Tahmini eşik: <b>{rec["estimated_date"]}</b>{_t}</span>'
        st.markdown(f"""
        <div class="{css}">
          <div style="margin-bottom:6px;">
            {icon} <span class="{badge}">{label}</span>
            <span style="font-size:.82rem;color:#6b7280;"> #{i} · {cat_label}</span>
            {est_line}
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


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM ÖNERİ KARŞILAŞTIRMASI
# ═══════════════════════════════════════════════════════════════════════════════
elif page.startswith("🤖"):
    st.title("🤖 LLM Öneri Karşılaştırması")
    st.markdown("""<div class="box">
    Aynı sistem bağlamı (metrik özeti + SHAP özellikleri + hata örüntüleri) hem
    <b>Claude Sonnet (Anthropic)</b> hem <b>GPT-4o (OpenAI)</b> modeline gönderilir.
    Yanıtlar yan yana karşılaştırılır. API anahtarlarınızı girin ve <b>Analiz Et</b>'e tıklayın.
    </div>""", unsafe_allow_html=True)

    if not results_exist():
        st.warning("Önce pipeline'ı çalıştırın (sidebar → 🔄 Pipeline'ı Yeniden Çalıştır).")
        st.stop()

    # ── API Anahtarları ───────────────────────────────────────────────────────
    with st.expander("🔑 API Anahtarları", expanded=True):
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            anthropic_key = st.text_input(
                "Anthropic API Key",
                value=os.environ.get("ANTHROPIC_API_KEY", ""),
                type="password",
                placeholder="sk-ant-...",
            )
        with col_k2:
            openai_key = st.text_input(
                "OpenAI API Key",
                value=os.environ.get("OPENAI_API_KEY", ""),
                type="password",
                placeholder="sk-...",
            )
        if not anthropic_key and not openai_key:
            st.info("En az bir API anahtarı girin.")

    # ── Analiz Et butonu ──────────────────────────────────────────────────────
    run_btn = st.button(
        "🚀 Analiz Et",
        type="primary",
        disabled=(not anthropic_key and not openai_key),
        use_container_width=True,
    )

    if run_btn:
        sys.path.insert(0, BASE)
        from src.recommendations.llm_advisor import LLMAdvisor, PROVIDERS
        from src.preprocessing.feature_engineering import build_feature_matrix

        with st.spinner("Veriler hazırlanıyor…"):
            try:
                metrics_df = pd.read_csv(os.path.join(DATA_DIR, "raw", "metrics.csv"),
                                         parse_dates=["timestamp"])
                log_df     = pd.read_csv(os.path.join(DATA_DIR, "raw", "log_features.csv"),
                                         parse_dates=["timestamp"])
                feature_df = build_feature_matrix(metrics_df, log_df)

                cap_res    = load_json("capacity_results.json")
                health_res = load_json("health_results.json")

                # SHAP CSV'lerinden top özellikleri yükle
                shap_health = {}
                shap_h_path = os.path.join(REP_DIR, "shap_health.csv")
                if os.path.exists(shap_h_path):
                    df_sh = pd.read_csv(shap_h_path)
                    shap_health = {"top_features": df_sh.head(10).to_dict("records")}

                shap_cap = {}
                for fname in os.listdir(REP_DIR):
                    if fname.startswith("shap_capacity_cpu") and fname.endswith(".csv"):
                        df_sc = pd.read_csv(os.path.join(REP_DIR, fname))
                        shap_cap["cpu_24h"] = {"top_features": df_sc.head(5).to_dict("records")}
                        break

                advisor = LLMAdvisor(
                    anthropic_key=anthropic_key,
                    openai_key=openai_key,
                )
            except Exception as e:
                st.error(f"Veri yükleme hatası: {e}")
                st.stop()

        # ── Önerileri yükle (LLM prompt'a verilecek) ─────────────────────────
        rec_json   = load_json("recommendations.json")
        saved_recs = rec_json.get("recommendations", []) if isinstance(rec_json, dict) else []

        # ── LLM çağrıları ─────────────────────────────────────────────────────
        with st.spinner("LLM modelleri sorgulanıyor…"):
            result = advisor.get_recommendations(
                feature_df,
                shap_cap,
                capacity_results=cap_res,
                recommendations=saved_recs,
            )

        responses = result.get("responses", {})

        # ── Gönderilen prompt ─────────────────────────────────────────────────
        with st.expander("📋 Gönderilen Prompt (her iki modele aynısı)"):
            st.code(result.get("prompt", ""), language=None)

        # ── Sistem bağlamı özeti ──────────────────────────────────────────────
        ctx  = result.get("context", {})
        ms   = ctx.get("metric_summary", {})
        cdates = ctx.get("capacity_dates", {})
        if ms:
            st.subheader("Sistem Bağlamı")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("vCPU (ort)",    f"%{ms.get('vcpu_mean', 0):.1f}")
            c2.metric("Bellek (ort)",  f"%{ms.get('mem_mean', 0):.1f}")
            c3.metric("Disk (ort)",    f"%{ms.get('disk_mean', 0):.1f}")
            c4.metric("Latency (ort)", f"{ms.get('latency_mean', 0):.0f} ms")
            c5.metric("Hata (ort)",    f"%{ms.get('error_mean', 0):.2f}")

        if cdates:
            st.subheader("Kapasite Bitiş Tahminleri")
            dc1, dc2, dc3 = st.columns(3)
            _dl = {"vcpu_usage_percent": "vCPU", "memory_usage_percent": "Bellek", "disk_usage_percent": "Disk"}
            for col_w, (col_key, lbl) in zip([dc1, dc2, dc3], _dl.items()):
                info   = cdates.get(col_key, {})
                est    = info.get("estimated_date")
                days   = info.get("days_to_threshold")
                months = info.get("months_to_threshold")
                stb    = info.get("stable", False)
                if stb or est is None:
                    col_w.metric(f"📅 {lbl} Bitiş", "Kararlı")
                else:
                    time_str = f"~{months:.1f} ay (~{int(days)} gün)" if months else f"~{int(days)} gün"
                    col_w.metric(f"📅 {lbl} Bitiş", est, time_str)

        st.divider()

        # ── Yanıtlar yan yana ────────────────────────────────────────────────
        st.subheader("Model Yanıtları")
        provider_keys = ["claude", "openai"]
        cols = st.columns(len(provider_keys))

        for col, pkey in zip(cols, provider_keys):
            info   = PROVIDERS.get(pkey, {})
            resp   = responses.get(pkey, {})
            status = resp.get("status", "skipped")
            label  = info.get("label", pkey)
            color  = info.get("color", "#666")

            with col:
                st.markdown(
                    f'<div style="border-top:4px solid {color};padding-top:8px;'
                    f'margin-bottom:12px;font-weight:700;font-size:1.05rem;">'
                    f'{label}</div>',
                    unsafe_allow_html=True,
                )

                if status == "ok":
                    st.markdown(resp.get("text", ""))
                elif status == "skipped":
                    st.info(f"API anahtarı girilmedi — {label} atlandı.")
                else:
                    st.error(f"Hata: {resp.get('error', 'bilinmeyen hata')}")

        # ── JSON indirme ──────────────────────────────────────────────────────
        st.divider()
        st.download_button(
            "📥 Yanıtları JSON olarak indir",
            data=json.dumps(
                {k: {"status": v["status"], "text": v.get("text", ""), "error": v.get("error", "")}
                 for k, v in responses.items()},
                ensure_ascii=False, indent=2
            ),
            file_name="llm_recommendations.json",
            mime="application/json",
        )
