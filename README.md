# CloudMon — Bulut Altyapısı Proaktif İzleme ve Karar Destek Sistemi

OpenStack ortamları için uçtan uca ML pipeline'ı. Sentetik metrik + log verisi üretir, üç farklı yöntemle kapasite ve sağlık riski tahmin eder, SHAP ile açıklar ve sistem yöneticisine önceliklendirilmiş operasyonel öneriler sunar.

---

## Mimari Genel Bakış

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (Pipeline)                       │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────────────┐
│  1. Veri        │    │  metrics.csv          │ 7.296 saatlik nokta
│  Üretimi        │───▶│  log_features.csv     │ (10 ay, saatlik)
│                 │    │  raw_logs.csv         │ 5.000 log kaydı
└────────┬────────┘    └──────────────────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────────────┐
│  2. Özellik     │    │  features.csv         │ ~170 özellik sütunu
│  Mühendisliği   │───▶│  (rolling/lag/delta/  │ (zaman, metrik, log)
│                 │    │   zaman/log)          │
└────────┬────────┘    └──────────────────────┘
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
┌─────────────────┐              ┌──────────────────────┐
│  3. Kapasite    │              │  4. Sağlık Riski      │
│  Tahmini        │              │  Sınıflandırması      │
│  (Regresyon)    │              │  (Binary Classifier)  │
│                 │              │                        │
│  3 Model        │              │  3 Model               │
│  × 3 Kaynak     │              │  Hedef: sonraki 4      │
│  × 2 Ufuk       │              │  saatte risk var mı?   │
│  = 18 model     │              │                        │
└────────┬────────┘              └──────────┬────────────┘
         │                                  │
         └──────────────┬───────────────────┘
                        ▼
             ┌──────────────────────┐
             │  5. SHAP             │
             │  Açıklanabilirlik    │
             │  (TreeExplainer)     │
             └──────────┬───────────┘
                        ▼
             ┌──────────────────────┐
             │  6. Öneri Motoru     │
             │  (Kapasite / Sağlık  │
             │   / Trend tabanlı)   │
             └──────────────────────┘
```

---

## Pipeline Adımları

### Adım 1 — Sentetik Veri Üretimi

**Kaynak:** `src/data_generation/`

Gerçek bir OpenStack cluster'ından gelen 10 aylık (Eyl 2025 – Haz 2026) saatlik veriyi simüle eder. Üç çıktı üretilir:

| Dosya | Açıklama | Boyut |
|---|---|---|
| `data/raw/metrics.csv` | Cluster düzeyi metrikler | ~7.296 satır |
| `data/raw/log_features.csv` | Saatlik log sayımları (ML için) | ~7.296 satır |
| `data/raw/raw_logs.csv` | Gerçekçi ham log örnekleri | 5.000 kayıt |

**Metrik sütunları:** `cpu_usage_percent`, `memory_usage_percent`, `disk_usage_percent`, `api_latency_ms`, `error_rate_percent`, `network_in_mbps`, `network_out_mbps`, `api_requests_per_hour`, `vm_count`

**Gerçekçilik mekanizmaları:**
- Günlük yük örüntüsü: 04:00 dip → 13:00 zirve (sinüsoidal)
- Haftalık örüntü: hafta sonları %35 daha az yük
- Uzun vadeli trendler: disk kullanımı 10 ayda +32 puan, VM sayısı +78 artış
- **10 tanımlanmış incident:** Her biri buildup (karesel artış) + aktif dönem (Gauss çanı) profiliyle simüle edilir

<details>
<summary>10 Incident Listesi (genişlet)</summary>

| # | Olay | Tarih | En Belirgin Etki |
|---|---|---|---|
| 1 | Bellek Baskısı — Hesaplama Grubu A | Eyl 2025 | Memory +38% |
| 2 | Disk Kapasitesi Kritik Seviyeye Yaklaşıyor | Eki 2025 | Disk +24% |
| 3 | Yüksek CPU Yükü Olayı | Kas 2025 | CPU +48% |
| 4 | Yılsonu Trafik Artışı | Ara 2025 | Latency +1100ms |
| 5 | Veritabanı Bağlantı Fırtınası | Oca 2026 | Latency +3400ms, Error +30% |
| 6 | Ağ Katmanı Kesintisi | Şub 2026 | Latency +2600ms |
| 7 | API Gecikme Bozunumu | Mar 2026 | Latency +2200ms |
| 8 | Zamanlayıcı Kaynak Yetersizliği | Nis 2026 | CPU +35% |
| 9 | Çok Kaynaklı Basamaklı Arıza | May 2026 | CPU +45%, Memory +34% |
| 10 | Keystone Kimlik Doğrulama Krizi | Haz 2026 | Error +24%, Latency +1800ms |

</details>

**Log verisi:** `log_features.csv`, metrik değerleriyle (CPU, latency, error rate) istatistiksel olarak ilişkilendirilmiş saatlik servis hata sayımlarıdır. `raw_logs.csv` ise görsel doğrulama için gerçekçi Nova/Keystone/Neutron/Cinder log satırları içerir — ML'de doğrudan kullanılmaz.

---

### Adım 2 — Özellik Mühendisliği

**Kaynak:** `src/preprocessing/feature_engineering.py`

Metrik + log verisi birleştirilerek ~170 sütunlu özellik matrisi oluşturulur:

| Kategori | Özellikler | Amaç |
|---|---|---|
| **Ham metrikler** | CPU, Bellek, Disk, Latency, Error Rate | Anlık durum |
| **Zaman** | hour_of_day, day_of_week, is_weekend, is_business_hour | Döngüsel örüntüler |
| **Rolling ortalama** | 3h / 6h / 12h / 24h / 168h | Kısa–uzun vadeli trend |
| **Rolling std** | 3h / 6h / 12h / 24h | Volatilite |
| **Lag** | 1h / 6h / 24h / 48h / 168h | Geçmiş bağlam |
| **Delta** | 1h / 6h / 24h farkları | Değişim hızı |
| **Log (rolling)** | total_errors, critical_count, timeout, db_connection, resource_exhausted — 6h ortalama, 24h toplam | Servis sağlığı sinyali |

**Train / Test bölünmesi:** Tüm modeller için ortak:
- **Train:** Eyl 2025 – Mar 2026 (7 ay)
- **Test:** Nis 2026 – Haz 2026 (3 ay)

---

### Adım 3 — Kapasite Tahmin Modelleri

**Kaynak:** `src/models/capacity_predictor.py`

Her kaynak için gelecekteki kullanım yüzdesi tahmin edilir.

**Hedef değişkenler:** `cpu_usage_percent`, `memory_usage_percent`, `disk_usage_percent`

**Tahmin ufukları:** 24 saat ve 72 saat

**Toplam:** 3 kaynak × 2 ufuk × 3 model = **18 model**

| Model | Parametreler |
|---|---|
| Linear Regression | StandardScaler + standart LR |
| Random Forest | 200 ağaç, max_depth=12, min_samples_leaf=4 |
| Gradient Boosting | 200 iter, max_depth=5, lr=0.08, subsample=0.85 |

**Değerlendirme metrikleri:** MAE, RMSE, R², MAPE. En iyi model RMSE bazlı seçilir.

**Eşiğe kaç gün kaldığı (`predict_days_to_threshold`):** Son 72 saatin trendi lineer ekstrapolasyonla uzatılarak eşiğe (CPU: 85%, Memory: 88%, Disk: 82%) kaç gün kaldığı hesaplanır. Bu değer öneri motorunu besler.

**Çıktılar:** `results/figures/capacity_forecasts.png`, `capacity_model_comparison.png`, `results/reports/capacity_results.json`, `data/processed/models/capacity_*.pkl`

---

### Adım 4 — Sağlık Riski Sınıflandırması

**Kaynak:** `src/models/health_classifier.py`

**Soru:** *"Sonraki 4 saatte sistem kritik bir eşiği aşacak mı?"*

**Etiket oluşturma:** `_create_risk_label()` — sonraki 4 saatlik pencerede şu koşullardan herhangi biri sağlanırsa etiket = 1:

```
CPU > 85%  VEYA  Memory > 88%  VEYA  Error Rate > 5%  VEYA  API Latency > 1200ms
```

**Sınıf dengesizliği:** `class_weight='balanced'` ile giderilir.

| Model | Parametreler |
|---|---|
| Logistic Regression | max_iter=2000, C=0.5, balanced |
| Random Forest | 200 ağaç, max_depth=12, balanced |
| Gradient Boosting | 200 iter, max_depth=5, lr=0.08 |

**Değerlendirme:** Accuracy, Precision, Recall, F1, AUC-ROC. En iyi model F1 bazlı seçilir.

**Çıktılar:** ROC eğrileri, confusion matrisleri, risk zaman çizelgesi grafikleri, `results/reports/health_results.json`, `data/processed/models/health_*.pkl`

---

### Adım 5 — SHAP Açıklanabilirlik Analizi

**Kaynak:** `src/explainability/shap_explainer.py`

Ağaç tabanlı modeller için `shap.TreeExplainer` kullanılır. Her kapasite modeli (kaynak × ufuk) ve sağlık sınıflandırıcısı için ayrı analiz yapılır.

**Test setinden 500 örnek** alınarak hesaplama süresi sınırlanır.

**Üretilen grafikler her model için:**
- **Bar plot:** İlk 20 özelliğin ortalama |SHAP| değeri
- **Beeswarm:** Özellik değer dağılımı ile SHAP etkisinin ilişkisi
- **Dependence plot:** En önemli 2 özellik için scatter

**Öneri motoruna katkısı:** Her kapasite önerisine en etkili SHAP özelliği eklenir (`"En etkili özellik (SHAP): 'cpu_usage_percent_rmean_24h'"`). Sağlık önerisinde ise en yüksek SHAP'lı hata tipi özelliği (timeout, db_connection, vb.) öneri aksiyonunu belirler.

**Çıktılar:** `results/figures/shap_*.png`, `results/reports/shap_capacity_*.csv`, `results/reports/shap_health.csv`

---

### Adım 6 — Operasyonel Öneri Motoru

**Kaynak:** `src/recommendations/engine.py`

Üç kanal üzerinden öneri üretilir:

#### Kapasite Önerileri
`predict_days_to_threshold()` çıktısı kullanılır:
- **Kritik:** Eşiğe ≤7 gün kaldıysa (disk: ≤10 gün)
- **Uyarı:** Eşiğe ≤14 gün kaldıysa (disk: ≤21 gün)
- Her öneriye 3 somut aksiyon ve SHAP'tan gelen en etkili özellik eklenir

#### Sağlık Riski Önerileri
Son 48 saatlik test penceresinin ortalama risk olasılığı kullanılır:
- **Kritik:** Olasılık ≥ 0.80
- **Uyarı:** Olasılık ≥ 0.65
- SHAP'tan tespit edilen en yüksek etkili hata tipine özgü aksiyon eklenir

#### Trend Önerileri
Son 1 hafta ile önceki hafta karşılaştırılır:
- Hata Oranı: +0.5% uyarı, +1.5% kritik
- API Gecikmesi: +50ms uyarı, +150ms kritik

**Çıktılar:** `results/reports/recommendations.json`, `results/reports/recommendations.txt`

---

## Proje Yapısı

```
cloudmon/
├── config.py                          # Merkezi parametreler (tarihler, eşikler, incident tanımları)
├── main.py                            # Pipeline giriş noktası
├── dashboard.py                       # (Opsiyonel) görsel dashboard
├── src/
│   ├── data_generation/
│   │   ├── metrics_generator.py       # Saatlik cluster metrikleri üretimi
│   │   └── log_generator.py          # Log özellikleri ve ham log örnekleri
│   ├── preprocessing/
│   │   └── feature_engineering.py    # Rolling/lag/delta/zaman/log özellikleri
│   ├── models/
│   │   ├── capacity_predictor.py     # Regresyon modelleri (18 model)
│   │   └── health_classifier.py      # Binary sınıflandırıcılar (3 model)
│   ├── explainability/
│   │   └── shap_explainer.py         # TreeExplainer + görselleştirme
│   └── recommendations/
│       └── engine.py                 # Öneri üretimi (kapasite/sağlık/trend)
├── data/
│   ├── raw/                          # Üretilen ham CSV'ler
│   └── processed/models/             # Eğitilmiş .pkl modeller (18 kapasite + 3 sağlık)
└── results/
    ├── figures/                      # Tüm PNG grafikler
    └── reports/                      # JSON ve TXT raporlar
```

---

## Kurulum ve Çalıştırma

```bash
pip install numpy pandas scikit-learn matplotlib seaborn shap joblib

python main.py
```

Pipeline yaklaşık 2–5 dakika sürer. Çıktılar `results/` dizinine yazılır.

---

## Kritik Eşik Değerleri

| Metrik | Eşik |
|---|---|
| CPU Kullanımı | > 85% |
| Bellek Kullanımı | > 88% |
| Disk Kullanımı | > 82% |
| API Gecikmesi | > 1.200 ms |
| Hata Oranı | > 5% |

---

## Kullanılan Teknolojiler

- **Python 3.10+**
- **scikit-learn** — Linear Regression, Random Forest, Gradient Boosting, StandardScaler
- **SHAP** — TreeExplainer
- **pandas / numpy** — Veri işleme
- **matplotlib / seaborn** — Görselleştirme
- **joblib** — Model serileştirme
