"""
SHAP tabanlı açıklanabilirlik modülü.

Kapasite tahmini ve sağlık sınıflandırması için en iyi modelin
SHAP değerlerini hesaplar ve görselleştirir:
  • Özellik önem sıralama grafiği (bar plot)
  • Beeswarm özet grafiği
  • En etkili iki özelliğin bağımlılık grafiği
  • Sayısal önem tablosu (CSV)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import RESULTS_DIR, CAPACITY_TARGETS, FORECAST_HORIZONS

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    print("  [UYARI] shap kütüphanesi bulunamadı. SHAP analizi atlanıyor.")


_TOP_N = 20   # görselleştirme için gösterilecek özellik sayısı


# ─────────────────────────────────────────────────────────────────────────────

def explain_capacity(capacity_results: dict,
                     datasets: dict) -> dict:
    """
    En iyi kapasite modelinin (en düşük RMSE) SHAP açıklamasını üretir.
    Her kaynak × ufuk çifti için ayrı analiz yapılır.
    """
    if not _SHAP_AVAILABLE:
        return {}

    summary = {}
    # Her kaynak × ufuk için en iyi modeli seç
    for horizon in FORECAST_HORIZONS:
        for col in CAPACITY_TARGETS:
            key = f"{col}_{horizon}h"
            val = capacity_results.get(key)
            if val is None:
                continue

            best_m  = min(val["models"].items(), key=lambda x: x[1]["RMSE"])[0]
            best_mv = val["models"][best_m]
            ds      = datasets[(col, horizon)]
            fnames  = ds["feature_names"]
            X_test  = ds["X_test"]

            # Ağaç tabanlı modeller için TreeExplainer
            if "Gradient" in best_m or "Forest" in best_m:
                import joblib
                from config import DATA_DIR
                model_path = os.path.join(
                    DATA_DIR, "processed", "models",
                    f"capacity_{col}_{horizon}h_{best_m.lower().replace(' ', '_')}.pkl",
                )
                pkg   = joblib.load(model_path)
                model = pkg["model"]

                explainer   = shap.TreeExplainer(model)
                # Büyük test setlerinde örnekle (max 500)
                sample_size = min(500, len(X_test))
                idx_sample  = np.random.choice(len(X_test), sample_size, replace=False)
                X_sample    = X_test[idx_sample]
                shap_values = explainer.shap_values(X_sample)

                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                imp_df = pd.DataFrame({
                    "feature":    fnames,
                    "importance": mean_abs_shap,
                }).sort_values("importance", ascending=False).reset_index(drop=True)

                imp_df.to_csv(
                    os.path.join(RESULTS_DIR, "reports",
                                 f"shap_capacity_{col}_{horizon}h.csv"),
                    index=False,
                )

                _plot_shap_bar(imp_df, f"Kapasite Tahmini SHAP — {col} ({horizon}s)",
                               f"shap_capacity_bar_{col}_{horizon}h.png")
                _plot_shap_beeswarm(shap_values, fnames,
                                    f"Kapasite Tahmini SHAP Özet — {col} ({horizon}s)",
                                    f"shap_capacity_beeswarm_{col}_{horizon}h.png",
                                    X_sample)
                _plot_shap_dependence(shap_values, fnames, imp_df,
                                      f"shap_capacity_dep_{col}_{horizon}h.png",
                                      X_sample)

                summary[key] = {
                    "best_model":    best_m,
                    "top_features":  imp_df.head(10)[["feature", "importance"]].to_dict("records"),
                }
                print(f"    SHAP → {key} ({best_m})  top özellik: {imp_df.iloc[0]['feature']}")

    return summary


def explain_health(health_results: dict, dataset: dict) -> dict:
    """
    En iyi sağlık sınıflandırıcısının SHAP açıklamasını üretir.
    """
    if not _SHAP_AVAILABLE:
        return {}

    best_m  = max(health_results["models"].items(),
                  key=lambda x: x[1]["F1"])[0]
    fnames  = dataset["feature_names"]
    X_test  = dataset["X_test"]

    if "Gradient" in best_m or "Forest" in best_m:
        import joblib
        from config import DATA_DIR
        model_path = os.path.join(
            DATA_DIR, "processed", "models",
            f"health_{best_m.lower().replace(' ', '_')}.pkl",
        )
        pkg   = joblib.load(model_path)
        model = pkg["model"]

        explainer   = shap.TreeExplainer(model)
        sample_size = min(500, len(X_test))
        idx_sample  = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample    = X_test[idx_sample]
        shap_values = explainer.shap_values(X_sample)

        # İkili sınıflandırma: class 1 SHAP değerleri
        sv = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

        mean_abs_shap = np.abs(sv).mean(axis=0)
        imp_df = pd.DataFrame({
            "feature":    fnames,
            "importance": mean_abs_shap,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        imp_df.to_csv(
            os.path.join(RESULTS_DIR, "reports", "shap_health.csv"),
            index=False,
        )

        _plot_shap_bar(imp_df, f"Sağlık Riski SHAP — {best_m}",
                       "shap_health_bar.png")
        _plot_shap_beeswarm(sv, fnames,
                            f"Sağlık Riski SHAP Özet — {best_m}",
                            "shap_health_beeswarm.png", X_sample)
        _plot_shap_dependence(sv, fnames, imp_df,
                              "shap_health_dep.png", X_sample)

        print(f"    SHAP → Sağlık ({best_m})  top özellik: {imp_df.iloc[0]['feature']}")

        return {
            "best_model":   best_m,
            "top_features": imp_df.head(10)[["feature", "importance"]].to_dict("records"),
        }

    return {}


# ── Yardımcı görselleştirme fonksiyonları ────────────────────────────────────

def _plot_shap_bar(imp_df: pd.DataFrame, title: str, fname: str) -> None:
    top = imp_df.head(_TOP_N)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"][::-1], top["importance"][::-1],
            color="steelblue", alpha=0.80)
    ax.set_xlabel("Ortalama |SHAP Değeri|")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def _plot_shap_beeswarm(shap_vals: np.ndarray, feature_names: list,
                         title: str, fname: str, X_sample: np.ndarray) -> None:
    top_idx = np.argsort(np.abs(shap_vals).mean(axis=0))[::-1][:_TOP_N]
    sv_top  = shap_vals[:, top_idx]
    fn_top  = [feature_names[i] for i in top_idx]
    X_top   = X_sample[:, top_idx]

    expl = shap.Explanation(
        values=sv_top,
        data=X_top,
        feature_names=fn_top,
    )
    # beeswarm ax parametresi desteklemiyor; figure boyutunu plt ile ayarla
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(expl, show=False, max_display=_TOP_N)
    plt.title(title, fontsize=11)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def _plot_shap_dependence(shap_vals: np.ndarray, feature_names: list,
                           imp_df: pd.DataFrame, fname: str,
                           X_sample: np.ndarray) -> None:
    """En önemli iki özellik için bağımlılık (scatter) grafiği."""
    top2 = imp_df.head(2)["feature"].tolist()
    if len(top2) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, feat in zip(axes, top2):
        f_idx = feature_names.index(feat)
        ax.scatter(
            X_sample[:, f_idx],
            shap_vals[:, f_idx],
            alpha=0.4, s=10, color="steelblue",
        )
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP Değeri")
        ax.set_title(f"Bağımlılık: {feat}")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")
