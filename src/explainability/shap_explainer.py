"""
SHAP tabanlı açıklanabilirlik modülü — Kapasite tahmini.

En iyi kapasite modelinin SHAP değerlerini hesaplar ve görselleştirir:
  • Özellik önem sıralama grafiği (bar plot)
  • Beeswarm özet grafiği
  • En etkili iki özelliğin bağımlılık grafiği
  • Sayısal önem tablosu (CSV)
"""

import os
import sys
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

_TOP_N = 20


def explain_capacity(capacity_results: dict, datasets: dict) -> dict:
    """
    En iyi kapasite modelinin SHAP açıklamasını üretir.
    Her kaynak × ufuk çifti için ayrı analiz yapılır.
    """
    if not _SHAP_AVAILABLE:
        return {}

    summary = {}
    for horizon in FORECAST_HORIZONS:
        for col in CAPACITY_TARGETS:
            key = f"{col}_{horizon}h"
            val = capacity_results.get(key)
            if val is None:
                continue

            best_m  = min(val["models"].items(), key=lambda x: x[1]["RMSE"])[0]
            ds      = datasets.get((col, horizon)) or datasets.get(key)
            if ds is None:
                continue
            fnames  = ds["feature_names"]
            X_test  = ds["X_test"]

            import joblib
            from config import DATA_DIR
            model_path = os.path.join(
                DATA_DIR, "processed", "models",
                f"capacity_{col}_{horizon}h_{best_m.lower().replace(' ', '_')}.pkl",
            )
            if not os.path.exists(model_path):
                continue
            pkg   = joblib.load(model_path)
            model = pkg["model"]

            sample_size = min(500, len(X_test))
            idx_sample  = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample    = X_test[idx_sample]

            try:
                if "Gradient" in best_m or "Forest" in best_m:
                    explainer   = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                elif "Linear" in best_m:
                    X_train    = ds["X_train"]
                    bg_size    = min(200, len(X_train))
                    bg_idx     = np.random.choice(len(X_train), bg_size, replace=False)
                    background = shap.maskers.Independent(X_train[bg_idx], max_samples=bg_size)
                    explainer   = shap.LinearExplainer(model, background)
                    shap_values = explainer.shap_values(X_sample)
                else:
                    continue
            except Exception as e:
                print(f"    [UYARI] SHAP hesaplanamadı ({key}): {e}")
                continue

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            imp_df = pd.DataFrame({
                "feature":    fnames,
                "importance": mean_abs_shap,
            }).sort_values("importance", ascending=False).reset_index(drop=True)

            imp_df.to_csv(
                os.path.join(RESULTS_DIR, "reports", f"shap_capacity_{col}_{horizon}h.csv"),
                index=False,
            )

            _plot_shap_bar(imp_df, f"Kapasite SHAP — {col} ({horizon}s)",
                           f"shap_capacity_bar_{col}_{horizon}h.png")
            _plot_shap_beeswarm(shap_values, fnames,
                                f"Kapasite SHAP Özet — {col} ({horizon}s)",
                                f"shap_capacity_beeswarm_{col}_{horizon}h.png", X_sample)
            _plot_shap_dependence(shap_values, fnames, imp_df,
                                  f"shap_capacity_dep_{col}_{horizon}h.png", X_sample)

            summary[key] = {
                "best_model":   best_m,
                "top_features": imp_df.head(10)[["feature", "importance"]].to_dict("records"),
            }
            print(f"    SHAP → {key} ({best_m})  top: {imp_df.iloc[0]['feature']}")

    return summary


def _plot_shap_bar(imp_df, title, fname):
    top = imp_df.head(_TOP_N)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="steelblue", alpha=0.80)
    ax.set_xlabel("Ortalama |SHAP Değeri|")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def _plot_shap_beeswarm(shap_vals, feature_names, title, fname, X_sample):
    top_idx = np.argsort(np.abs(shap_vals).mean(axis=0))[::-1][:_TOP_N]
    sv_top  = shap_vals[:, top_idx]
    fn_top  = [feature_names[i] for i in top_idx]
    X_top   = X_sample[:, top_idx]

    expl = shap.Explanation(values=sv_top, data=X_top, feature_names=fn_top)
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(expl, show=False, max_display=_TOP_N)
    plt.title(title, fontsize=11)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def _plot_shap_dependence(shap_vals, feature_names, imp_df, fname, X_sample):
    top2 = imp_df.head(2)["feature"].tolist()
    if len(top2) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, feat in zip(axes, top2):
        f_idx = feature_names.index(feat)
        ax.scatter(X_sample[:, f_idx], shap_vals[:, f_idx],
                   alpha=0.4, s=10, color="steelblue")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP Değeri")
        ax.set_title(f"Bağımlılık: {feat}")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")
