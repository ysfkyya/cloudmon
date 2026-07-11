"""
Servis sağlığı risk sınıflandırıcısı.

Üç model karşılaştırılır:
  • Lojistik Regresyon     (LogisticRegression)
  • Rastgele Orman         (RandomForestClassifier)
  • Gradient Boosting      (GradientBoostingClassifier)

Hedef: Sonraki RISK_HORIZON saatte sistemin riskli duruma girip girmeyeceğini tahmin et.
Sınıf dengesizliği class_weight='balanced' ile ele alınır.
"""

import os
import sys
import json
import joblib
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble     import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics      import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import RISK_HORIZON, RESULTS_DIR, DATA_DIR, TRAIN_END_DATE, END_DATE

_MODEL_DIR = os.path.join(DATA_DIR, "processed", "models")
os.makedirs(_MODEL_DIR, exist_ok=True)

_MODELS = {
    "Logistic Regression":  LogisticRegression(
                                max_iter=2000, class_weight="balanced",
                                C=0.5, solver="lbfgs", random_state=42),
    "Random Forest":        RandomForestClassifier(
                                n_estimators=200, max_depth=12,
                                min_samples_leaf=4, class_weight="balanced",
                                n_jobs=-1, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(
                                n_estimators=200, max_depth=5,
                                learning_rate=0.08, subsample=0.85,
                                random_state=42),
}


def train_health_models(dataset: dict) -> dict:
    """
    dataset : prepare_health_dataset() çıktısı
    Döndürür: model sonuçlarını içeren sözlük
    """
    X_train, y_train = dataset["X_train"], dataset["y_train"]
    X_test,  y_test  = dataset["X_test"],  dataset["y_test"]
    fnames            = dataset["feature_names"]
    ts_test           = dataset["timestamps_test"]

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_train)
    X_te_s  = scaler.transform(X_test)

    # Sınıf dağılımı bilgisi
    pos_rate = y_train.mean()
    print(f"    Eğitim seti risk oranı: {pos_rate:.1%}  "
          f"({y_train.sum()} riskli / {len(y_train)} toplam)")

    results = {
        "feature_names":   fnames,
        "timestamps_test": ts_test,
        "y_test":          y_test,
        "pos_rate_train":  round(float(pos_rate), 4),
        "models":          {},
    }

    for mname, model_proto in _MODELS.items():
        model = copy.deepcopy(model_proto)

        if "Logistic" in mname:
            model.fit(X_tr_s, y_train)
            y_pred  = model.predict(X_te_s)
            y_proba = model.predict_proba(X_te_s)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred, zero_division=0)
        rec   = recall_score(y_test, y_pred, zero_division=0)
        f1    = f1_score(y_test, y_pred, zero_division=0)
        auc   = roc_auc_score(y_test, y_proba)
        cm    = confusion_matrix(y_test, y_pred)

        model_path = os.path.join(
            _MODEL_DIR,
            f"health_{mname.lower().replace(' ', '_')}.pkl",
        )
        joblib.dump(
            {"model": model, "scaler": scaler, "feature_names": fnames},
            model_path,
        )

        results["models"][mname] = {
            "Accuracy":  round(acc,  4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
            "F1":        round(f1,   4),
            "AUC-ROC":   round(auc,  4),
            "confusion_matrix": cm.tolist(),
            "y_pred":    y_pred,
            "y_proba":   y_proba,
        }

        print(
            f"    [{mname}]  Acc={acc:.3f}  Prec={prec:.3f}  "
            f"Rec={rec:.3f}  F1={f1:.3f}  AUC={auc:.3f}"
        )

    _save_health_report(results)
    _plot_roc_curves(results)
    _plot_confusion_matrices(results)
    _plot_risk_timeline(results)
    _plot_calibration_curve(results)

    return results


def _save_health_report(results: dict) -> None:
    report = {
        "risk_horizon_hours": RISK_HORIZON,
        "pos_rate_train":     results["pos_rate_train"],
        "models": {
            mn: {k: v for k, v in mv.items() if k not in ("y_pred", "y_proba")}
            for mn, mv in results["models"].items()
        },
    }
    out = os.path.join(RESULTS_DIR, "reports", "health_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  → {out}")


def _plot_roc_curves(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = ["steelblue", "darkorange", "green"]
    y_test  = results["y_test"]

    for (mname, mv), color in zip(results["models"].items(), colors):
        fpr, tpr, _ = roc_curve(y_test, mv["y_proba"])
        ax.plot(fpr, tpr, label=f"{mname}  (AUC={mv['AUC-ROC']:.3f})", color=color, linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("Yanlış Pozitif Oranı")
    ax.set_ylabel("Gerçek Pozitif Oranı")
    ax.set_title(f"ROC Eğrileri — Sağlık Riski Sınıflandırması ({RISK_HORIZON}s ufuk)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", "health_roc_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def _plot_confusion_matrices(results: dict) -> None:
    n = len(results["models"])
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (mname, mv) in zip(axes, results["models"].items()):
        cm = np.array(mv["confusion_matrix"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Güvenli", "Riskli"],
            yticklabels=["Güvenli", "Riskli"],
        )
        ax.set_title(f"{mname}\nF1={mv['F1']:.3f}", fontsize=10)
        ax.set_ylabel("Gerçek")
        ax.set_xlabel("Tahmin")

    plt.suptitle("Karmaşıklık Matrisleri — Test Dönemi", fontsize=11)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", "health_confusion_matrices.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def _plot_calibration_curve(results: dict) -> None:
    """Reliability Diagram: tahmin edilen olasılık vs gerçek pozitif oranı."""
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = ["steelblue", "darkorange", "green"]
    y_test  = results["y_test"]

    for (mname, mv), color in zip(results["models"].items(), colors):
        try:
            prob_true, prob_pred = calibration_curve(
                y_test, mv["y_proba"], n_bins=10, strategy="uniform"
            )
            ax.plot(prob_pred, prob_true, marker="o", linewidth=1.5, color=color,
                    label=f"{mname}  (AUC={mv['AUC-ROC']:.3f})")
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Mükemmel kalibrasyon")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlabel("Tahmin Edilen Olasılık")
    ax.set_ylabel("Gerçek Pozitif Oranı")
    ax.set_title(f"Reliability Diagram — Sağlık Riski Olasılık Kalibrasyonu ({RISK_HORIZON}s ufuk)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", "health_calibration.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def _plot_risk_timeline(results: dict) -> None:
    """Zaman ekseninde gerçek risk etiketleri vs. en iyi modelin tahmini."""
    ts     = results["timestamps_test"]
    y_test = results["y_test"]

    best_m  = max(results["models"].items(), key=lambda x: x[1]["F1"])[0]
    y_proba = results["models"][best_m]["y_proba"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    test_period = f"{pd.Timestamp(TRAIN_END_DATE).strftime('%b %Y')} – {pd.Timestamp(END_DATE).strftime('%b %Y')}"
    fig.suptitle(f"Risk Zaman Çizelgesi — {best_m} (Test: {test_period})", fontsize=11)

    ax1.fill_between(ts, y_test, alpha=0.5, color="crimson", label="Gerçek Risk")
    ax1.set_ylabel("Risk Etiketi (0/1)")
    ax1.set_ylim(-0.1, 1.3)
    ax1.legend(fontsize=9)

    ax2.plot(ts, y_proba, color="steelblue", linewidth=0.7, label="Risk Olasılığı")
    ax2.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Karar eşiği (0.5)")
    ax2.set_ylabel("Risk Olasılığı")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", "health_risk_timeline.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")
