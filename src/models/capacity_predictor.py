"""
Kapasite tahmin modülleri.

Yedi model karşılaştırılır:
  • Doğrusal Regresyon (LinearRegression)      — ölçeklenmiş
  • Rastgele Orman    (RandomForestRegressor)
  • Gradient Boosting (GradientBoostingRegressor)
  • XGBoost           (XGBRegressor)
  • LightGBM          (LGBMRegressor)
  • CatBoost          (CatBoostRegressor)
  • MLP               (MLPRegressor)           — ölçeklenmiş

Her kaynak (vCPU / Bellek / Disk) × her zaman ufku (24h / 72h / 168h) için ayrı
modeller eğitilir. En düşük RMSE'ye sahip model gelecek tahmini için seçilir.
Sonuçlar results/reports/capacity_results.json dosyasına yazılır.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network  import MLPRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from xgboost                 import XGBRegressor
from lightgbm                import LGBMRegressor
from catboost                import CatBoostRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CAPACITY_TARGETS, FORECAST_HORIZONS, THRESHOLDS, RESULTS_DIR, DATA_DIR, TRAIN_END_DATE, END_DATE

_MODEL_DIR = os.path.join(DATA_DIR, "processed", "models")
os.makedirs(_MODEL_DIR, exist_ok=True)

# Ölçekleme gerektiren modeller: Linear Regression, MLP
_NEEDS_SCALING = {"Linear Regression", "MLP"}

_MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(
                             n_estimators=200, max_depth=12,
                             min_samples_leaf=4, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(
                             n_estimators=200, max_depth=5,
                             learning_rate=0.08, subsample=0.85, random_state=42),
    "XGBoost":           XGBRegressor(
                             n_estimators=200, max_depth=5,
                             learning_rate=0.08, subsample=0.85,
                             n_jobs=-1, random_state=42, verbosity=0),
    "LightGBM":          LGBMRegressor(
                             n_estimators=200, max_depth=5,
                             learning_rate=0.08, subsample=0.85,
                             n_jobs=-1, random_state=42, verbose=-1),
    "CatBoost":          CatBoostRegressor(
                             iterations=200, depth=5,
                             learning_rate=0.08, subsample=0.85,
                             random_state=42, verbose=0),
    "MLP":               MLPRegressor(
                             hidden_layer_sizes=(128, 64), activation="relu",
                             max_iter=500, early_stopping=True,
                             random_state=42),
}

_RESOURCE_LABELS = {
    "vcpu_usage_percent":   "vCPU Kullanımı (%)",
    "memory_usage_percent": "Bellek Kullanımı (%)",
    "disk_usage_percent":   "Disk Kullanımı (%)",
}


def _mape(y_true, y_pred, eps=1e-6):
    return 100.0 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps)))


def train_capacity_models(datasets: dict) -> dict:
    """
    datasets : prepare_capacity_dataset() çıktısı
    Döndürür : sonuçları içeren iç içe sözlük
    """
    all_results = {}

    for (col, h), ds in [(k, v) for k, v in datasets.items() if isinstance(k, tuple)]:
        X_train, y_train = ds["X_train"], ds["y_train"]
        X_test,  y_test  = ds["X_test"],  ds["y_test"]
        fnames            = ds["feature_names"]
        ts_test           = ds["timestamps_test"]

        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_train)
        X_te_s  = scaler.transform(X_test)

        key = f"{col}_{h}h"
        all_results[key] = {
            "resource":        col,
            "horizon_h":       h,
            "feature_names":   fnames,
            "timestamps_test": ts_test,
            "y_test":          y_test,
            "models":          {},
        }

        for mname, model_proto in _MODELS.items():
            import copy
            model = copy.deepcopy(model_proto)

            if mname in _NEEDS_SCALING:
                model.fit(X_tr_s, y_train)
                y_pred = model.predict(X_te_s)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2   = r2_score(y_test, y_pred)
            mape = _mape(y_test, y_pred)

            model_path = os.path.join(
                _MODEL_DIR, f"capacity_{col}_{h}h_{mname.lower().replace(' ', '_')}.pkl"
            )
            joblib.dump({"model": model, "scaler": scaler, "feature_names": fnames}, model_path)

            all_results[key]["models"][mname] = {
                "MAE":    round(mae,  4),
                "RMSE":   round(rmse, 4),
                "R2":     round(r2,   4),
                "MAPE":   round(mape, 4),
                "y_pred": y_pred,
            }

            print(f"    [{_RESOURCE_LABELS.get(col, col)} | {h}s | {mname}]  "
                  f"MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    _save_capacity_report(all_results)
    _plot_model_comparison(all_results)
    _plot_forecasts(all_results)
    _plot_residuals(all_results)

    return all_results


def _save_capacity_report(results: dict) -> None:
    report = {}
    for key, val in results.items():
        report[key] = {
            "resource":  val["resource"],
            "horizon_h": val["horizon_h"],
            "models": {
                mn: {k: v for k, v in mv.items() if k != "y_pred"}
                for mn, mv in val["models"].items()
            },
        }
    out = os.path.join(RESULTS_DIR, "reports", "capacity_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  → {out}")


def _best_model_name(models_dict: dict) -> str:
    return min(models_dict.items(), key=lambda x: x[1]["RMSE"])[0]


def _plot_model_comparison(results: dict) -> None:
    rows = []
    for key, val in results.items():
        for mname, metrics in val["models"].items():
            rows.append({
                "Kaynak": _RESOURCE_LABELS.get(val["resource"], val["resource"]),
                "Ufuk":   f"{val['horizon_h']}s",
                "Model":  mname,
                "RMSE":   metrics["RMSE"],
                "R²":     metrics["R2"],
            })
    df = pd.DataFrame(rows)

    n_horizons = len(FORECAST_HORIZONS)
    fig, axes = plt.subplots(1, n_horizons, figsize=(7 * n_horizons, 6), sharey=False)
    if n_horizons == 1:
        axes = [axes]

    palette = "tab10"
    for ax, h in zip(axes, FORECAST_HORIZONS):
        sub = df[df["Ufuk"] == f"{h}s"]
        sns.barplot(data=sub, x="Kaynak", y="RMSE", hue="Model", ax=ax, palette=palette)
        ax.set_title(f"Kapasite Tahmini RMSE — {h} Saatlik Ufuk")
        ax.set_xlabel("")
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", "capacity_model_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def _plot_forecasts(results: dict) -> None:
    n_rows = len(CAPACITY_TARGETS)
    fig, axes = plt.subplots(n_rows, len(FORECAST_HORIZONS),
                              figsize=(14, 4 * n_rows), squeeze=False)
    test_period = (f"{pd.Timestamp(TRAIN_END_DATE).strftime('%b %Y')} – "
                   f"{pd.Timestamp(END_DATE).strftime('%b %Y')}")
    fig.suptitle(f"Kapasite Tahmin Sonuçları — Test Dönemi ({test_period})", fontsize=12)

    for r_idx, col in enumerate(CAPACITY_TARGETS):
        for h_idx, h in enumerate(FORECAST_HORIZONS):
            key    = f"{col}_{h}h"
            val    = results.get(key)
            if val is None:
                continue
            ts_test  = val["timestamps_test"]
            y_test   = val["y_test"]
            best_m   = _best_model_name(val["models"])
            y_pred   = val["models"][best_m]["y_pred"]
            r2       = val["models"][best_m]["R2"]

            ax = axes[r_idx][h_idx]
            ax.plot(ts_test, y_test,  label="Gerçek", color="steelblue",  linewidth=0.8)
            ax.plot(ts_test, y_pred,  label="Tahmin", color="crimson",    linewidth=0.8, linestyle="--")
            ax.set_title(
                f"{_RESOURCE_LABELS.get(col, col)} — {h}s ufku\n"
                f"({best_m}, R²={r2:.3f})",
                fontsize=9,
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax.tick_params(axis="x", rotation=30)
            if h_idx == 0:
                ax.set_ylabel(_RESOURCE_LABELS.get(col, col))
            if r_idx == 0 and h_idx == 0:
                ax.legend(fontsize=8)
            if col in THRESHOLDS:
                ax.axhline(THRESHOLDS[col], color="red", linestyle=":", linewidth=0.8)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", "capacity_forecasts.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def _plot_residuals(results: dict) -> None:
    n_rows = len(CAPACITY_TARGETS)
    n_cols = len(FORECAST_HORIZONS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle("Kapasite Tahmin Hata Dağılımı (Gerçek − Tahmin)", fontsize=12)

    for r_idx, col in enumerate(CAPACITY_TARGETS):
        for h_idx, h in enumerate(FORECAST_HORIZONS):
            key = f"{col}_{h}h"
            val = results.get(key)
            if val is None:
                continue
            best_m    = _best_model_name(val["models"])
            y_test    = val["y_test"]
            y_pred    = val["models"][best_m]["y_pred"]
            residuals = y_test - y_pred

            ax = axes[r_idx][h_idx]
            ax.hist(residuals, bins=50, color="steelblue", alpha=0.78, edgecolor="none")
            ax.axvline(0,                color="red",    linewidth=1.4, linestyle="--")
            ax.axvline(residuals.mean(), color="orange", linewidth=1.1, linestyle=":",
                       label=f"Ort={residuals.mean():.2f}%")
            ax.set_title(
                f"{_RESOURCE_LABELS.get(col, col)} — {h}s\n{best_m}  σ={residuals.std():.2f}%",
                fontsize=9,
            )
            if r_idx == n_rows - 1:
                ax.set_xlabel("Hata (Gerçek − Tahmin) %")
            if h_idx == 0:
                ax.set_ylabel("Frekans")
            ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", "capacity_residuals.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


def predict_days_to_threshold(feature_df: pd.DataFrame,
                               capacity_results: dict,
                               col: str = "disk_usage_percent",
                               thr: float = None) -> dict:
    """
    24s ufuklu en iyi ML modeli (RMSE bazlı) ile iteratif ileriye tahmin yapar.

    thr=None  → config.THRESHOLDS'tan alınır (90/88/85%)
    thr=100.0 → fiziksel dolum tarihi hesaplanır
    """
    last_ts = feature_df.index[-1]
    if thr is None:
        thr = THRESHOLDS.get(col, 90.0)
    current = float(feature_df[col].dropna().iloc[-1])

    if current >= thr:
        return {
            "resource": col, "current_value": round(current, 2),
            "threshold": thr, "days_to_threshold": 0, "months_to_threshold": 0,
            "estimated_date": str(last_ts.date()), "stable": False,
            "daily_trend": 0.0, "method": "ML", "as_of": str(last_ts),
        }

    key_24  = f"{col}_24h"
    val_24  = capacity_results.get(key_24, {})
    best_m  = None
    if val_24 and val_24.get("models"):
        best_m = min(val_24["models"].items(), key=lambda x: x[1]["RMSE"])[0]

    if best_m is None:
        return _threshold_fallback(feature_df, col, last_ts, thr, current)

    model_path = os.path.join(
        DATA_DIR, "processed", "models",
        f"capacity_{col}_24h_{best_m.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl",
    )
    if not os.path.exists(model_path):
        return _threshold_fallback(feature_df, col, last_ts, thr, current)

    pkg           = joblib.load(model_path)
    model         = pkg["model"]
    scaler        = pkg.get("scaler")
    feature_names = list(pkg["feature_names"])

    # Tüm 1.5 yıllık gerçek veri — kırpılmaz; lag/delta özellikleri
    # büyüme trendini korur (disk gibi yavaş kaynaklarda kritik)
    hist  = feature_df[col].dropna().values.tolist()
    base  = {}
    for f in feature_names:
        if f in feature_df.columns:
            v = feature_df[f].dropna()
            base[f] = float(v.iloc[-1]) if len(v) else 0.0
        else:
            base[f] = 0.0

    STEP_H    = 24
    MAX_STEPS = 365 * 10   # 10 yıl — yavaş büyüyen kaynaklar için yeterli
    elapsed_h = 0

    for _ in range(MAX_STEPS):
        elapsed_h += STEP_H
        pred_ts = last_ts + pd.Timedelta(hours=elapsed_h)
        row = dict(base)
        n   = len(hist)

        if col in feature_names:
            row[col] = hist[-1]
        for lag in [1, 6, 24, 48, 168]:
            f = f"{col}_lag_{lag}h"
            if f in feature_names:
                row[f] = hist[max(0, n - lag)]
        for w in [3, 6, 12, 24, 168]:
            f = f"{col}_rmean_{w}h"
            if f in feature_names:
                row[f] = float(np.mean(hist[-w:]))
        for w in [3, 6, 12, 24]:
            f = f"{col}_rstd_{w}h"
            if f in feature_names:
                wv = hist[-w:]
                row[f] = float(np.std(wv)) if len(wv) > 1 else 0.0
        for d in [1, 6, 24]:
            f = f"{col}_delta_{d}h"
            if f in feature_names:
                row[f] = hist[-1] - hist[max(0, n - d)]
        _time_map = {
            "hour_of_day":      pred_ts.hour,
            "day_of_week":      pred_ts.dayofweek,
            "day_of_month":     pred_ts.day,
            "month":            pred_ts.month,
            "week_of_year":     pred_ts.isocalendar()[1],
            "is_weekend":       int(pred_ts.dayofweek >= 5),
            "is_business_hour": int(8 <= pred_ts.hour <= 18 and pred_ts.dayofweek < 5),
        }
        for feat, val in _time_map.items():
            if feat in feature_names:
                row[feat] = val

        X = np.array([[row[f] for f in feature_names]], dtype=float)
        if best_m in _NEEDS_SCALING and scaler is not None:
            X = scaler.transform(X)

        predicted = float(np.clip(model.predict(X)[0], 0.0, 100.0))
        hist.append(predicted)   # kırpılmaz — tam geçmiş korunur

        if predicted >= thr:
            days        = elapsed_h / 24.0
            months      = round(days / 30.44, 1)
            daily_trend = (predicted - current) / days if days > 0 else 0.0
            return {
                "resource":            col,
                "current_value":       round(current, 2),
                "threshold":           thr,
                "daily_trend":         round(daily_trend, 4),
                "days_to_threshold":   round(days, 1),
                "months_to_threshold": months,
                "estimated_date":      str(pred_ts.date()),
                "stable":              False,
                "method":              f"ML-{best_m}",
                "as_of":               str(last_ts),
            }

    # 10 yıl içinde eşiğe ulaşmadı — gerçekten kararlı veya azalıyor
    return {
        "resource": col, "current_value": round(current, 2),
        "threshold": thr, "days_to_threshold": None, "months_to_threshold": None,
        "estimated_date": None, "stable": True,
        "daily_trend": 0.0, "method": f"ML-{best_m}",
        "as_of": str(last_ts),
    }


def _threshold_fallback(feature_df, col, last_ts, thr, current):
    """Model dosyası bulunamadığında devreye girer. Tüm geçmişe OLS uydurur."""
    vals = feature_df[col].dropna().values
    if len(vals) < 168:
        return {"resource": col, "current_value": round(current, 2), "threshold": thr,
                "days_to_threshold": None, "months_to_threshold": None,
                "estimated_date": None, "stable": True,
                "daily_trend": 0.0, "method": "OLS-fallback", "as_of": str(last_ts)}
    x = np.arange(len(vals), dtype=float)
    slope, _ = np.polyfit(x, vals, 1)
    daily_slope = slope * 24
    if slope <= 0:
        return {"resource": col, "current_value": round(current, 2), "threshold": thr,
                "days_to_threshold": None, "months_to_threshold": None,
                "estimated_date": None, "stable": True,
                "daily_trend": round(daily_slope, 4), "method": "OLS-fallback",
                "as_of": str(last_ts)}
    hours_to_thr = (thr - current) / slope
    days         = hours_to_thr / 24.0
    months       = round(days / 30.44, 1)
    target       = last_ts + pd.Timedelta(hours=hours_to_thr)
    return {"resource": col, "current_value": round(current, 2), "threshold": thr,
            "daily_trend": round(daily_slope, 4), "days_to_threshold": round(days, 0),
            "months_to_threshold": months, "estimated_date": str(target.date()),
            "stable": False, "method": "OLS-fallback", "as_of": str(last_ts)}
