"""
Log verisi üretici — OpenStack kontrol düzlemi servisleri.

Staj dönemlerinde quota_exceeded ve resource_exhausted hataları belirgin biçimde artar.
İki çıktı:
  1. log_features.csv  — ML için saatlik agregasyon
  2. raw_logs.csv      — Gerçekçi ham log örnekleri
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SEED, RESULTS_DIR, INCIDENTS, START_DATE, END_DATE, STAJ_PERIODS


_TEMPLATES = {
    "timeout": [
        "Timeout connecting to {svc} after {ms}ms",
        "Request to {svc} timed out (deadline exceeded)",
        "RPC call to {svc} exceeded timeout of {ms}ms",
        "Service {svc} did not respond within {ms}ms, giving up",
    ],
    "db_connection": [
        "DBConnectionError: Unable to connect to MySQL: connection refused",
        "Connection pool exhausted — no available DB connections",
        "Lost connection to MySQL server during query",
        "oslo_db: Database connection error — retry attempt {n} failed",
        "peewee.OperationalError: (2003, Can't connect to MySQL server)",
    ],
    "resource_exhausted": [
        "No valid host was found — insufficient RAM on compute nodes",
        "nova.scheduler: Filter scheduling failed — all hosts exhausted",
        "No hypervisor resources available to fulfill request",
        "nova.compute.resource_tracker: Insufficient vCPUs available",
        "Compute node overcommit ratio exceeded — VM placement failed",
    ],
    "quota_exceeded": [
        "OSError: QuotaExceeded: Disk quota exceeded for project {proj}",
        "nova.quota: vCPU quota exceeded — user {user} requested {n} vCPUs",
        "cinder.quota: Volume count quota exceeded for project {proj}",
        "Floating IP quota reached for project {proj} — request denied",
        "Network quota exceeded: too many ports for project {proj}",
        "RAM quota exceeded for project {proj} — increase quota or free resources",
    ],
    "auth_failure": [
        "keystoneauth1.exceptions: Token validation failed — token expired",
        "Invalid credentials for user {user}",
        "Permission denied: action not allowed for role {role}",
        "Token revocation list lookup failed",
        "User {user} login failed — too many attempts, account temporarily locked",
    ],
    "network_error": [
        "neutron.plugins: VLAN {vlan} configuration failed on host {host}",
        "Floating IP allocation failed — pool exhausted",
        "Port {port} binding failed on network node",
        "OVS bridge missing on host {host}",
    ],
    "scheduler_error": [
        "nova.scheduler: No valid host found after {n} retries",
        "Retry limit exceeded for instance placement",
        "WeightedHost selection failed — all weights zero",
        "nova.scheduler.filters: ComputeFilter rejected all hosts",
        "VM creation failed: scheduler could not find suitable host for flavor {n}",
    ],
    "io_error": [
        "cinder.volume: I/O error writing to volume {vol}",
        "Volume attachment failed — iSCSI connection refused",
        "Snapshot {snap} creation timed out",
    ],
}

_SERVICES = {
    "nova_api":       "nova.api",
    "nova_compute":   "nova.compute",
    "nova_scheduler": "nova.scheduler",
    "keystone":       "keystone.token",
    "neutron":        "neutron.server",
    "cinder":         "cinder.volume",
}

_SERVICE_ERROR_AFFINITY = {
    "nova_api":       ["timeout", "resource_exhausted", "auth_failure", "quota_exceeded"],
    "nova_compute":   ["resource_exhausted", "timeout", "io_error", "quota_exceeded"],
    "nova_scheduler": ["scheduler_error", "resource_exhausted", "timeout"],
    "keystone":       ["auth_failure", "db_connection", "timeout"],
    "neutron":        ["network_error", "timeout", "db_connection"],
    "cinder":         ["io_error", "db_connection", "timeout", "quota_exceeded"],
}


def _staj_factor(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Staj dönemlerinde resource_exhausted ve quota_exceeded hataları
    normale göre 3-4x daha fazla görülür.
    """
    factor = np.ones(len(timestamps))
    ts_f = timestamps.astype(np.int64).values.astype(float) / 1e9
    for staj in STAJ_PERIODS:
        ss = pd.Timestamp(staj["start"]).timestamp()
        es = pd.Timestamp(staj["end"]).timestamp()
        factor[(ts_f >= ss) & (ts_f <= es)] = 3.5
    return factor


def _sample_message(error_type: str, rng: np.random.Generator) -> str:
    templates = _TEMPLATES.get(error_type, ["Unknown error in service"])
    tmpl = templates[int(rng.integers(len(templates)))]
    return tmpl.format(
        svc=random.choice(list(_SERVICES.values())),
        ms=int(rng.integers(500, 8000)),
        n=int(rng.integers(3, 10)),
        proj=f"project-staj-{int(rng.integers(1000, 9999))}",
        user=f"staj-user-{int(rng.integers(100, 999))}",
        role=random.choice(["member", "reader", "compute_admin"]),
        vlan=int(rng.integers(100, 4094)),
        host=f"compute-node-{int(rng.integers(1, 12)):02d}",
        port=f"port-{int(rng.integers(10000, 99999))}",
        vol=f"volume-{int(rng.integers(1000, 9999))}",
        snap=f"snap-{int(rng.integers(1000, 9999))}",
    )


def generate_log_features(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Metrik verisiyle korele saatlik log sayımları üretir.
    Staj dönemlerinde quota_exceeded ve resource_exhausted belirgin artar.
    """
    rng = np.random.default_rng(SEED + 1)
    n   = len(metrics_df)

    cpu_f     = np.clip((metrics_df["cpu_usage_percent"].values     - 60) / 40, 0, 1)
    mem_f     = np.clip((metrics_df["memory_usage_percent"].values  - 70) / 30, 0, 1)
    lat_f     = np.clip((metrics_df["api_latency_ms"].values        - 200) / 800, 0, 1)
    err_f     = metrics_df["error_rate_percent"].values / 100.0
    req_base  = metrics_df["api_requests_per_hour"].values / 50.0
    staj_f    = _staj_factor(pd.DatetimeIndex(metrics_df["timestamp"]))

    # ── Servis başına hata sayısı ──────────────────────────────────────────────
    nova_api_e   = np.clip(
        req_base * err_f * 0.26 * (1 + 2.0 * cpu_f) + rng.poisson(0.4, n), 0, None
    ).astype(int)
    nova_comp_e  = np.clip(
        req_base * err_f * 0.21 * (1 + 2.5 * cpu_f + 2.0 * mem_f) + rng.poisson(0.3, n), 0, None
    ).astype(int)
    nova_sched_e = np.clip(
        req_base * err_f * 0.13 * (1 + 3.0 * cpu_f) + rng.poisson(0.2, n), 0, None
    ).astype(int)
    keystone_e   = np.clip(
        req_base * err_f * 0.10 * (1 + 1.5 * lat_f) + rng.poisson(0.2, n), 0, None
    ).astype(int)
    neutron_e    = np.clip(
        req_base * err_f * 0.17 * (1 + 1.8 * lat_f) + rng.poisson(0.2, n), 0, None
    ).astype(int)
    cinder_e     = np.clip(
        req_base * err_f * 0.13 + rng.poisson(0.1, n), 0, None
    ).astype(int)

    total_e = nova_api_e + nova_comp_e + nova_sched_e + keystone_e + neutron_e + cinder_e

    # ── Hata tipi dağılımı ────────────────────────────────────────────────────
    timeout_c     = np.clip(
        total_e * (0.35 + 0.25 * lat_f) + rng.poisson(0.4, n), 0, None
    ).astype(int)
    db_conn_c     = np.clip(
        total_e * (0.12 + 0.10 * lat_f) + rng.poisson(0.1, n), 0, None
    ).astype(int)
    # resource_exhausted ve quota_exceeded staj döneminde yükseliyor
    res_exhaust_c = np.clip(
        total_e * (0.12 + 0.18 * cpu_f + 0.15 * mem_f) * (1 + 0.6 * (staj_f - 1) / 2.5)
        + rng.poisson(0.1, n), 0, None
    ).astype(int)
    quota_exc_c   = np.clip(
        total_e * 0.04 * staj_f + rng.poisson(0.05, n), 0, None
    ).astype(int)
    auth_fail_c   = np.clip(
        total_e * 0.08 * (1 + 0.3 * (staj_f - 1) / 2.5) + rng.poisson(0.3, n), 0, None
    ).astype(int)
    network_err_c = np.clip(
        total_e * (0.09 + 0.08 * lat_f) + rng.poisson(0.1, n), 0, None
    ).astype(int)
    sched_err_c   = np.clip(
        total_e * (0.07 + 0.12 * cpu_f) * (1 + 0.4 * (staj_f - 1) / 2.5)
        + rng.poisson(0.1, n), 0, None
    ).astype(int)
    io_err_c      = np.clip(
        total_e * 0.06 + rng.poisson(0.05, n), 0, None
    ).astype(int)

    critical_c = np.clip(
        (total_e * 0.08 * err_f) + rng.poisson(0.03, n), 0, None
    ).astype(int)

    total_logs = np.clip(req_base * 8 + rng.normal(0, 15, n), 30, None).astype(int)
    err_ratio  = np.where(total_logs > 0, total_e / total_logs, 0.0)

    return pd.DataFrame({
        "timestamp":                metrics_df["timestamp"].values,
        "nova_api_errors":          nova_api_e,
        "nova_compute_errors":      nova_comp_e,
        "nova_scheduler_errors":    nova_sched_e,
        "keystone_errors":          keystone_e,
        "neutron_errors":           neutron_e,
        "cinder_errors":            cinder_e,
        "timeout_count":            timeout_c,
        "db_connection_count":      db_conn_c,
        "resource_exhausted_count": res_exhaust_c,
        "quota_exceeded_count":     quota_exc_c,
        "auth_failure_count":       auth_fail_c,
        "network_error_count":      network_err_c,
        "scheduler_error_count":    sched_err_c,
        "io_error_count":           io_err_c,
        "critical_count":           critical_c,
        "total_errors":             total_e,
        "total_log_count":          total_logs,
        "error_log_ratio":          err_ratio,
    })


def generate_raw_logs(metrics_df: pd.DataFrame, n_samples: int = 5000) -> pd.DataFrame:
    """
    Gerçekçi OpenStack log satırları üretir.
    Staj dönemlerinde quota_exceeded ve scheduler_error örnekleri daha sık seçilir.
    """
    rng    = np.random.default_rng(SEED + 2)
    random.seed(SEED + 2)
    rows   = []

    err_weights = metrics_df["error_rate_percent"].values + 0.1
    err_weights = err_weights / err_weights.sum()
    chosen_idx  = rng.choice(len(metrics_df), size=n_samples, p=err_weights)

    ts_index = pd.DatetimeIndex(metrics_df["timestamp"])
    staj_mask = np.zeros(len(metrics_df), dtype=bool)
    for staj in STAJ_PERIODS:
        staj_mask |= (
            (ts_index >= pd.Timestamp(staj["start"])) &
            (ts_index <= pd.Timestamp(staj["end"]))
        )

    for idx in chosen_idx:
        row      = metrics_df.iloc[idx]
        ts_base  = row["timestamp"]
        offset_s = int(rng.integers(0, 3600))
        log_ts   = ts_base + pd.Timedelta(seconds=offset_s)

        svc_key  = random.choice(list(_SERVICE_ERROR_AFFINITY.keys()))

        # Staj döneminde quota/scheduler hatası ağırlığı artır
        if staj_mask[idx]:
            err_pool = _SERVICE_ERROR_AFFINITY[svc_key] + ["quota_exceeded", "scheduler_error"]
        else:
            err_pool = _SERVICE_ERROR_AFFINITY[svc_key]

        err_type  = random.choice(err_pool)
        err_rate  = row["error_rate_percent"]
        level     = "CRITICAL" if (rng.random() < 0.08 and err_rate > 3) else "ERROR"
        message   = _sample_message(err_type, rng)
        req_id    = f"req-{rng.integers(10**14, 10**15):016x}"

        rows.append({
            "timestamp":  log_ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "level":      level,
            "service":    _SERVICES[svc_key],
            "request_id": req_id,
            "error_type": err_type,
            "message":    message,
            "staj_period": staj_mask[idx],
        })

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def plot_log_analysis(log_feat_df: pd.DataFrame) -> None:
    """Servis bazlı hata dağılımı ve zaman serisi grafiğini kaydeder."""
    n_months = round((pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days / 30)
    date_range_str = (
        f"{pd.Timestamp(START_DATE).strftime('%b %Y')} – "
        f"{pd.Timestamp(END_DATE).strftime('%b %Y')}"
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f"OpenStack Servis Log Analizi — {n_months} Aylık Hata Dağılımı ({date_range_str})",
        fontsize=12,
    )

    ts = log_feat_df["timestamp"]

    services = [
        ("nova_api_errors",       "Nova API",       "steelblue"),
        ("nova_compute_errors",   "Nova Compute",   "darkorange"),
        ("nova_scheduler_errors", "Nova Scheduler", "green"),
        ("keystone_errors",       "Keystone",       "red"),
        ("neutron_errors",        "Neutron",        "purple"),
        ("cinder_errors",         "Cinder",         "brown"),
    ]
    bottoms = np.zeros(len(log_feat_df))
    for col, label, color in services:
        vals = log_feat_df[col].values.astype(float)
        ax1.fill_between(ts, bottoms, bottoms + vals, alpha=0.75, label=label, color=color)
        bottoms += vals

    for staj in STAJ_PERIODS:
        ax1.axvspan(pd.Timestamp(staj["start"]), pd.Timestamp(staj["end"]),
                    alpha=0.10, color="royalblue", linewidth=0)

    ax1.set_ylabel("Saatlik Hata Sayısı")
    ax1.legend(loc="upper left", fontsize=8, ncol=3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    for inc in INCIDENTS:
        ax1.axvspan(pd.Timestamp(inc["start"]), pd.Timestamp(inc["end"]),
                    alpha=0.12, color="red", linewidth=0)

    err_types = {
        "Timeout":             log_feat_df["timeout_count"].sum(),
        "DB Bağlantı":         log_feat_df["db_connection_count"].sum(),
        "Kaynak Yetersiz":     log_feat_df["resource_exhausted_count"].sum(),
        "Kota Aşımı":          log_feat_df["quota_exceeded_count"].sum(),
        "Kimlik Doğrulama":    log_feat_df["auth_failure_count"].sum(),
        "Ağ Hatası":           log_feat_df["network_error_count"].sum(),
        "Zamanlayıcı Hatası":  log_feat_df["scheduler_error_count"].sum(),
        "I/O Hatası":          log_feat_df["io_error_count"].sum(),
    }
    ax2.barh(list(err_types.keys()), list(err_types.values()),
             color="steelblue", alpha=0.75)
    ax2.set_xlabel(f"Toplam Hata Sayısı ({n_months} ay)")
    ax2.set_title("Hata Tipi Dağılımı (mavi gölge = staj dönemleri)")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", "log_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")
