"""
Log verisi üretici — OpenStack kontrol düzlemi servisleri.

Tez hedefi: Problem gelişimini önceden sezmek.
Bu modül, gerçek sistemlerde arıza öncesi görülen ERKEN UYARI sinyallerini
içeren zengin log verisi üretir.

Log kategorileri:
  ── Hata logları (mevcut)
  1. timeout            — servis zaman aşımı
  2. db_connection      — veritabanı bağlantı sorunları
  3. resource_exhausted — kaynak tükenmesi
  4. quota_exceeded     — kota aşımı
  5. auth_failure       — kimlik doğrulama hatası
  6. network_error      — ağ katmanı hataları
  7. scheduler_error    — zamanlayıcı hataları
  8. io_error           — disk G/Ç hataları

  ── Proaktif erken uyarı logları (yeni — tezin temel katkısı)
  9. slow_response      — giderek artan yanıt süreleri (arıza öncesi)
  10. memory_pressure   — bellek baskısı uyarıları (OOM öncesi)
  11. connection_pool_warning — bağlantı havuzu doluyor (tükenme öncesi)
  12. scheduler_degradation   — zamanlayıcı yavaşlaması (arıza öncesi)
  13. quota_pressure    — kotaya yaklaşma (aşım öncesi)
  14. disk_io_degradation     — disk yavaşlaması (arıza öncesi)
  15. cascade_indicator       — çok servisli bozulma sinyalleri

İki çıktı:
  1. raw_logs.csv      — Gerçekçi ham log satırları
  2. log_features.csv  — ML için saatlik agregasyon
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SEED, RESULTS_DIR, INCIDENTS, START_DATE, END_DATE, STAJ_PERIODS, VCPU_TOTAL


# ── Log şablonları ────────────────────────────────────────────────────────────

_TEMPLATES = {
    # ── Klasik hata logları ────────────────────────────────────────────────────
    "timeout": [
        "Timeout connecting to {svc} after {ms}ms",
        "Request to {svc} timed out (deadline exceeded after {ms}ms)",
        "RPC call to {svc} exceeded timeout limit of {ms}ms",
        "Service {svc} did not respond within {ms}ms",
        "HTTP 504 Gateway Timeout on {svc} endpoint after {ms}ms",
    ],
    "db_connection": [
        "DBConnectionError: Unable to connect to MySQL — connection refused",
        "Connection pool exhausted — no available DB connections ({n}/{n} slots used)",
        "Lost connection to MySQL server during query — errno=2006",
        "oslo_db: Database connection retry attempt {n} of 5 failed",
        "peewee.OperationalError: Can't connect to MySQL server on '127.0.0.1'",
        "oslo.messaging: AMQP connection lost, attempting reconnect ({n}/3)",
    ],
    "resource_exhausted": [
        "nova.scheduler: No valid host found — all compute nodes at vCPU capacity",
        "Insufficient vCPUs available on any host (requested: {n} vCPUs)",
        "nova.compute.resource_tracker: vCPU overcommit ratio exceeded on {host}",
        "VM placement failed: filter AvailableVCPUFilter rejected all {n} hosts",
        "nova.conductor: BuildAbortException — resource_exhausted after {n} retries",
        "Resource claim failed on {host}: vcpus_used={n} + requested > vcpus_total",
    ],
    "quota_exceeded": [
        "nova.quota: vCPU quota exceeded for project {proj} (limit: {n}, current: {n})",
        "cinder.quota: Volume count limit reached for project {proj}",
        "OSError: QuotaExceeded: Disk quota exceeded for project {proj}",
        "Floating IP quota reached for project {proj} — request denied",
        "RAM quota exceeded for project {proj}: requested {n}MB, quota {n}MB",
        "nova.api: 403 Forbidden — quota:instances exceeded for project {proj}",
    ],
    "auth_failure": [
        "keystoneauth1: Token validation failed — token expired or revoked",
        "Invalid credentials for user {user} — incorrect password or token",
        "Permission denied: action {n} not allowed for role 'member'",
        "User {user} login failed — account temporarily locked (5 attempts)",
        "Keystone: Token revocation list size exceeds safe threshold",
        "keystoneauth1.exceptions.http.Unauthorized: 401 Unauthorized",
    ],
    "network_error": [
        "neutron.plugins: VLAN {vlan} configuration failed on {host}",
        "Floating IP allocation failed — external network pool exhausted",
        "Port {port} binding failed — no available agent on {host}",
        "OVS bridge br-int missing or unresponsive on {host}",
        "neutron: L3 agent failed to configure router on {host}",
        "BGP route propagation failure detected for subnet on {host}",
    ],
    "scheduler_error": [
        "nova.scheduler: No valid host found after {n} retries",
        "WeightedHost selection failed — all host weights are zero",
        "nova.scheduler.filters.ComputeFilter: all {n} hosts rejected",
        "VM creation failed: scheduler timeout after {ms}ms",
        "nova.scheduler: RetryFilter eliminated all hosts after {n} attempts",
    ],
    "io_error": [
        "cinder.volume: I/O error writing to volume {vol} — errno=5",
        "Volume attachment failed — iSCSI target {vol} connection refused",
        "Snapshot {snap} creation timed out after {ms}ms",
        "cinder: LUN {vol} read error — storage backend unresponsive",
        "iSCSI session to {host} lost — initiator reconnecting",
    ],

    # ── Proaktif erken uyarı logları ──────────────────────────────────────────
    "slow_response": [
        "WARN nova.api: Response time degraded — {svc} p95={ms}ms (baseline: 150ms)",
        "WARN Endpoint latency rising on {svc}: avg={ms}ms (SLA threshold: 500ms)",
        "WARN oslo.service: {svc} handler queue depth increasing: {n} pending requests",
        "WARN nova.compute: VM operation took {ms}ms (expected: <120ms), retrying",
        "WARN api: Slow response detected on {svc} — consecutive slow calls: {n}",
        "WARN keystone.middleware: Auth validation latency high: {ms}ms",
        "WARN nova.scheduler: Host selection latency degraded to {ms}ms (normal: 45ms)",
    ],
    "memory_pressure": [
        "WARN nova.compute: Host {host} memory utilization at {n}% — approaching limit",
        "WARN oslo.cache: Memory cache hit rate dropped to {n}% — eviction rate high",
        "WARN nova.compute: Balloon driver reports high memory pressure on {host}",
        "WARN cinder.volume: iSCSI buffer pool at {n}% capacity on {host}",
        "WARN nova.compute: KSM (kernel samepage merging) activated on {host} — memory tight",
        "WARN nova.libvirt: Instance {snap} swapping to disk — RAM exhaustion imminent",
        "WARN oslo.db: Connection pool GC pressure — {n} connections awaiting release",
    ],
    "connection_pool_warning": [
        "WARN oslo.db: DB connection pool at {n}% capacity ({n}/{n} connections active)",
        "WARN nova.api: Connection pool high watermark reached: {n}/{n} connections",
        "WARN oslo.db: Connection acquisition wait time elevated: avg {ms}ms",
        "WARN nova.compute: RPC connection pool saturation: {n} requests queued",
        "WARN keystone.db: DB connection reuse rate declining — pool pressure",
        "WARN oslo.messaging: AMQP consumer prefetch limit approaching: {n}/{n}",
        "WARN cinder.api: Backend connection pool at {n}% — {n} requests queuing",
    ],
    "scheduler_degradation": [
        "WARN nova.scheduler: Host selection time elevated: {ms}ms (normal: 45ms)",
        "WARN nova.scheduler: Filter pass rate: {n}% — host pool nearly exhausted",
        "WARN nova.scheduler: Retry count per placement increasing: avg {n} retries",
        "WARN nova.conductor: Scheduler response time degraded by {n}% vs. last hour",
        "WARN nova.scheduler: WeightedHost score variance dropping — all hosts similar load",
        "WARN nova.scheduler: RetryFilter seeing increasing rejections: {n}% of requests",
        "WARN nova.scheduler: Host capacity headroom shrinking — avg free_vcpus={n}",
    ],
    "quota_pressure": [
        "WARN nova.quota: Project {proj} vCPU usage at {n}% of quota — approaching limit",
        "WARN cinder.quota: Project {proj} volume count at {n}% of limit",
        "WARN neutron.quota: Project {proj} port quota at {n}% consumed",
        "WARN nova.quota: Project {proj} RAM usage {n}MB of {n}MB quota",
        "WARN keystone: Tenant {proj} API call rate at {n}% of rate limit",
        "WARN nova.api: Project {proj} server count approaching quota: {n}/{n}",
    ],
    "disk_io_degradation": [
        "WARN cinder.volume: I/O latency elevated on backend: p99={ms}ms (normal: 45ms)",
        "WARN cinder.volume: Write throughput degraded to {n}MB/s (baseline: 450MB/s)",
        "WARN nova.compute: Disk I/O wait high on {host}: iowait={n}%",
        "WARN cinder: Storage array queue depth high: {n} I/O ops queued",
        "WARN nova.libvirt: Block device throttling active on {host} — I/O pressure",
        "WARN cinder.volume: iSCSI reconnect rate increasing: {n} reconnects/hour",
        "WARN nova.compute: Ephemeral disk fill rate accelerating on {host}",
    ],
    "cascade_indicator": [
        "WARN nova.conductor: Multiple services reporting degraded state simultaneously",
        "WARN nova.api: Cross-service latency correlation detected — system under stress",
        "WARN oslo.service: {n} services exceeded response SLA in last 10 minutes",
        "WARN nova.compute: Cascading failure pattern detected: nova + neutron + keystone",
        "WARN openstack.health: System-wide error rate spike: {n}% above baseline",
        "WARN nova.conductor: Dependency graph saturation — {n} pending cross-service calls",
        "WARN oslo.service: Health check degradation detected on {n} components",
    ],
}

_SERVICES = {
    "nova_api":       "nova.api",
    "nova_compute":   "nova.compute",
    "nova_scheduler": "nova.scheduler",
    "keystone":       "keystone.token",
    "neutron":        "neutron.server",
    "cinder":         "cinder.volume",
    "nova_conductor": "nova.conductor",
}

_SERVICE_ERROR_AFFINITY = {
    "nova_api":       ["timeout", "resource_exhausted", "auth_failure", "quota_exceeded",
                       "slow_response", "connection_pool_warning"],
    "nova_compute":   ["resource_exhausted", "timeout", "io_error", "quota_exceeded",
                       "memory_pressure", "disk_io_degradation"],
    "nova_scheduler": ["scheduler_error", "resource_exhausted", "timeout",
                       "scheduler_degradation"],
    "keystone":       ["auth_failure", "db_connection", "timeout",
                       "slow_response", "connection_pool_warning"],
    "neutron":        ["network_error", "timeout", "db_connection",
                       "slow_response"],
    "cinder":         ["io_error", "db_connection", "timeout", "quota_exceeded",
                       "disk_io_degradation"],
    "nova_conductor": ["cascade_indicator", "timeout", "scheduler_degradation",
                       "slow_response", "db_connection"],
}

# Erken uyarı log tipleri (WARNING severity)
_WARNING_TYPES = {
    "slow_response", "memory_pressure", "connection_pool_warning",
    "scheduler_degradation", "quota_pressure", "disk_io_degradation",
    "cascade_indicator",
}


def _staj_factor(timestamps: pd.DatetimeIndex) -> np.ndarray:
    factor = np.ones(len(timestamps))
    ts_f = timestamps.astype(np.int64).values.astype(float) / 1e9
    for staj in STAJ_PERIODS:
        ss = pd.Timestamp(staj["start"]).timestamp()
        es = pd.Timestamp(staj["end"]).timestamp()
        factor[(ts_f >= ss) & (ts_f <= es)] = 3.5
    return factor


def _buildup_factor(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Incident öncesi buildup dönemlerinde erken uyarı logları artar."""
    factor = np.ones(len(timestamps))
    ts_f = timestamps.astype(np.int64).values.astype(float) / 1e9
    for inc in INCIDENTS:
        buildup_start = (pd.Timestamp(inc["start"]) - pd.Timedelta(days=inc["buildup_days"])).timestamp()
        inc_start = pd.Timestamp(inc["start"]).timestamp()
        mask = (ts_f >= buildup_start) & (ts_f < inc_start)
        if mask.any():
            frac = (ts_f[mask] - buildup_start) / max(inc_start - buildup_start, 1.0)
            factor[mask] = np.maximum(factor[mask], 1.0 + 4.0 * frac ** 2)
    return factor


def _sample_message(error_type: str, rng: np.random.Generator) -> str:
    templates = _TEMPLATES.get(error_type, ["Unknown service event"])
    tmpl = templates[int(rng.integers(len(templates)))]
    return tmpl.format(
        svc=random.choice(list(_SERVICES.values())),
        ms=int(rng.integers(150, 8000)),
        n=int(rng.integers(3, 98)),
        proj=f"project-{random.choice(['staj','research','prod','dev'])}-{int(rng.integers(1000, 9999))}",
        user=f"user-{int(rng.integers(100, 999))}",
        vlan=int(rng.integers(100, 4094)),
        host=f"compute-node-{int(rng.integers(1, 16)):02d}",
        port=f"port-{int(rng.integers(10000, 99999))}",
        vol=f"volume-{int(rng.integers(1000, 9999))}",
        snap=f"snap-{int(rng.integers(1000, 9999))}",
    )


def generate_log_features(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Metrik verisiyle korele saatlik log sayımları üretir.
    Hem hata logları hem de erken uyarı logları üretilir.
    """
    rng = np.random.default_rng(SEED + 1)
    n   = len(metrics_df)

    vcpu_f    = np.clip((metrics_df["vcpu_usage_percent"].values    - 50) / 50, 0, 1)
    mem_f     = np.clip((metrics_df["memory_usage_percent"].values  - 60) / 40, 0, 1)
    lat_f     = np.clip((metrics_df["api_latency_ms"].values        - 200) / 800, 0, 1)
    disk_f    = np.clip((metrics_df["disk_usage_percent"].values    - 55) / 45, 0, 1)
    err_f     = metrics_df["error_rate_percent"].values / 100.0
    req_base  = metrics_df["api_requests_per_hour"].values / 50.0
    staj_f    = _staj_factor(pd.DatetimeIndex(metrics_df["timestamp"]))
    buildup_f = _buildup_factor(pd.DatetimeIndex(metrics_df["timestamp"]))

    # ── Hata sayıları (servis bazlı) ──────────────────────────────────────────
    nova_api_e   = np.clip(
        req_base * err_f * 0.26 * (1 + 2.0 * vcpu_f) + rng.poisson(0.4, n), 0, None
    ).astype(int)
    nova_comp_e  = np.clip(
        req_base * err_f * 0.21 * (1 + 2.5 * vcpu_f + 2.0 * mem_f) + rng.poisson(0.3, n), 0, None
    ).astype(int)
    nova_sched_e = np.clip(
        req_base * err_f * 0.13 * (1 + 3.5 * vcpu_f) + rng.poisson(0.2, n), 0, None
    ).astype(int)
    keystone_e   = np.clip(
        req_base * err_f * 0.10 * (1 + 1.5 * lat_f) + rng.poisson(0.2, n), 0, None
    ).astype(int)
    neutron_e    = np.clip(
        req_base * err_f * 0.17 * (1 + 1.8 * lat_f) + rng.poisson(0.2, n), 0, None
    ).astype(int)
    cinder_e     = np.clip(
        req_base * err_f * 0.13 * (1 + 2.0 * disk_f) + rng.poisson(0.1, n), 0, None
    ).astype(int)

    total_e = nova_api_e + nova_comp_e + nova_sched_e + keystone_e + neutron_e + cinder_e

    # ── Hata tipi dağılımı ────────────────────────────────────────────────────
    timeout_c      = np.clip(total_e * (0.32 + 0.22 * lat_f)  + rng.poisson(0.4, n), 0, None).astype(int)
    db_conn_c      = np.clip(total_e * (0.12 + 0.10 * lat_f)  + rng.poisson(0.1, n), 0, None).astype(int)
    res_exhaust_c  = np.clip(
        total_e * (0.10 + 0.22 * vcpu_f + 0.12 * mem_f) * (1 + 0.6 * (staj_f - 1) / 2.5)
        + rng.poisson(0.1, n), 0, None).astype(int)
    quota_exc_c    = np.clip(total_e * 0.04 * staj_f + rng.poisson(0.05, n), 0, None).astype(int)
    auth_fail_c    = np.clip(
        total_e * 0.08 * (1 + 0.3 * (staj_f - 1) / 2.5) + rng.poisson(0.3, n), 0, None).astype(int)
    network_err_c  = np.clip(total_e * (0.08 + 0.07 * lat_f)  + rng.poisson(0.1, n), 0, None).astype(int)
    sched_err_c    = np.clip(
        total_e * (0.06 + 0.14 * vcpu_f) * (1 + 0.4 * (staj_f - 1) / 2.5)
        + rng.poisson(0.1, n), 0, None).astype(int)
    io_err_c       = np.clip(total_e * (0.05 + 0.10 * disk_f) + rng.poisson(0.05, n), 0, None).astype(int)
    critical_c     = np.clip((total_e * 0.08 * err_f)          + rng.poisson(0.03, n), 0, None).astype(int)

    # ── Erken Uyarı Logları (proaktif) ────────────────────────────────────────
    # Bu loglar arıza öncesi dönemde artış gösterir; buildup_f ile güçlenir.
    slow_resp_c    = np.clip(
        (req_base * 0.06 * (1 + 3.0 * lat_f) + 0.3) * buildup_f
        + rng.poisson(0.5, n), 0, None).astype(int)

    mem_pressure_c = np.clip(
        (req_base * 0.04 * (1 + 4.0 * mem_f) + 0.2) * buildup_f
        + rng.poisson(0.3, n), 0, None).astype(int)

    conn_pool_warn_c = np.clip(
        (req_base * 0.05 * (1 + 2.5 * lat_f + 1.5 * vcpu_f) + 0.2) * buildup_f
        + rng.poisson(0.4, n), 0, None).astype(int)

    sched_degrad_c = np.clip(
        (req_base * 0.03 * (1 + 5.0 * vcpu_f) + 0.1) * buildup_f
        + rng.poisson(0.2, n), 0, None).astype(int)

    quota_press_c  = np.clip(
        (req_base * 0.02 * staj_f * (1 + vcpu_f) + 0.1) * buildup_f
        + rng.poisson(0.15, n), 0, None).astype(int)

    disk_io_degrad_c = np.clip(
        (req_base * 0.03 * (1 + 4.0 * disk_f) + 0.1) * buildup_f
        + rng.poisson(0.2, n), 0, None).astype(int)

    # Cascade indicator — birden fazla servis aynı anda bozulunca artar
    multi_service_stress = (vcpu_f + mem_f + lat_f) / 3.0
    cascade_c  = np.clip(
        (req_base * 0.015 * (1 + 6.0 * multi_service_stress) + 0.05) * buildup_f
        + rng.poisson(0.1, n), 0, None).astype(int)

    total_warning_c = (slow_resp_c + mem_pressure_c + conn_pool_warn_c
                       + sched_degrad_c + quota_press_c + disk_io_degrad_c + cascade_c)

    total_logs = np.clip(req_base * 8 + total_warning_c * 2 + rng.normal(0, 15, n), 30, None).astype(int)
    err_ratio  = np.where(total_logs > 0, total_e / total_logs, 0.0)

    return pd.DataFrame({
        "timestamp":                  metrics_df["timestamp"].values,
        # Servis bazlı hata sayıları
        "nova_api_errors":            nova_api_e,
        "nova_compute_errors":        nova_comp_e,
        "nova_scheduler_errors":      nova_sched_e,
        "keystone_errors":            keystone_e,
        "neutron_errors":             neutron_e,
        "cinder_errors":              cinder_e,
        # Hata tipi sayıları
        "timeout_count":              timeout_c,
        "db_connection_count":        db_conn_c,
        "resource_exhausted_count":   res_exhaust_c,
        "quota_exceeded_count":       quota_exc_c,
        "auth_failure_count":         auth_fail_c,
        "network_error_count":        network_err_c,
        "scheduler_error_count":      sched_err_c,
        "io_error_count":             io_err_c,
        "critical_count":             critical_c,
        # Erken uyarı log sayıları
        "slow_response_count":        slow_resp_c,
        "memory_pressure_count":      mem_pressure_c,
        "connection_pool_warn_count": conn_pool_warn_c,
        "scheduler_degradation_count":sched_degrad_c,
        "quota_pressure_count":       quota_press_c,
        "disk_io_degradation_count":  disk_io_degrad_c,
        "cascade_indicator_count":    cascade_c,
        # Toplamlar
        "total_errors":               total_e,
        "total_warning_count":        total_warning_c,
        "total_log_count":            total_logs,
        "error_log_ratio":            err_ratio,
    })


def generate_raw_logs(metrics_df: pd.DataFrame, n_samples: int = 8000) -> pd.DataFrame:
    """
    Gerçekçi OpenStack log satırları üretir.
    Arıza öncesi buildup döneminde erken uyarı logları ağırlıklı seçilir.
    """
    rng    = np.random.default_rng(SEED + 2)
    random.seed(SEED + 2)
    rows   = []

    # Hata oranı + erken uyarı baskısı ile örnek seçimi
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

    buildup_mask = np.zeros(len(metrics_df), dtype=bool)
    for inc in INCIDENTS:
        buildup_start = pd.Timestamp(inc["start"]) - pd.Timedelta(days=inc["buildup_days"])
        buildup_mask |= (
            (ts_index >= buildup_start) &
            (ts_index < pd.Timestamp(inc["start"]))
        )

    incident_mask = np.zeros(len(metrics_df), dtype=bool)
    for inc in INCIDENTS:
        incident_mask |= (
            (ts_index >= pd.Timestamp(inc["start"])) &
            (ts_index <= pd.Timestamp(inc["end"]))
        )

    for idx in chosen_idx:
        row      = metrics_df.iloc[idx]
        ts_base  = row["timestamp"]
        offset_s = int(rng.integers(0, 3600))
        log_ts   = ts_base + pd.Timedelta(seconds=offset_s)

        svc_key  = random.choice(list(_SERVICE_ERROR_AFFINITY.keys()))
        err_pool = list(_SERVICE_ERROR_AFFINITY[svc_key])

        # Staj döneminde quota/scheduler hatası ağırlığı artır
        if staj_mask[idx]:
            err_pool += ["quota_exceeded", "scheduler_error", "resource_exhausted",
                         "quota_pressure", "scheduler_degradation"]

        # Buildup döneminde erken uyarı logları ağırlıklı
        if buildup_mask[idx]:
            err_pool += ["slow_response", "memory_pressure", "connection_pool_warning",
                         "scheduler_degradation", "disk_io_degradation", "cascade_indicator",
                         "slow_response", "connection_pool_warning"]  # 2x ağırlık

        # Incident döneminde kritik hatalar ağırlıklı
        if incident_mask[idx]:
            err_pool += ["resource_exhausted", "timeout", "scheduler_error",
                         "db_connection", "cascade_indicator"]

        err_type = random.choice(err_pool)
        err_rate = row["error_rate_percent"]

        # Severity belirleme
        if err_type in _WARNING_TYPES:
            level = "WARNING"
        elif rng.random() < 0.09 and err_rate > 2.5:
            level = "CRITICAL"
        else:
            level = "ERROR"

        message  = _sample_message(err_type, rng)
        req_id   = f"req-{rng.integers(10**14, 10**15):016x}"
        trace_id = f"trace-{rng.integers(10**10, 10**11):011x}"

        rows.append({
            "timestamp":    log_ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "level":        level,
            "service":      _SERVICES[svc_key],
            "request_id":   req_id,
            "trace_id":     trace_id,
            "error_type":   err_type,
            "message":      message,
            "is_staj":      bool(staj_mask[idx]),
            "is_buildup":   bool(buildup_mask[idx]),
            "is_incident":  bool(incident_mask[idx]),
        })

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def plot_log_analysis(log_feat_df: pd.DataFrame) -> None:
    """Servis bazlı hata dağılımı + erken uyarı trendi grafiklerini kaydeder."""
    n_months = round((pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days / 30)
    date_range_str = (
        f"{pd.Timestamp(START_DATE).strftime('%b %Y')} – "
        f"{pd.Timestamp(END_DATE).strftime('%b %Y')}"
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    fig.suptitle(
        f"OpenStack Log Analizi — {n_months} Aylık Hata ve Erken Uyarı Dağılımı ({date_range_str})",
        fontsize=12,
    )

    # ── Panel 1: Servis bazlı hata stacked area ───────────────────────────────
    ax1 = axes[0]
    ts  = log_feat_df["timestamp"]
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
    for inc in INCIDENTS:
        ax1.axvspan(pd.Timestamp(inc["start"]), pd.Timestamp(inc["end"]),
                    alpha=0.12, color="red", linewidth=0)
    ax1.set_ylabel("Saatlik Hata Sayısı")
    ax1.set_title("Servis Bazlı Hata Dağılımı (mavi=staj, kırmızı=incident)")
    ax1.legend(loc="upper left", fontsize=8, ncol=3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # ── Panel 2: Erken uyarı logları zaman serisi ─────────────────────────────
    ax2 = axes[1]
    early_warning_cols = [
        ("slow_response_count",         "Yavaş Yanıt",        "steelblue"),
        ("memory_pressure_count",       "Bellek Baskısı",     "darkorange"),
        ("connection_pool_warn_count",  "Bağlantı Havuzu",    "green"),
        ("scheduler_degradation_count", "Zamanlayıcı Bozul.", "red"),
        ("disk_io_degradation_count",   "Disk I/O Bozul.",    "purple"),
        ("cascade_indicator_count",     "Cascade Göstergesi", "black"),
    ]
    for col, label, color in early_warning_cols:
        if col in log_feat_df.columns:
            smoothed = log_feat_df[col].rolling(24, min_periods=1).mean()
            ax2.plot(ts, smoothed, label=label, color=color, linewidth=0.8, alpha=0.85)
    for inc in INCIDENTS:
        buildup_start = pd.Timestamp(inc["start"]) - pd.Timedelta(days=inc["buildup_days"])
        ax2.axvspan(buildup_start, pd.Timestamp(inc["start"]),
                    alpha=0.15, color="orange", linewidth=0)
        ax2.axvspan(pd.Timestamp(inc["start"]), pd.Timestamp(inc["end"]),
                    alpha=0.10, color="red", linewidth=0)
    ax2.set_ylabel("24s Hareketli Ort. (adet/saat)")
    ax2.set_title("Erken Uyarı Log Trendleri (turuncu=buildup dönemi, kırmızı=incident)")
    ax2.legend(loc="upper left", fontsize=8, ncol=3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # ── Panel 3: Hata tipi pasta dağılımı ────────────────────────────────────
    ax3 = axes[2]
    err_types = {
        "Timeout":              log_feat_df["timeout_count"].sum(),
        "DB Bağlantı":          log_feat_df["db_connection_count"].sum(),
        "Kaynak Yetersiz":      log_feat_df["resource_exhausted_count"].sum(),
        "Kota Aşımı":           log_feat_df["quota_exceeded_count"].sum(),
        "Kimlik Doğrulama":     log_feat_df["auth_failure_count"].sum(),
        "Ağ Hatası":            log_feat_df["network_error_count"].sum(),
        "Zamanlayıcı Hatası":   log_feat_df["scheduler_error_count"].sum(),
        "I/O Hatası":           log_feat_df["io_error_count"].sum(),
        "⚠ Yavaş Yanıt":        log_feat_df["slow_response_count"].sum(),
        "⚠ Bellek Baskısı":     log_feat_df["memory_pressure_count"].sum(),
        "⚠ Bağlantı Havuzu":    log_feat_df["connection_pool_warn_count"].sum(),
        "⚠ Sched. Bozulma":     log_feat_df["scheduler_degradation_count"].sum(),
        "⚠ Disk I/O Bozulma":   log_feat_df["disk_io_degradation_count"].sum(),
        "⚠ Cascade Göstergesi": log_feat_df["cascade_indicator_count"].sum(),
    }
    colors_bar = (["#d62728"] * 8) + (["#ff7f0e"] * 6)
    ax3.barh(list(err_types.keys()), list(err_types.values()),
             color=colors_bar, alpha=0.78)
    ax3.set_xlabel(f"Toplam Log Sayısı ({n_months} ay) | ⚠ = Erken Uyarı Logları")
    ax3.set_title("Log Tipi Dağılımı (kırmızı=hata, turuncu=erken uyarı)")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "figures", "log_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")
