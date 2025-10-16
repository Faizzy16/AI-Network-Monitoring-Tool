import asyncio, math, time, os
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np

# -------------------------
# Supported technologies
# -------------------------
TECHS = ("3g", "4g", "lte", "5g")

# -------------------------
# Tech profile
# -------------------------
@dataclass
class TechProfile:
    name: str
    bandwidth_mhz: float
    p_tx_dbm: float
    noise_fig_db: float
    inter_cell_i_dbm: float
    sinr_db: Tuple[float, float]
    base_latency_ms: Tuple[float, float]
    base_loss_pct: Tuple[float, float]
    base_thr_mbps: Tuple[float, float]
    cqi_offset: float = 10.0

DEFAULT_PROFILES: Dict[str, TechProfile] = {
    "3g":  TechProfile("3g",   5, 43, 7, -95,
                       sinr_db=(5, 6),   base_latency_ms=(150, 25),
                       base_loss_pct=(0.8, 0.8),  base_thr_mbps=(3, 1.5),  cqi_offset=6),
    "4g":  TechProfile("4g",  20, 46, 6, -96,
                       sinr_db=(10, 8),  base_latency_ms=(50, 10),
                       base_loss_pct=(0.3, 0.4),  base_thr_mbps=(25, 10), cqi_offset=8),
    "lte": TechProfile("lte", 20, 46, 5, -97,
                       sinr_db=(18, 6),  base_latency_ms=(35, 8),
                       base_loss_pct=(0.15,0.2),  base_thr_mbps=(120, 40), cqi_offset=10),
    "5g":  TechProfile("5g", 100, 49, 4, -98,
                       sinr_db=(28, 5),  base_latency_ms=(15, 5),
                       base_loss_pct=(0.05,0.08), base_thr_mbps=(500,200), cqi_offset=12),
}

def env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except Exception:
        return default

def load_profile(name: str) -> TechProfile:
    name = name.lower()
    if name not in DEFAULT_PROFILES:
        raise ValueError("invalid tech")
    p = DEFAULT_PROFILES[name]
    # allow ENV overrides
    return TechProfile(
        name=p.name,
        bandwidth_mhz=env_float(f"{name.upper()}_BW_MHZ", p.bandwidth_mhz),
        p_tx_dbm=env_float(f"{name.upper()}_PTX_DBM", p.p_tx_dbm),
        noise_fig_db=env_float(f"{name.upper()}_NF_DB", p.noise_fig_db),
        inter_cell_i_dbm=env_float(f"{name.upper()}_I_DBM", p.inter_cell_i_dbm),
        sinr_db=p.sinr_db,
        base_latency_ms=p.base_latency_ms,
        base_loss_pct=p.base_loss_pct,
        base_thr_mbps=p.base_thr_mbps,
        cqi_offset=p.cqi_offset,
    )

# -------------------------
# Topology entities
# -------------------------
class UE:
    def __init__(self, ue_id, x, y):
        self.id, self.x, self.y = ue_id, x, y

class Cell:
    def __init__(self, cell_id, x, y, p_tx_dbm: float = None):
        self.id, self.x, self.y = cell_id, x, y
        self.p_tx_dbm = p_tx_dbm  # if None, take from TechProfile

# -------------------------
# Dynamic simulator state
# -------------------------
class TechState:
    def __init__(self, tech: str = "5g"):
        if tech not in TECHS:
            raise ValueError("invalid tech")
        self.tech = tech
        self.profile = load_profile(tech)

        # Fault toggles
        self.inject_drop = False
        self.inject_rf_degrade = False
        self.inject_congestion = False

        # Average PRB load (0..1)
        self.load_mean = float(os.getenv("LOAD_MEAN", "0.35"))

        # Simulator tick (ms). Default 15000 (15s). Change via ENV or REST.
        self.step_ms = int(os.getenv("SIM_STEP_MS", "15000"))

    def set_tech(self, t: str):
        t = t.lower()
        if t not in TECHS:
            raise ValueError("invalid tech")
        self.tech = t
        self.profile = load_profile(t)

    def clear_faults(self):
        self.inject_drop = self.inject_rf_degrade = self.inject_congestion = False

# -------------------------
# Radio helpers
# -------------------------
K_B = 1.380649e-23  # Boltzmann
T_K = 290.0         # room temp

def noise_floor_dbm(bw_mhz: float, noise_fig_db: float) -> float:
    """kTB (W/Hz) -> W -> dBm, add noise figure."""
    bw_hz = bw_mhz * 1e6
    n_w = K_B * T_K * bw_hz
    n_dbm = 10*np.log10(n_w) + 30.0
    return float(n_dbm + noise_fig_db)

def rsrp_from_rssi_dbm(rssi_dbm: float, n_prb: int) -> float:
    """Very rough mapping using PRB count as proxy."""
    return float(rssi_dbm - 10*np.log10(max(1, n_prb*12)) - 3.0)

def rsrq_db(rsrp_dbm: float, rssi_dbm: float) -> float:
    return float(rsrp_dbm - rssi_dbm + 10*np.log10(12))

def cqi_from_sinr(sinr_db: float, offset: float) -> int:
    return int(np.clip(np.round((sinr_db + offset)/2.0), 0, 15))

# -------------------------
# Metric generation
# -------------------------
# --- keep existing imports and code ---

def gen_metrics_for_ue(ue: UE, cells: List[Cell], state: TechState):
    prof = state.profile

    # Serving cell = nearest
    dists = [(c, math.hypot(ue.x - c.x, ue.y - c.y)) for c in cells]
    serving, dist_m = min(dists, key=lambda z: z[1])

    # Tx power
    p_tx_dbm = serving.p_tx_dbm if serving.p_tx_dbm is not None else prof.p_tx_dbm

    # Simple path loss (bandwidth used as proxy for freq)
    A = 32.4 + 20*np.log10(max(1.0, prof.bandwidth_mhz))
    B = 31.7
    shadow = np.random.normal(0, 4.0)
    pl_db = A + B*np.log10(max(dist_m, 1.0)) + shadow  # <-- PATH LOSS (dB)

    # Fading and noise
    small_fade_db = np.random.normal(0, 2.0)
    rssi_dbm = p_tx_dbm - pl_db + small_fade_db       # <-- RSSI (dBm)
    noise_dbm = noise_floor_dbm(prof.bandwidth_mhz, prof.noise_fig_db)
    i_dbm = prof.inter_cell_i_dbm                     # <-- Interference average (dBm)

    # SINR
    p_sig = 10**(rssi_dbm/10)
    p_i   = 10**(i_dbm/10)
    p_n   = 10**(noise_dbm/10)
    sinr_db = float(10*np.log10(p_sig / (p_i + p_n)))

    # Fault injections
    rf_delta_sinr = 0.0
    rf_delta_rssi = 0.0
    if state.inject_rf_degrade:
        rf_delta_sinr = np.random.uniform(6, 12)
        rf_delta_rssi = np.random.uniform(4, 8)
        sinr_db -= rf_delta_sinr
        rssi_dbm -= rf_delta_rssi

    # PRB utilization
    prb_util = float(np.clip(np.random.normal(state.load_mean, 0.08), 0.0, 0.98))
    n_prb = int(np.clip(np.round(prof.bandwidth_mhz / 0.18), 6, 273))  # ~180 kHz/PRB

    # Base KPIs
    base_thr = max(0.0, np.random.normal(prof.base_thr_mbps[0], prof.base_thr_mbps[1]))
    base_lat = max(3.0,  np.random.normal(prof.base_latency_ms[0], prof.base_latency_ms[1]))
    base_loss= max(0.0, np.random.normal(prof.base_loss_pct[0],  prof.base_loss_pct[1]))

    # Map SINR & PRB load to throughput
    sinr_gain = max(0.1, (sinr_db + 5.0) / 25.0)
    thr_mbps = base_thr * sinr_gain * (1.0 - prb_util*0.6)

    latency_ms = base_lat + (prb_util**2) * 30.0
    jitter_ms  = max(0.5, 0.08*latency_ms + np.random.normal(0, 0.6))
    loss_pct   = base_loss + prb_util*1.2

    cong_lat_add = 0.0
    cong_jit_add = 0.0
    cong_loss_add= 0.0
    cong_thr_mul = 1.0
    drop_thr_mul = 1.0
    drop_lat_add = 0.0
    drop_loss_add= 0.0

    if state.inject_congestion:
        cong_lat_add = np.random.uniform(40, 120)
        cong_jit_add = np.random.uniform(5, 15)
        cong_thr_mul = np.random.uniform(0.3, 0.7)
        cong_loss_add= abs(np.random.normal(1.0, 0.7))
        latency_ms += cong_lat_add
        jitter_ms  += cong_jit_add
        thr_mbps   *= cong_thr_mul
        loss_pct   += cong_loss_add

    if state.inject_drop:
        drop_thr_mul = np.random.uniform(0.0, 0.1)
        drop_lat_add = np.random.uniform(100, 300)
        drop_loss_add= np.random.uniform(5, 15)
        thr_mbps   *= drop_thr_mul
        latency_ms += drop_lat_add
        loss_pct   += drop_loss_add

    # RSRP/RSRQ & CQI
    rsrp_dbm = rsrp_from_rssi_dbm(rssi_dbm, n_prb)
    rsrq     = rsrq_db(rsrp_dbm, rssi_dbm)
    cqi      = cqi_from_sinr(sinr_db, prof.cqi_offset)

    # ---- NEW: explain pack ----
    explain = {
        "radio_chain": {
            "p_tx_dbm": float(p_tx_dbm),
            "path_loss_db": float(pl_db),
            "fading_db": float(small_fade_db),
            "rssi_dbm": float(rssi_dbm),
            "noise_dbm": float(noise_dbm),
            "interference_dbm": float(i_dbm),
            "sinr_db": float(sinr_db),
            "rf_degrade_applied": state.inject_rf_degrade,
            "rf_delta_sinr_db": float(rf_delta_sinr),
            "rf_delta_rssi_db": float(rf_delta_rssi),
        },
        "scheduler_load": {
            "prb_util": float(prb_util),
            "n_prb": int(n_prb)
        },
        "kpi_assembly": {
            "base_thr_mbps": float(base_thr),
            "base_latency_ms": float(base_lat),
            "base_loss_pct": float(base_loss),
            "sinr_gain": float(sinr_gain),
            "throughput_mbps": float(max(0.0, thr_mbps)),
            "latency_ms": float(latency_ms),
            "jitter_ms": float(jitter_ms),
            "loss_pct": float(max(0.0, loss_pct)),
            "congestion": {
                "enabled": state.inject_congestion,
                "latency_add_ms": float(cong_lat_add),
                "jitter_add_ms": float(cong_jit_add),
                "loss_add_pct": float(cong_loss_add),
                "thr_multiplier": float(cong_thr_mul)
            },
            "drop": {
                "enabled": state.inject_drop,
                "latency_add_ms": float(drop_lat_add),
                "loss_add_pct": float(drop_loss_add),
                "thr_multiplier": float(drop_thr_mul)
            }
        }
    }

    return {
        "ue": ue.id,
        "cell": serving.id,
        "tech": state.tech,
        "bandwidth_mhz": float(prof.bandwidth_mhz),
        "p_tx_dbm": float(p_tx_dbm),
        "noise_fig_db": float(prof.noise_fig_db),
        "noise_dbm": float(noise_dbm),
        "rssi_dbm": float(rssi_dbm),
        "rsrp_dbm": float(rsrp_dbm),
        "rsrq_db": float(rsrq),
        "sinr_db": float(sinr_db),
        "sinr": float(sinr_db),
        "cqi": int(cqi),
        "prb_util": float(prb_util),
        "throughput_mbps": float(max(0.0, thr_mbps)),
        "latency_ms": float(latency_ms),
        "jitter_ms": float(jitter_ms),
        "loss_pct": float(max(0.0, loss_pct)),
        "explain": explain,  # <-- NEW
    }
# -------------------------
# Simulator: emits once per UE per tick
# Tick duration is taken from state.step_ms (default 15s)
# -------------------------
async def simulator(metric_queue: asyncio.Queue, cells, ues, state: TechState):
    while True:
        t = time.time()
        for ue in ues:
            m = gen_metrics_for_ue(ue, cells, state)
            m["ts"] = t
            await metric_queue.put(m)
        # dynamic tick
        await asyncio.sleep(max(0.05, state.step_ms / 1000.0))
