# import numpy as np
# from sklearn.ensemble import IsolationForest
# from sklearn.linear_model import Ridge

# class PerTechModels:
#     def __init__(self, contamination=0.02, random_state=42, train_min=500, train_max=3000):
#         self.models = {}
#         self.buffers = {}
#         self.contamination = contamination
#         self.random_state = random_state
#         self.train_min = train_min
#         self.train_max = train_max

#     def _ensure(self, tech):
#         if tech not in self.models:
#             iso = IsolationForest(n_estimators=200, contamination=self.contamination, random_state=self.random_state)
#             ridge = Ridge(alpha=0.8)
#             self.models[tech] = {"iso": iso, "ridge": ridge, "trained": False}
#             self.buffers[tech] = []

#     @staticmethod
#     def features(m):
#         return [m["sinr"], m["throughput_mbps"], m["latency_ms"], m["jitter_ms"], m["loss_pct"]]

#     @staticmethod
#     def target_mos(m):
#         return 1.0 + 4.0 * (
#             0.35*np.tanh((m["sinr"]-10)/15) +
#             0.35*np.tanh((m["throughput_mbps"]-10)/60) -
#             0.18*np.tanh((m["latency_ms"]-40)/40) -
#             0.08*np.tanh((m["jitter_ms"]-5)/10) -
#             0.04*np.tanh((m["loss_pct"]-0.5)/3)
#         )

#     def observe(self, tech, m):
#         self._ensure(tech)
#         buf = self.buffers[tech]
#         if len(buf) < self.train_max:
#             buf.append(m)
#         if (not self.models[tech]["trained"]) and len(buf) >= self.train_min:
#             X = np.array([self.features(x) for x in buf])
#             y = np.array([self.target_mos(x) for x in buf])
#             self.models[tech]["iso"].fit(X)
#             self.models[tech]["ridge"].fit(X, y)
#             self.models[tech]["trained"] = True

#     def infer(self, tech, m):
#         self._ensure(tech)
#         x = np.array(self.features(m)).reshape(1, -1)
#         if self.models[tech]["trained"]:
#             anomaly = (self.models[tech]["iso"].predict(x)[0] == -1)
#             mos = float(self.models[tech]["ridge"].predict(x)[0])
#             training = False
#         else:
#             anomaly = False
#             mos = float(self.target_mos(m))
#             training = True
#         return training, bool(anomaly), mos

#     def reset(self, tech=None):
#         if tech is None:
#             self.models.clear()
#             self.buffers.clear()
#         else:
#             self.models.pop(tech, None); self.buffers.pop(tech, None)
# ai_models.py  (drop-in replacement for your PerTechModels)
# ai-monitor/monitor/ai.py
# ------------------------------------------------------------
# Lightweight per-tech anomaly detector + MOS estimator.
# No external ML deps (NumPy only). Designed for streaming use.
# ------------------------------------------------------------
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np

# Features we consume from the simulator/ingestor
DEFAULT_FEATURES = ("sinr", "latency_ms", "loss_pct", "throughput_mbps")

@dataclass
class _TechModel:
    name: str
    features: Tuple[str, ...]
    contamination: float
    train_min: int
    train_max: int

    # internal buffers / params
    buf: List[np.ndarray] = field(default_factory=list)
    fitted: bool = False
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    z_feature_thr: Optional[np.ndarray] = None
    z_agg_thr: Optional[float] = None

    # observe one sample vector
    def observe_vec(self, x: np.ndarray) -> None:
        if len(self.buf) >= self.train_max:
            # drop oldest to keep a moving window
            self.buf.pop(0)
        self.buf.append(x)
        if not self.fitted and len(self.buf) >= self.train_min:
            self._fit()

    # fit thresholds from current buffer
    def _fit(self) -> None:
        X = np.vstack(self.buf)  # (n, d)
        # robust center/scale — fall back to mean/std if MAD degenerate
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        # guard: if MAD ~ 0, use std
        std = np.where(mad > 1e-6, 1.4826 * mad, np.std(X, axis=0, ddof=1) + 1e-6)

        self.mean = med
        self.std = std

        # Per-feature absolute z distribution
        Z = np.abs((X - self.mean) / self.std)
        p = max(0.0, min(100.0, 100.0 * (1.0 - self.contamination)))
        self.z_feature_thr = np.percentile(Z, p, axis=0)

        # Aggregate z-norm (L2 of per-feature z-scores)
        z_norm = np.linalg.norm(Z, ord=2, axis=1)
        self.z_agg_thr = float(np.percentile(z_norm, p))

        self.fitted = True

    # score a vector: return (training, is_anomaly)
    def score_vec(self, x: np.ndarray) -> Tuple[bool, bool]:
        if not self.fitted or self.mean is None or self.std is None:
            return True, False
        z = np.abs((x - self.mean) / self.std)
        # anomaly if any feature z > its learned percentile OR agg z-norm > learned percentile
        feat_flag = bool(np.any(z > self.z_feature_thr))
        agg_flag = bool(np.linalg.norm(z, ord=2) > self.z_agg_thr)
        return False, (feat_flag or agg_flag)

def _extract_features(m: Dict, features: Tuple[str, ...]) -> np.ndarray:
    # accept 'sinr' or 'sinr_db'
    x: List[float] = []
    for f in features:
        if f == "sinr":
            v = m.get("sinr", m.get("sinr_db", 0.0))
        else:
            v = m.get(f, 0.0)
        try:
            x.append(float(v))
        except Exception:
            x.append(0.0)
    return np.asarray(x, dtype=np.float64)

def _sigmoid(z: float) -> float:
    # numerically safe sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

def estimate_mos(m: Dict) -> float:
    """
    Simple QoE proxy in [1,5].
    Positive contributors: throughput, SINR
    Negative contributors: latency, loss
    Tuned for "relative" behavior, not absolute telephony MOS.
    """
    thr = float(m.get("throughput_mbps", 0.0))
    sinr = float(m.get("sinr", m.get("sinr_db", 0.0)))
    lat = float(m.get("latency_ms", 0.0))
    los = float(m.get("loss_pct", 0.0))

    # Normalize each into [0,1] using smooth sigmoids
    # Throughput: log-scale, 0..1 around 5→1000 Mbps
    thr_norm = _sigmoid((math.log10(max(0.01, thr)) - math.log10(5.0)) / (math.log10(1000.0) - math.log10(5.0) + 1e-6) * 8 - 4)
    # SINR: -10..30 dB roughly
    sinr_norm = _sigmoid((sinr - 5.0) / 6.0)
    # Latency penalty: 10..150 ms
    lat_pen = _sigmoid((lat - 50.0) / 10.0)
    # Loss penalty: 0..3 %
    loss_pen = _sigmoid((los - 0.5) / 0.3)

    # Weighted blend → [0,1], then map to [1,5]
    score = 0.40 * thr_norm + 0.35 * sinr_norm + 0.15 * (1.0 - lat_pen) + 0.10 * (1.0 - loss_pen)
    mos = 1.0 + 4.0 * max(0.0, min(1.0, score))
    return float(max(1.0, min(5.0, mos)))

class PerTechModels:
    """
    Maintains a tiny anomaly model per technology.
    Usage:
        models = PerTechModels()
        models.observe("5g", metric_dict)
        training, anomaly, mos = models.infer("5g", metric_dict)
    """
    def __init__(self,
                 features: Iterable[str] = DEFAULT_FEATURES,
                 contamination: float = 0.01,
                 train_min: int = 800,
                 train_max: int = 4000) -> None:
        self.features = tuple(features)
        self.contamination = float(contamination)
        self.train_min = int(train_min)
        self.train_max = int(train_max)
        self.models: Dict[str, _TechModel] = {}

    def _get(self, tech: str) -> _TechModel:
        t = (tech or "lte").lower()
        if t not in self.models:
            self.models[t] = _TechModel(
                name=t,
                features=self.features,
                contamination=self.contamination,
                train_min=self.train_min,
                train_max=self.train_max,
            )
        return self.models[t]

    def observe(self, tech: str, m: Dict) -> None:
        """Feed a raw metric dict (does not return anything)."""
        model = self._get(tech)
        x = _extract_features(m, model.features)
        model.observe_vec(x)

    def infer(self, tech: str, m: Dict) -> Tuple[bool, bool, float]:
        """
        Returns: (training: bool, anomaly: bool, mos: float)
        - training=True → model is still learning baseline; ignore anomaly.
        """
        model = self._get(tech)
        x = _extract_features(m, model.features)
        training, anomaly = model.score_vec(x)
        mos = estimate_mos(m)
        return training, anomaly, mos

    def reset(self, tech: Optional[str] = None) -> None:
        """Reset one tech or all."""
        if tech:
            t = tech.lower()
            if t in self.models:
                del self.models[t]
        else:
            self.models.clear()

    # Optional introspection helpers
    def debug_snapshot(self) -> Dict[str, Dict]:
        snap: Dict[str, Dict] = {}
        for t, mdl in self.models.items():
            snap[t] = {
                "fitted": mdl.fitted,
                "count": len(mdl.buf),
                "mean": None if mdl.mean is None else mdl.mean.tolist(),
                "std": None if mdl.std is None else mdl.std.tolist(),
                "z_feature_thr": None if mdl.z_feature_thr is None else mdl.z_feature_thr.tolist(),
                "z_agg_thr": mdl.z_agg_thr,
                "features": mdl.features,
            }
        return snap
