# ai-monitor/server.py
# ------------------------------------------------------------
# AI-driven monitor that:
#  - connects to the simulator WebSocket (SIM_WS)
#  - aggregates ALL samples into ONE global 15s rollup
#  - runs per-tech anomaly detection + MOS via monitor/ai.py
#  - classifies global status (Healthy / Warning / Degraded / Critical)
#  - streams only the rollup to browser clients at /ws
# ------------------------------------------------------------
import os, asyncio, json, time
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import websockets

from monitor.ai import PerTechModels

# ------------------------
# App + static
# ------------------------
app = FastAPI(title="AI Monitor (Global 15s)")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------------
# Config
# ------------------------
SIM_WS = os.getenv("SIM_WS", "ws://127.0.0.1:9001/metrics")
ROLLUP_SECS = int(os.getenv("ROLLUP_SECS", "15"))

# ------------------------
# Globals
# ------------------------
clients: List[WebSocket] = []
models = PerTechModels(contamination=0.01, train_min=800, train_max=4000)

TECH_ORDER = {"3g": 0, "4g": 1, "lte": 2, "5g": 3}
prev_dominant_tech = "5g"  # assume healthy starting point

# global rollup bins by bucket
# each bin accumulates sums and counts across all incoming samples for that 15s window
global_bins = defaultdict(lambda: {
    "n": 0, "ts": 0.0,
    "sinr_sum": 0.0, "lat_sum": 0.0, "loss_sum": 0.0, "thr_sum": 0.0, "mos_sum": 0.0,
    "anomaly_count": 0, "rule_count": 0,
    "tech_counts": defaultdict(int),
    # track training frames to optionally show a training banner while learning baseline
    "training_count": 0,
})

# ------------------------
# Health / rule checks
# ------------------------
def hard_rule(tech: str, m: Dict[str, Any]) -> bool:
    """
    Simple threshold guardrails per technology.
    (We made 5G slightly lenient as requested.)
    """
    t = (tech or "").lower()
    lat = float(m.get("latency_ms", 0.0))
    loss = float(m.get("loss_pct", 0.0))
    sinr = float(m.get("sinr", m.get("sinr_db", 0.0)))

    if t == "5g":
        return (lat > 120.0) or (loss > 3.0) or (sinr < -6.0)
    elif t in ("lte", "4g"):
        return (lat > 120.0) or (loss > 2.5) or (sinr < -6.0)
    else:  # 3g
        return (lat > 220.0) or (loss > 3.0) or (sinr < -8.0)

def classify_global(avg: Dict[str, float], dominant_tech: str, prev_dom: str) -> Tuple[str, str, str, bool, bool]:
    """
    Returns: (status, severity_icon, summary_text, alert_bool, training_bool)
    """
    rank_now  = TECH_ORDER.get(dominant_tech, 0)
    rank_prev = TECH_ORDER.get(prev_dom, 0)
    downgraded = rank_prev > rank_now

    rule_hit = hard_rule(dominant_tech, {
        "latency_ms": avg["latency_ms"],
        "loss_pct":   avg["loss_pct"],
        "sinr":       avg["sinr"]
    })
    anomaly_rate = avg["anomaly_rate"]
    training = bool(avg.get("training_rate", 0.0) > 0.2)  # banner while collecting baseline

    # Priority: downgrade → critical if KPIs also poor
    if downgraded and (rule_hit or (rank_prev - rank_now) >= 2):
        status, severity = "Critical: Technology downgraded and KPIs impacted", "🔴 Critical"
    elif downgraded:
        status, severity = "Degraded: Technology downgraded", "🟠 Degraded"
    elif rule_hit:
        status, severity = "Degraded: Threshold breach", "🟠 Degraded"
    else:
        # LENIENT 5G: default Healthy; warn only on strong anomaly
        if dominant_tech == "5g":
            if anomaly_rate >= 0.15:
                status, severity = "Fair (monitor)", "🟡 Warning"
            else:
                status, severity = "Healthy", "🟢 Healthy"
        else:
            # non-5G: if not excellent MOS/anomaly, call it Degraded
            if avg["mos"] >= 4.0 and anomaly_rate < 0.02:
                status, severity = "Healthy", "🟢 Healthy"
            else:
                status, severity = "Degraded", "🟠 Degraded"

    summary = (
        f"Status: {status}\n"
        f"DominantTech={dominant_tech.upper()} | "
        f"SINR={avg['sinr']:.1f} dB, "
        f"Latency={avg['latency_ms']:.1f} ms, "
        f"Loss={avg['loss_pct']:.2f}%, "
        f"Throughput={avg['throughput_mbps']:.1f} Mbps, "
        f"MOS={avg['mos']:.2f} | "
        f"AnomalyRate={anomaly_rate*100:.1f}%"
    )
    # Only raise alert if downgrade/rule/strong anomaly
    alert = downgraded or rule_hit or (dominant_tech != "5g" and anomaly_rate >= 0.15)
    return status, severity, summary, alert, training

# ------------------------
# HTTP routes
# ------------------------
@app.get("/")
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/reset")
async def reset(tech: str | None = None):
    models.reset(tech)
    return {"ok": True, "tech": tech or "all"}

@app.get("/ai/debug")
async def ai_debug():
    """Optional: inspect learned baselines (means/std/thresholds) per tech."""
    return models.debug_snapshot()

# ------------------------
# WebSocket for browser clients
# ------------------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            _ = await ws.receive_text()  # keepalive pings
    except WebSocketDisconnect:
        try: clients.remove(ws)
        except ValueError: pass
    except Exception:
        try: clients.remove(ws)
        except ValueError: pass

# ------------------------
# Ingest from simulator, aggregate, broadcast
# ------------------------
async def ingest_from_simulator():
    global prev_dominant_tech
    while True:
        try:
            print(f"[AI] Connecting to simulator at {SIM_WS} ...")
            async with websockets.connect(SIM_WS, ping_interval=None) as ws:
                print("[AI] Connected! Streaming…")
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        if msg.get("type") != "metric":
                            continue
                        m = msg["payload"]

                        # normalize fields
                        if "sinr" not in m and "sinr_db" in m:
                            m["sinr"] = m["sinr_db"]
                        tech = (m.get("tech") or "lte").lower()

                        # per-tech AI
                        models.observe(tech, m)
                        training, anomaly, mos = models.infer(tech, m)

                        # put into 15s global bucket
                        bucket = int(m["ts"] // ROLLUP_SECS)
                        b = global_bins[bucket]
                        b["n"] += 1
                        b["ts"] = max(b["ts"], m["ts"])
                        b["sinr_sum"] += float(m.get("sinr", 0.0))
                        b["lat_sum"]  += float(m.get("latency_ms", 0.0))
                        b["loss_sum"] += float(m.get("loss_pct", 0.0))
                        b["thr_sum"]  += float(m.get("throughput_mbps", 0.0))
                        b["mos_sum"]  += float(mos)
                        b["tech_counts"][tech] += 1
                        if anomaly:  b["anomaly_count"] += 1
                        if training: b["training_count"] += 1
                        if hard_rule(tech, m): b["rule_count"] += 1

                        # flush finished buckets (anything older than "now" bucket)
                        now_bucket = int(time.time() // ROLLUP_SECS)
                        old_keys = [k for k in list(global_bins.keys()) if k < now_bucket]
                        for k in old_keys:
                            bb = global_bins.pop(k)
                            n = max(1, bb["n"])
                            # dominant technology by count
                            if bb["tech_counts"]:
                                dominant = max(bb["tech_counts"].items(), key=lambda kv: kv[1])[0]
                            else:
                                dominant = "lte"

                            avg = {
                                "ts": bb["ts"],
                                "sinr":            bb["sinr_sum"] / n,
                                "latency_ms":      bb["lat_sum"] / n,
                                "loss_pct":        bb["loss_sum"] / n,
                                "throughput_mbps": bb["thr_sum"] / n,
                                "mos":             bb["mos_sum"] / n,
                                "anomaly_rate":    bb["anomaly_count"] / n,
                                "rule_rate":       bb["rule_count"] / n,
                                "training_rate":   bb["training_count"] / n,
                            }

                            status, severity, summary, alert, training_flag = classify_global(avg, dominant, prev_dominant_tech)
                            prev_dominant_tech = dominant

                            rollup = {
                                "type": "rollup",
                                "payload": {
                                    **avg,
                                    "tech": dominant,
                                    "status": status,
                                    "severity": severity,
                                    "summary": summary,
                                    "alert": alert,
                                    "training": training_flag
                                }
                            }

                            # broadcast to all connected browsers
                            dead = []
                            for u in list(clients):
                                try:
                                    await u.send_text(json.dumps(rollup))
                                except Exception:
                                    dead.append(u)
                            for d in dead:
                                try: clients.remove(d)
                                except ValueError: pass

                    except Exception as e:
                        print("[AI] frame error:", repr(e))
                        continue
        except Exception as e:
            print(f"[AI] Can't reach simulator ({e}); retrying in 2s…")
            await asyncio.sleep(2.0)

# ------------------------
# Startup
# ------------------------
@app.on_event("startup")
async def on_startup():
    asyncio.create_task(ingest_from_simulator())

# ------------------------
# Dev entry
# ------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8001")), reload=False)
