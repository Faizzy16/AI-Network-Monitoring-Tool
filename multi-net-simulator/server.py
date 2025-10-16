
import asyncio, json, random, argparse, os, statistics as stats
from typing import List
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from sim.core import UE, Cell, TechState, simulator, TECHS

app = FastAPI(title="Multi-Network Simulator")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------------
# Globals
# ----------------------------
metric_queue: asyncio.Queue = asyncio.Queue(maxsize=20000)
clients: List[WebSocket] = []
state = TechState(os.getenv("TECH", "5g"))
bg_started = False

# Topology (UE count can be overridden via CLI or env)
cells = [Cell("cell-A", 0, 0), Cell("cell-B", 500, 0), Cell("cell-C", 0, 500)]
ues: List[UE] = []

# A safe, read-only sliding window for console printer (do NOT drain the queue)
recent_window: deque = deque(maxlen=2000)

# ----------------------------
# Helpers
# ----------------------------
def build_ues(n_ues: int):
    """(Re)build the UE list deterministically."""
    global ues
    rng = random.Random(42)
    ues = [UE(f"ue-{i}", rng.uniform(-200, 600), rng.uniform(-200, 600)) for i in range(n_ues)]

async def broadcaster():
    """
    Fan out metrics to WS clients.
    IMPORTANT: Only this task consumes metric_queue.
    Console printer reads from 'recent_window' (filled here) to avoid starving clients.
    """
    while True:
        m = await metric_queue.get()

        # compatibility shim: emit both 'sinr' and 'sinr_db' so older monitors work
        if "sinr" not in m and "sinr_db" in m:
            m = {**m, "sinr": m["sinr_db"]}

        # add to console window (copy lightweight fields)
        recent_window.append(m)

        out = {"type": "metric", "payload": m}
        dead = []
        for ws in list(clients):
            try:
                await ws.send_text(json.dumps(out))
            except Exception:
                dead.append(ws)
        for d in dead:
            try:
                clients.remove(d)
            except ValueError:
                pass

        metric_queue.task_done()

async def console_printer():
    """
    Pretty console dashboard: prints 1 Hz aggregate using 'recent_window'.
    DO NOT read metric_queue here (that would starve broadcaster/UI).
    """
    while True:
        await asyncio.sleep(1.0)
        if not recent_window:
            continue

        # use only current-tech frames from the sliding window
        cur = [x for x in list(recent_window) if x.get("tech") == state.tech]
        if not cur:
            cur = list(recent_window)

        def series(key): return [x.get(key) for x in cur if key in x]

        def fmt(x):
            try:
                return f"{x:.2f}"
            except Exception:
                return str(x)

        thr = series("throughput_mbps")
        lat = series("latency_ms")
        jit = series("jitter_ms")
        loss = series("loss_pct")
        sinr = series("sinr") or series("sinr_db")
        rssi = series("rssi_dbm")
        rsrp = series("rsrp_dbm")
        rsrq = series("rsrq_db")
        cqi  = series("cqi")
        prb  = series("prb_util")
        noise = series("noise_dbm")

        def sstat(v):
            if not v:
                return "-", "-"
            try:
                mean_v = stats.mean(v)
            except stats.StatisticsError:
                mean_v = v[-1]
            if len(v) >= 20:
                try:
                    p95 = stats.quantiles(v, n=20)[-1]
                except Exception:
                    p95 = max(v)
            else:
                p95 = max(v)
            return fmt(mean_v), fmt(p95)

        m_thr, p95_thr = sstat(thr)
        m_lat, p95_lat = sstat(lat)
        m_jit, p95_jit = sstat(jit)
        m_loss, p95_loss = sstat(loss)
        m_sinr, p95_sinr = sstat(sinr)
        m_rssi, _ = sstat(rssi)
        m_rsrp, _ = sstat(rsrp)
        m_rsrq, _ = sstat(rsrq)
        m_prb, _ = sstat(prb)
        m_noise, _ = sstat(noise)
        m_cqi = fmt(stats.mean(cqi)) if cqi else "-"

        sample = cur[-1]

        print("\n" + "="*88)
        print(f" Tech={state.tech.upper()}  UEs={len(ues)}  LoadMean={state.load_mean:.2f}")
        print(f" Radio:  Noise={m_noise} dBm  RSSI={m_rssi} dBm  RSRP={m_rsrp} dBm  RSRQ={m_rsrq} dB  SINR={m_sinr} dB  CQI≈{m_cqi}  PRB={m_prb}")
        print(f" Core :  Thr={m_thr} Mbps (p95 {p95_thr})  Lat={m_lat} ms (p95 {p95_lat})  Jit={m_jit} ms (p95 {p95_jit})  Loss={m_loss}% (p95 {p95_loss})")
        print("-"*88)
        print(" Sample UE:",
              f"ue={sample.get('ue')} cell={sample.get('cell')}",
              f"rssi={fmt(sample.get('rssi_dbm'))}dBm",
              f"rsrp={fmt(sample.get('rsrp_dbm'))}dBm",
              f"rsrq={fmt(sample.get('rsrq_db'))}dB",
              f"sinr={fmt(sample.get('sinr', sample.get('sinr_db')))}dB",
              f"cqi={sample.get('cqi')}",
              f"prb={fmt(sample.get('prb_util'))}",
              f"thr={fmt(sample.get('throughput_mbps'))}Mbps",
              f"lat={fmt(sample.get('latency_ms'))}ms",
              f"jit={fmt(sample.get('jitter_ms'))}ms",
              f"loss={fmt(sample.get('loss_pct'))}%"
        )

# ----------------------------
# FastAPI endpoints
# ----------------------------
@app.on_event("startup")
async def on_startup():
    global bg_started, ues
    if not bg_started:
        # Ensure UEs exist when running via `uvicorn server:app`
        if not ues:
            build_ues(int(os.getenv("UE_COUNT", "50")))
        asyncio.create_task(simulator(metric_queue, cells, ues, state))
        asyncio.create_task(broadcaster())
        if os.getenv("PRINT_CONSOLE", "1") == "1":
            asyncio.create_task(console_printer())
        bg_started = True

@app.get("/")
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/topology")
async def get_topology():
    return {"cells": [c.id for c in cells], "ues": [u.id for u in ues], "tech": state.tech}

@app.post("/tech/{name}")
async def set_tech(name: str):
    try:
        name = name.lower()
        if name not in TECHS:
            raise ValueError("invalid tech")
        state.set_tech(name)
        state.clear_faults()
        return JSONResponse({"ok": True, "tech": name})
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

@app.post("/fault/{kind}")
async def set_fault(kind: str):
    k = kind.lower()
    if k == "none":
        state.clear_faults()
    elif k == "rf":
        state.clear_faults(); state.inject_rf_degrade = True
    elif k == "congestion":
        state.clear_faults(); state.inject_congestion = True
    elif k == "drop":
        state.inject_drop = True
        async def clear_drop():
            await asyncio.sleep(3.0)
            state.inject_drop = False
        asyncio.create_task(clear_drop())
    else:
        return JSONResponse({"ok": False, "error": "invalid fault (none|rf|congestion|drop)"}, status_code=400)
    return {"ok": True, "fault": k}

@app.websocket("/metrics")
async def ws_metrics(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        while True:
            _ = await ws.receive_text()  # pings
    except WebSocketDisconnect:
        try: clients.remove(ws)
        except ValueError: pass
    except Exception:
        try: clients.remove(ws)
        except ValueError: pass

# ----------------------------
# CLI runner (optional)
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Multi-Network Simulator with Console View")
    parser.add_argument("--tech", choices=TECHS, default=os.getenv("TECH", "5g"))
    parser.add_argument("--ues", type=int, default=int(os.getenv("UE_COUNT", "50")))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "9001")))
    parser.add_argument("--no-console", action="store_true", help="Disable console dashboard")
    args = parser.parse_args()

    os.environ["PRINT_CONSOLE"] = "0" if args.no_console else "1"
    state = TechState(args.tech)
    build_ues(args.ues)

    uvicorn.run("server:app", host="0.0.0.0", port=args.port, reload=True)
