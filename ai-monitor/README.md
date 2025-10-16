# AI Monitor (3G/4G/LTE/5G)

## Run
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# If your simulator is on a different host/port, set:
# $env:SIM_WS="ws://127.0.0.1:9001/metrics"
python -m uvicorn server:app --reload --port 8001
```
Open http://127.0.0.1:8001 — charts will update live.  
Click **Reset AI Models** after switching tech to retrain per-tech baseline.
