# Multi-Network Simulator (3G/4G/LTE/5G)
Run:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn server:app --reload --port 9001
```
Open http://127.0.0.1:9001 to switch tech and inject faults.  
WebSocket: `ws://127.0.0.1:9001/metrics`
