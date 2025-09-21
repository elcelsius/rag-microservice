import json
import os
from datetime import datetime

def log_event(log_dir: str, payload: dict, filename: str = "queries.log"):
    """
    Escreve um JSON por linha com telemetria da consulta.
    Cria diretório automaticamente. Tolerante a falhas.
    """
    try:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, filename)
        payload = dict(payload)
        payload["ts_iso"] = datetime.utcnow().isoformat() + "Z"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
