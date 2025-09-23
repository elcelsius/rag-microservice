# telemetry.py
import json
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

# --- Logger Global ---
# Mantém uma instância única do logger para evitar reconfiguração.
_telemetry_logger = None

def _get_logger(log_dir: str, filename: str) -> logging.Logger | None:
    """
    Configura e retorna um logger singleton para telemetria, de forma segura para concorrência.

    Utiliza o módulo `logging` do Python, que é process-safe e thread-safe.
    Usa um `RotatingFileHandler` para gerenciar o tamanho dos arquivos de log,
    evitando que eles cresçam indefinidamente.
    """
    global _telemetry_logger
    if _telemetry_logger is not None:
        return _telemetry_logger

    try:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, filename)

        # Configura um handler que rotaciona os logs quando atingem 10MB, mantendo 5 backups.
        handler = RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        
        # O formato do log é apenas a mensagem em si, pois a mensagem já será um JSON completo.
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)

        # Cria e configura a instância do logger.
        logger = logging.getLogger('telemetry')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        # Impede que o log seja propagado para o logger raiz, evitando duplicação de saída.
        logger.propagate = False

        _telemetry_logger = logger
        return _telemetry_logger
    except Exception as e:
        # Se a configuração do logger falhar, imprime um erro crítico no console.
        print(f"CRITICAL: Falha ao inicializar o logger de telemetria: {e}", flush=True)
        return None

def log_event(log_dir: str, payload: dict, filename: str = "queries.log"):
    """
    Escreve um evento de telemetria em formato JSON para um arquivo de log rotativo.
    Esta função é segura para ser chamada de múltiplos processos ou threads.
    """
    logger = _get_logger(log_dir, filename)
    if not logger:
        # Falha silenciosamente se o logger não pôde ser inicializado.
        return

    try:
        # Cria uma cópia para não modificar o dicionário original.
        log_payload = dict(payload)
        # Adiciona um timestamp ISO 8601 em UTC.
        log_payload["ts_iso"] = datetime.utcnow().isoformat() + "Z"
        
        # Loga a string JSON.
        logger.info(json.dumps(log_payload, ensure_ascii=False))
    except Exception as e:
        # Se houver um erro durante a serialização do JSON, imprime um aviso.
        print(f"WARN: Falha ao registrar o evento de telemetria: {e}", flush=True)
