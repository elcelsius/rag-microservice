# loaders/json_loader.py
# Lê JSON e "achata" (flatten) para texto chave:valor.
def read_json(path: str) -> str:
    import json
    def flatten(obj, prefix=""):
        lines = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                lines += flatten(v, f"{prefix}{k}.")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                lines += flatten(v, f"{prefix}{i}.")
        else:
            s = str(obj).replace("\n", " ").strip()
            key = prefix[:-1] if prefix.endswith(".") else prefix
            lines.append(f"{key}: {s}")
        return lines
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    return "\n".join(flatten(data))
