# loaders/csv_loader.py
# Lê CSV e converte para texto tabular simples.
def read_csv(path: str) -> str:
    try:
        import pandas as pd
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        lines = []
        lines.append("| " + " | ".join(df.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
        for _, row in df.iterrows():
            vals = [str(row[c]).replace("\n", " ").strip() for c in df.columns]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)
    except Exception:
        import csv
        out = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                out.append(" | ".join([str(x).replace("\n", " ").strip() for x in row]))
        return "\n".join(out)
