# tools/check_data_and_loaders.py
# Uso: python tools/check_data_and_loaders.py ./data ./loaders "txt,md,pdf,docx,csv,json"
import sys
from pathlib import Path

def main():
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "./data")
    loaders_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "./loaders")
    exts = set((",".join(sys.argv[3:]) if len(sys.argv) > 3 else "txt,md,pdf,docx").split(","))
    exts = {"."+e.strip().lower() for e in exts if e.strip()}
    print(f"[check] data_dir={data_dir} loaders_dir={loaders_dir} exts={sorted(exts)}")

    counts = {e:0 for e in exts}
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in counts:
            counts[p.suffix.lower()] += 1
    print("[check] arquivos por extensão:")
    for e, n in sorted(counts.items()):
        print(f"  {e}: {n}")

    found = []
    if loaders_dir.exists():
        for py in loaders_dir.glob("*.py"):
            txt = py.read_text(encoding="utf-8", errors="ignore")
            for e in exts:
                if f"def read_{e[1:]}(" in txt or "def load(" in txt:
                    found.append((py.name, e))
    if found:
        print("[check] loaders detectados:")
        for name, e in found:
            print(f"  {name} -> {e}")
    else:
        print("[check] nenhum loader detectado para as extensões informadas.")

if __name__ == "__main__":
    main()
