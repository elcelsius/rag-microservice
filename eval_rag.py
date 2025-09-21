#!/usr/bin/env python
import csv
import json
import math
import os
import re
import sys
from urllib.request import Request, urlopen

API_URL = os.getenv("API_URL", "http://localhost:5000/query")

def query_api(q):
    req = Request(API_URL, data=json.dumps({"question": q, "debug": True}).encode("utf-8"), headers={"Content-Type":"application/json"})
    with urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))

def dcg(rels):
    return sum((rel / math.log2(i+2)) for i, rel in enumerate(rels))

def ndcg(golds, preds_texts, k=5):
    rels = []
    for i, t in enumerate(preds_texts[:k]):
        rel = 1 if any(_match(g, t) for g in golds) else 0
        rels.append(rel)
    ideal = sorted(rels, reverse=True)
    denom = dcg(ideal) or 1.0
    return dcg(rels) / denom

def mrr(golds, preds_texts, k=10):
    for i, t in enumerate(preds_texts[:k]):
        if any(_match(g, t) for g in golds):
            return 1.0 / (i+1)
    return 0.0

def recall_at_k(golds, preds_texts, k=5):
    return 1.0 if any(_match(g, t) for t in preds_texts[:k] for g in golds) else 0.0

def _match(pattern, text):
    try:
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    except re.error:
        return pattern.lower() in text.lower()

def main(csv_path):
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    r5 = []; mrr10=[]; ndcg5=[]
    for r in rows:
        q = r["query"].strip()
        golds = [g.strip() for g in r["expected"].split("||") if g.strip()]
        resp = query_api(q)
        cands = resp.get("debug",{}).get("rerank",{}).get("scored",[])
        if not cands:
            cits = resp.get("citations", [])
            preds = [c.get("text","") or c.get("preview","") or "" for c in cits]
        else:
            preds = [c.get("preview") or c.get("text","") or "" for c in cands]
        r5.append(recall_at_k(golds, preds, k=5))
        mrr10.append(mrr(golds, preds, k=10))
        ndcg5.append(ndcg(golds, preds, k=5))
    out = {
        "recall@5": round(sum(r5)/len(r5), 4) if r5 else 0.0,
        "MRR@10": round(sum(mrr10)/len(mrr10), 4) if mrr10 else 0.0,
        "nDCG@5": round(sum(ndcg5)/len(ndcg5), 4) if ndcg5 else 0.0,
        "n": len(rows)
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python eval_rag.py tests/eval_sample.csv")
        sys.exit(1)
    main(sys.argv[1])
