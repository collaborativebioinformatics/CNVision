# make_meta_infer.py
import csv, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out  = sys.argv[2] if len(sys.argv) > 2 else "meta_infer.csv"

rows = []
for p in sorted(root.glob("*.pt")):
    rows.append({"path": str(p.resolve()), "event_id": p.stem, "sample_id": ""})

with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["path","event_id","sample_id"])
    w.writeheader(); w.writerows(rows)

print(f"[OK] wrote {out} with {len(rows)} rows")
