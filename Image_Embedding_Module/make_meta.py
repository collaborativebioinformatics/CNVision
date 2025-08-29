#!/usr/bin/env python3
import re, csv, sys
from pathlib import Path

# Could possibly be:
# HG002_GRCh38_FP_chr22_19202406_19203410_INS_1004bp.pt
# HG002_GRCh37_FP_2_242145688_242145799_DEL_112bp.pt
PAT = re.compile(
    r'^(?P<sample>[^_]+)_(?P<ref>(?:GRCh\d+|hg\d+|[^_]+))_(?P<label>TP|FP)_'
    r'(?P<chrom>(?:chr)?(?:[0-9]{1,2}|X|Y|M|MT))_'
    r'(?P<start>\d+)_(?P<end>\d+)_'
    r'(?P<svtype>[A-Z]+)_(?P<size>\d+)bp\.pt$'
)

def norm_chrom(ch):
    ch = ch if ch.startswith("chr") else f"chr{ch}"
    # unify mitochondria variants
    if ch in ("chrMT", "chrMt"): ch = "chrM"
    return ch

def to_row(p):
    m = PAT.match(p.name)
    if not m:
        # Fallback: write a row you can edit later
        return {"path": str(p.resolve()), "label": "", "sample_id": "", "event_id": p.stem,
                "ref":"", "chrom":"", "chrom_norm":"", "start":"", "end":"", "svtype":"", "size":"", "filename": p.name}
    d = m.groupdict()
    label = 1 if d["label"] == "TP" else 0
    chrom_norm = norm_chrom(d["chrom"])
    # include build in event_id so 37 vs 38 never collide
    event_id = f'{d["sample"]}:{d["ref"]}:{chrom_norm}:{d["start"]}-{d["end"]}:{d["svtype"]}'
    return {
        "path": str(p.resolve()),
        "label": label,
        "sample_id": d["sample"],
        "event_id": event_id,
        "ref": d["ref"],
        "chrom": d["chrom"],
        "chrom_norm": chrom_norm,
        "start": d["start"],
        "end": d["end"],
        "svtype": d["svtype"],
        "size": d["size"],
        "filename": p.name
    }

def main(root, out_csv="meta.csv"):
    ps = sorted(Path(root).glob("*.pt"))
    rows = [to_row(p) for p in ps]
    with open(out_csv, "w", newline="") as f:
        cols = ["path","label","sample_id","event_id","ref","chrom","chrom_norm",
                "start","end","svtype","size","filename"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"[OK] wrote {out_csv} ({len(rows)} rows)")

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "meta.csv"
    main(root, out)
