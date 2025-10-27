import csv, os, sys, hashlib
from pathlib import Path
import argparse

def iter_dat(root: Path):
    for logger in sorted(p.name for p in root.iterdir() if p.is_dir()):
        lpath = root / logger
        for date_dir in sorted(p.name for p in lpath.iterdir() if p.is_dir()):
            r = lpath / date_dir / "raw"
            if not r.exists(): 
                continue
            for f in sorted(r.glob("*.DAT")):
                yield logger, date_dir, f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["logger","date","dat_path","size_bytes","md5_8"])
        for logger, date_str, f in iter_dat(raw_root):
            size = f.stat().st_size
            md5 = hashlib.md5(f.read_bytes()).hexdigest()[:8]
            w.writerow([logger, date_str, str(f), size, md5])

if __name__ == "__main__":
    main()
