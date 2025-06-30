# compute_meta_from_csv.py
import glob, csv, json, os
import numpy as np
import argparse

# only the keys that actually live in your per-volume CSVs:
NUMERIC_FIELDS = [
    "FieldStrength(T)",
    "FOVx(mm)",
    "FOVy(mm)",
    "ReconMatrix_X",
    "ReconMatrix_Y",
    "SliceNum",
    "SliceThickness(mm)",
    "CoilNumber",
    "TemporalPhase",
    "ReadOutOversample",
    "TR(ms)",
    "TE(ms)",
    "TI(ms)",
    "FlipAngle(degree)"
]

def compute_stats(csv_root, out_json="meta_stats.json"):
    # gather
    vals   = {k: [] for k in NUMERIC_FIELDS}
    counts = {k:  0 for k in NUMERIC_FIELDS}

    # recurse for any *_info.csv under csv_root
    pattern = os.path.join(csv_root, "**", "*_info.csv")
    for path in glob.glob(pattern, recursive=True):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            keycol = "Parameter" if "Parameter" in reader.fieldnames else reader.fieldnames[0]
            valcol = "Value"     if "Value"     in reader.fieldnames else reader.fieldnames[1]

            for row in reader:
                key = row[keycol].strip()
                if key not in NUMERIC_FIELDS:
                    continue

                raw = row[valcol].strip()
                # 1) blank?
                if raw == "":
                    print(f"   SKIP {path}: {key!r} is blank")
                    continue

                # 2) parse float
                try:
                    f = float(raw)
                except Exception as e:
                    print(f"   SKIP {path}: {key!r} has non-float value {raw!r} ({e})")
                    continue

                # 3) literal NaN?
                if np.isnan(f):
                    print(f"   SKIP {path}: {key!r} is NaN")
                    continue

                # good!
                vals[key].append(f)
                counts[key] += 1

    # report your counts
    print("non-empty, non-NaN samples per field:")
    for k in NUMERIC_FIELDS:
        print(f"  {k:20s}: {counts[k]}")

    # now compute mean/std (we know xs is non-empty if count > 0)
    stats = {}
    for k, xs in vals.items():
        arr = np.array(xs, dtype=np.float64)
        if arr.size == 0:
            # guard against a completely empty field
            stats[k] = {"mean": 0.0, "std": 1.0}
        else:
            stats[k] = {
                "mean": float(arr.mean()),
                "std":  float(arr.std(ddof=0))
            }

    # dump
    with open(out_json, "w") as fo:
        json.dump(stats, fo, indent=2)
    print(f"Wrote stats for {len(stats)} fields to {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_root",
        type=str,
        required=True,
        help="root folder containing your *_info.csv files"
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="meta_stats.json",
        help="where to write mean/std for each field"
    )
    args = parser.parse_args()

    compute_stats(args.csv_root, args.out_json)
