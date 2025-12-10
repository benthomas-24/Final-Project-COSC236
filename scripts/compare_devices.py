# scripts/compare_devices.py
import os, argparse, json
import pandas as pd
import matplotlib.pyplot as plt

def load(csv):
    df = pd.read_csv(csv)
    if "t" not in df.columns:
        df["t"] = (df.index - df.index.min()) * 1.0
    meta_path = csv.replace(".csv", ".meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f: meta = json.load(f)
    label = meta.get("device_name", os.path.basename(csv).replace(".csv",""))
    return df, label, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="Two or more metrics CSVs (e.g., one from Pi, one from Mac)")
    ap.add_argument("--out", default="plots/compare_fps.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.figure(figsize=(12,6))

    target = None
    for csv in args.csvs:
        df, label, meta = load(csv)
        if target is None and "target_fps" in meta:
            target = float(meta["target_fps"])
        # Normalize time to start at 0
        t0 = df["t"].iloc[0]
        plt.plot(df["t"] - t0, df["fps"], label=label)

    if target is not None:
        plt.axhline(target, linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("FPS")
    plt.title("FPS: Device Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print("Saved", args.out)

if __name__ == "__main__":
    main()

