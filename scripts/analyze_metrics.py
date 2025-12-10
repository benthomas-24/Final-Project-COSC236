# scripts/analyze_metrics.py
import os, argparse, json
import pandas as pd
import matplotlib.pyplot as plt

def load_meta(csv_path):
    meta_path = csv_path.replace(".csv", ".meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to a metrics_*.csv file")
    ap.add_argument("--outdir", default="plots", help="Where to save figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    meta = load_meta(args.csv)
    title_prefix = f"{meta.get('device_name','Run')} — {meta.get('resolution','')}".strip()

    # Basic cleaning: if 't' missing, synthesize from index
    if "t" not in df.columns:
        df["t"] = (df.index - df.index.min()) * 1.0
    
    # Handle NaN values (especially for GPU which might be empty)
    if "gpu" in df.columns:
        # Only plot GPU if there are non-NaN values
        if df["gpu"].isna().all():
            df = df.drop(columns=["gpu"])  # Remove GPU column if all NaN

    # 1) FPS over time
    plt.figure(figsize=(12,5))
    plt.plot(df["t"], df["fps"], linewidth=1.5)
    if "target_fps" in meta:
        plt.axhline(float(meta["target_fps"]), linestyle="--", color="red", alpha=0.7, label=f"Target: {meta['target_fps']} FPS")
        plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("FPS")
    plt.title(f"{title_prefix} — FPS")
    plt.grid(True, alpha=0.3)
    out1 = os.path.join(args.outdir, os.path.basename(args.csv).replace(".csv", "_fps.png"))
    plt.tight_layout()
    plt.savefig(out1, dpi=180)
    plt.close()

    # 2) CPU/GPU/Mem
    plt.figure(figsize=(12,5))
    if "cpu" in df.columns and not df["cpu"].isna().all():
        plt.plot(df["t"], df["cpu"], label="CPU %", linewidth=1.5)
    if "gpu" in df.columns and not df["gpu"].isna().all():
        plt.plot(df["t"], df["gpu"], label="GPU %", linewidth=1.5)
    if "mem" in df.columns and not df["mem"].isna().all():
        plt.plot(df["t"], df["mem"], label="Mem %", linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Utilization (%)")
    plt.title(f"{title_prefix} — Utilization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out2 = os.path.join(args.outdir, os.path.basename(args.csv).replace(".csv", "_util.png"))
    plt.tight_layout()
    plt.savefig(out2, dpi=180)
    plt.close()

    # 3) Inference latency
    if "model_latency_ms" in df.columns and not df["model_latency_ms"].isna().all():
        plt.figure(figsize=(12,5))
        plt.plot(df["t"], df["model_latency_ms"], linewidth=1.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Latency (ms)")
        plt.title(f"{title_prefix} — YOLO Latency")
        plt.grid(True, alpha=0.3)
        out3 = os.path.join(args.outdir, os.path.basename(args.csv).replace(".csv", "_latency.png"))
        plt.tight_layout()
        plt.savefig(out3, dpi=180)
        plt.close()

    # 4) Quick text summary
    summary = {
        "fps_mean": float(df["fps"].mean()) if "fps" in df.columns and not df["fps"].isna().all() else None,
        "fps_p95": float(df["fps"].quantile(0.95)) if "fps" in df.columns and not df["fps"].isna().all() else None,
        "latency_ms_mean": float(df["model_latency_ms"].mean()) if "model_latency_ms" in df.columns and not df["model_latency_ms"].isna().all() else None,
        "cpu_mean": float(df["cpu"].mean()) if "cpu" in df.columns and not df["cpu"].isna().all() else None,
        "gpu_mean": float(df["gpu"].mean()) if "gpu" in df.columns and not df["gpu"].isna().all() else None,
        "mem_mean": float(df["mem"].mean()) if "mem" in df.columns and not df["mem"].isna().all() else None,
    }
    print("SUMMARY:", json.dumps(summary, indent=2))
    saved_files = [out1, out2]
    if "model_latency_ms" in df.columns and not df["model_latency_ms"].isna().all():
        saved_files.append(out3)
    print("Saved:", saved_files)

if __name__ == "__main__":
    main()

