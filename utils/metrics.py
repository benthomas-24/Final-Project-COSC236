# utils/metrics.py
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import csv, json, os, sys, time, platform

@dataclass
class RunMetadata:
    device_name: str
    model_name: str
    target_fps: float
    resolution: str
    os_name: str = platform.system()
    os_release: str = platform.release()
    python: str = sys.version.split()[0]
    extra: Dict[str, str] = None

class MetricsLogger:
    """
    Lightweight metrics recorder. Appends tiny dicts in-memory and writes a CSV once at the end.
    Optional sampling via sample_every to reduce overhead on slower devices.
    """
    def __init__(self, out_dir: str = "metrics", run_tag: Optional[str] = None,
                 meta: Optional[RunMetadata] = None, sample_every: int = 1):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.sample_every = max(1, int(sample_every))
        self.rows: List[Dict] = []
        self.t0 = time.time()
        self.meta = meta
        self.run_tag = run_tag or time.strftime("%Y%m%d_%H%M%S")
        base = f"{self.meta.device_name if self.meta else 'device'}_{self.run_tag}"
        self.csv_path = os.path.join(out_dir, f"metrics_{base}.csv")
        self.meta_path = os.path.join(out_dir, f"metrics_{base}.meta.json")
        if self.meta:
            with open(self.meta_path, "w") as f:
                json.dump(asdict(self.meta), f, indent=2)

    def log(self, frame: int, **kwargs):
        # Sampling (e.g., log every N frames)
        if frame % self.sample_every != 0:
            return
        row = {"t": time.time() - self.t0, "frame": frame}
        row.update(kwargs)
        self.rows.append(row)

    def save(self) -> Optional[str]:
        if not self.rows:
            return None
        # Stable column order
        fieldnames = sorted({k for r in self.rows for k in r.keys()})
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.rows)
        return self.csv_path
