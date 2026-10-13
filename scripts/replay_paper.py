"""Replay all original PaLR-DF/V2 states with one portable command."""
import argparse
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--assets", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--device", default="cpu")
p.add_argument("--mode", choices=["evaluate", "spectrum"], default="evaluate")
p.add_argument("--threads", type=int, default=4)
args = p.parse_args()
repo = Path(__file__).resolve().parents[1]
for family in ("palr", "v2"):
    for horizon in ("12h", "24h"):
        for seed in range(2024, 2029):
            output = args.output.resolve() / family / horizon / f"seed_{seed}"
            subprocess.run([
                sys.executable, "-m", "palr_df", args.mode,
                "--repo", str(repo), "--assets", str(args.assets.resolve()),
                "--family", family, "--horizon", horizon, "--seed", str(seed),
                "--satellites", "all", "--device", args.device,
                "--threads", str(args.threads), "--output", str(output),
            ], check=True, cwd=repo)
