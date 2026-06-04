#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compute PSNR + SSIM proxy for ASA probe MP4s vs baseline.

Expects an output dir produced by running mgm_video_t2v.py three times
(baseline / dense_probe / asa_0p20) per prompt with naming convention
``{run}_{prompt_id}_*.mp4``, e.g.::

    runs/asa_probe_2026-06-03/
      baseline_p01_person.mp4
      dense_probe_p01_person.mp4
      asa_0p20_p01_person.mp4
      ...

Writes ``<output-dir>/metrics.json`` and prints the table.

Usage::

    python scripts/asa_metrics.py --output-dir runs/asa_probe_2026-06-03
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from torchvision.io import read_video

# Match {run}_{prompt_id}*.mp4 where run is one of the known names.
RUN_NAMES = ("baseline", "dense_probe", "asa_0p20")
_PATTERN = re.compile(r"^(baseline|dense_probe|asa_0p20)_(p\d+(?:_[a-z]+)?)(?:_.*)?\.mp4$")


def discover(out_dir: Path) -> dict[str, dict[str, Path]]:
    """Return {run_name: {prompt_id: path}}."""
    found: dict[str, dict[str, Path]] = defaultdict(dict)
    for p in sorted(out_dir.glob("*.mp4")):
        m = _PATTERN.match(p.name)
        if m is None:
            continue
        run, pid = m.group(1), m.group(2)
        found[run][pid] = p
    return dict(found)


def load_video(path: Path) -> torch.Tensor:
    """Decode mp4 to uint8 [T, H, W, C]."""
    frames, _audio, _info = read_video(str(path), pts_unit="sec", output_format="THWC")
    if frames.dtype != torch.uint8:
        frames = frames.clamp(0, 255).to(torch.uint8)
    return frames


def psnr(pred: torch.Tensor, ref: torch.Tensor) -> float:
    mse = (pred.float() - ref.float()).pow(2).mean().item()
    if mse < 1e-12:
        return float("inf")
    return 10.0 * math.log10(255.0 ** 2 / mse)


def ssim_proxy(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """Lightweight SSIM proxy: per-channel mean structural correlation."""
    p = pred.float() / 255.0
    r = ref.float() / 255.0
    p_flat = p.reshape(-1, p.shape[-1])
    r_flat = r.reshape(-1, r.shape[-1])
    mu_p = p_flat.mean(0)
    mu_r = r_flat.mean(0)
    sigma_p = ((p_flat - mu_p) ** 2).mean(0).clamp(min=0).sqrt()
    sigma_r = ((r_flat - mu_r) ** 2).mean(0).clamp(min=0).sqrt()
    cov = ((p_flat - mu_p) * (r_flat - mu_r)).mean(0)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = (2 * mu_p * mu_r + c1) * (2 * cov + c2) / (
        (mu_p ** 2 + mu_r ** 2 + c1) * (sigma_p ** 2 + sigma_r ** 2 + c2)
    )
    return ssim.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory containing {run}_{prompt}*.mp4 files")
    args = parser.parse_args()

    found = discover(args.output_dir)
    if "baseline" not in found:
        raise SystemExit(f"No baseline_*.mp4 found in {args.output_dir}")

    baseline_videos = {pid: load_video(p) for pid, p in found["baseline"].items()}
    if not baseline_videos:
        raise SystemExit("baseline directory matched but contained no parseable mp4s")

    table: dict[str, dict[str, dict[str, float]]] = {}
    for run in ("dense_probe", "asa_0p20"):
        if run not in found:
            print(f"[skip] {run}: no mp4s found")
            continue
        table[run] = {}
        for pid, ref in baseline_videos.items():
            test_path = found[run].get(pid)
            if test_path is None:
                print(f"[skip] {run}/{pid}: missing")
                continue
            test = load_video(test_path)
            if test.shape != ref.shape:
                print(f"[warn] {run}/{pid}: shape mismatch test={test.shape} ref={ref.shape}, skipping")
                continue
            table[run][pid] = {
                "psnr_db": psnr(test, ref),
                "ssim_proxy": ssim_proxy(test, ref),
            }

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(table, indent=2))
    print(f"\nWrote {metrics_path}")
    print(json.dumps(table, indent=2))

    # P3 acceptance gate (plan §7): dense_probe PSNR >= 35 dB on every prompt.
    if "dense_probe" in table:
        gate_ok = True
        for pid, m in table["dense_probe"].items():
            if m["psnr_db"] < 35.0:
                print(f"\nWARN: dense_probe PSNR < 35 dB for {pid}: {m['psnr_db']:.2f} dB")
                gate_ok = False
        if gate_ok:
            print("\nP3 gate PASSED: dense_probe PSNR >= 35 dB on all prompts.")
        else:
            print("\nP3 gate FAILED: see WARN lines above.")


if __name__ == "__main__":
    main()
