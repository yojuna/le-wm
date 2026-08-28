#!/usr/bin/env python3
"""Verify local PushT checkpoint matches Hugging Face quentinll/lewm-pusht.

    python scripts/verify_hf_pusht.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STABLEWM_HOME", str(ROOT.parent / "stablewm"))

HF_REPO = "quentinll/lewm-pusht"
EXPECTED_WEIGHT_BYTES = 72_290_721  # HF listing ~72.3 MB


def sha256_file(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
            if limit is not None and n >= limit:
                break
    return h.hexdigest()


def main() -> int:
    from eval_setup import load_lewm_checkpoint

    folder = Path(os.environ["STABLEWM_HOME"]) / "checkpoints" / "hf_pusht"
    cfg_path = folder / "config.json"
    wt_path = folder / "weights.pt"
    assert cfg_path.exists() and wt_path.exists(), f"missing files under {folder}"

    local_cfg = json.loads(cfg_path.read_text())
    remote_cfg = json.loads(
        urllib.request.urlopen(
            f"https://huggingface.co/{HF_REPO}/raw/main/config.json"
        ).read()
    )
    print(f"repo: https://huggingface.co/{HF_REPO}")
    print(f"local dir: {folder}")
    print(f"config.json match remote: {local_cfg == remote_cfg}")
    print(f"weights.pt size: {wt_path.stat().st_size} bytes "
          f"(expected ~{EXPECTED_WEIGHT_BYTES})")
    size_ok = abs(wt_path.stat().st_size - EXPECTED_WEIGHT_BYTES) < 1024
    print(f"weights size OK: {size_ok}")
    print(f"weights sha256: {sha256_file(wt_path)}")

    print("architecture:")
    print(f"  _target_={local_cfg['_target_']}")
    print(f"  encoder={local_cfg['encoder']['size']} "
          f"patch={local_cfg['encoder']['patch_size']} "
          f"image={local_cfg['encoder']['image_size']}")
    print(f"  predictor frames={local_cfg['predictor']['num_frames']} "
          f"depth={local_cfg['predictor']['depth']}")
    print(f"  action_encoder input_dim={local_cfg['action_encoder']['input_dim']} "
          f"(PushT: action_block 5 × 2)")

    model = load_lewm_checkpoint("hf_pusht")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded type: {type(model).__module__}.{type(model).__name__}")
    print(f"parameters: {n_params:,}")
    print(f"has get_cost: {hasattr(model, 'get_cost')}")

    ok = local_cfg == remote_cfg and size_ok and hasattr(model, "get_cost")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
