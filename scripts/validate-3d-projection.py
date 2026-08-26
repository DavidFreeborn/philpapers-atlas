#!/usr/bin/env python3
"""Validate the published 3D projection and its recorded provenance."""

from __future__ import annotations

from array import array
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA = ROOT / "public" / "data"
ANALYSIS = ROOT / "analysis"


def close(left: float, right: float, tolerance: float = 2e-6) -> bool:
    return abs(left - right) <= tolerance


def main() -> None:
    public_metadata = json.loads((PUBLIC_DATA / "projection-3d.json").read_text(encoding="utf-8"))
    selected_metadata = json.loads((ANALYSIS / "projection-3d-selected.json").read_text(encoding="utf-8"))
    if public_metadata != selected_metadata:
        raise ValueError("The published and analytical metadata differ")

    paper_count = int(public_metadata["paperCount"])
    if paper_count != 69_400 or public_metadata["dimensions"] != 3:
        raise ValueError("The projection dimensions or paper count are incorrect")
    map_metadata = json.loads((PUBLIC_DATA / "map.json").read_text(encoding="utf-8"))
    lens_catalog = json.loads((PUBLIC_DATA / "lenses" / "catalog.json").read_text(encoding="utf-8"))
    if map_metadata["count"] != paper_count or lens_catalog["paperCount"] != paper_count:
        raise ValueError("The 2D map, clustering lenses, and 3D projection are not aligned")

    binary_path = PUBLIC_DATA / "projection-3d.bin"
    payload = binary_path.read_bytes()
    expected_bytes = paper_count * 3 * 4
    if len(payload) != expected_bytes:
        raise ValueError(f"Expected {expected_bytes:,} coordinate bytes, found {len(payload):,}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != public_metadata["binarySha256"]:
        raise ValueError("The 3D coordinate checksum does not match its metadata")

    values = array("f")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("The 3D projection contains non-finite coordinates")

    means = [sum(values[axis::3]) / paper_count for axis in range(3)]
    if max(abs(value) for value in means) > 2e-6:
        raise ValueError(f"The normalized projection is not centred: {means}")
    radii = [
        math.sqrt(values[offset] ** 2 + values[offset + 1] ** 2 + values[offset + 2] ** 2)
        for offset in range(0, len(values), 3)
    ]
    if not close(max(radii), 1.0):
        raise ValueError(f"The normalized projection radius is {max(radii):.8f}, not 1")
    bounds = public_metadata["normalization"]["bounds"]
    actual_min = [min(values[axis::3]) for axis in range(3)]
    actual_max = [max(values[axis::3]) for axis in range(3)]
    if any(not close(actual_min[index], bounds["min"][index]) for index in range(3)):
        raise ValueError("The minimum coordinate bounds do not match the metadata")
    if any(not close(actual_max[index], bounds["max"][index]) for index in range(3)):
        raise ValueError("The maximum coordinate bounds do not match the metadata")

    with (ANALYSIS / "projection-3d-scan.csv").open(newline="", encoding="utf-8") as handle:
        scan = list(csv.DictReader(handle))
    if len(scan) != 10:
        raise ValueError(f"Expected 10 parameter-scan rows, found {len(scan)}")
    selected = max(
        (row for row in scan if row["selectionScore"]),
        key=lambda row: float(row["selectionScore"]),
    )
    display = public_metadata["display"]
    if int(selected["nNeighbors"]) != display["nNeighbors"] or not close(
        float(selected["minDist"]), display["minDist"], tolerance=1e-12
    ):
        raise ValueError("The published parameters are not the best stability-reviewed finalist")
    if public_metadata["testMetrics"]["qualityScore"] <= public_metadata["baseline2dTest"]["qualityScore"]:
        raise ValueError("The selected 3D projection does not improve on the recorded 2D holdout baseline")

    print(
        "Validated "
        f"{paper_count:,} papers, {len(payload) / 1024:.1f} KiB, "
        f"10 scan configurations, SHA-256 {digest[:12]}..., "
        f"holdout quality {public_metadata['testMetrics']['qualityScore']:.3f}."
    )


if __name__ == "__main__":
    main()
