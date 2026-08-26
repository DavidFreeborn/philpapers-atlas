#!/usr/bin/env python3
"""Render fixed diagnostic views of the selected 3D atlas projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NOISE = "#777b84"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", type=Path, default=Path("analysis/projection-3d.npy"))
    parser.add_argument("--map", type=Path, default=Path("public/data/map.json"))
    parser.add_argument("--catalog", type=Path, default=Path("public/data/lenses/catalog.json"))
    parser.add_argument("--output", type=Path, default=Path("analysis/projection-3d-diagnostics.png"))
    args = parser.parse_args()

    values = np.load(args.projection, allow_pickle=False)
    map_data = json.loads(args.map.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    selected = next(lens for lens in catalog["lenses"] if lens["id"] == catalog["defaultLens"])
    palette = {cluster["id"]: cluster["color"] for cluster in selected["clusters"]}
    labels = np.asarray([point[2] for point in map_data["points"]], dtype=np.int16)
    colours = np.asarray([palette.get(int(label), NOISE) for label in labels])

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#12121a")
    views = ((0, 1, "x / y"), (0, 2, "x / z"), (1, 2, "y / z"))
    for axis, (horizontal, vertical, title) in zip(axes, views, strict=True):
        axis.set_facecolor("#12121a")
        noise = labels < 0
        axis.scatter(
            values[noise, horizontal],
            values[noise, vertical],
            s=0.35,
            c=NOISE,
            alpha=0.24,
            linewidths=0,
            rasterized=True,
        )
        axis.scatter(
            values[~noise, horizontal],
            values[~noise, vertical],
            s=0.45,
            c=colours[~noise],
            alpha=0.72,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(title, color="#d9dbe4", fontsize=11, pad=8)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle("Selected 3D UMAP · fixed orthographic diagnostics", color="#ececf1", fontsize=14)
    figure.tight_layout(pad=1.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
