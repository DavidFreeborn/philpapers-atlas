# Generated PhilPapers Atlas data

These files are generated from the saved PhilPapers clustering and topic-model artefacts.
The application does not refit PCA, UMAP, HDBSCAN, k-means, or LDA.

- `map.json`: saved two-dimensional UMAP coordinates and HDBSCAN labels.
- `clusters.json`: reviewed cluster names, sizes, and TF-IDF terms.
- `search.json`: compact title/author/date/URL index used for immediate paper details and search.
- `details/*.json`: abstract-only shards, prefetched on hover and fetched on demand.
- `lenses/catalog.json`: methods, metrics, names, terms, and counts for the available clustering lenses.
- `lenses/labels/*.bin`: compact Int16 assignment arrays, loaded on demand; all use the same paper order.
- `projection-3d.bin`: normalized little-endian Float32 XYZ coordinates in the same paper order, loaded only when 3D is requested.
- `projection-3d.json`: 3D method, parameter, validation, normalization, corpus-hash, and checksum metadata.
- `exploration/neighbors.json`: SPECTER nearest-neighbour method, shard, encoding, and exact-audit metadata.
- `exploration/neighbors/*.bin`: packed 30-neighbour UInt24 rows in 200-paper shards, fetched only when requested.
- `exploration/representatives.json`: the ten papers closest to every lens cluster's centroid in the original normalised 768D SPECTER space.
- `manifest.json`: provenance and validation counts.

Regenerate with `scripts/build-map-data.py`; its source directory must contain
the seven saved inputs named in that script. Regenerate the clustering lenses with
`scripts/build-lenses-data.py` and a source directory containing Prajakta's saved
label arrays, reviewed names, TF-IDF terms, and metric tables. The additional
display-space HDBSCAN labels and scan metrics are stored under `analysis/` and
can be reproduced with `scripts/scan-hdbscan-2d.py`. LDA scan results, selected
model metadata, and assignments are also stored under `analysis/`; reproduce them
with `scripts/scan-lda.py`, then update an existing catalogue with
`scripts/build-lenses-data.py --lda-only`.

The two robustness-study HDBSCAN lenses use the exact frozen 30D/49-cluster and
20D/29-cluster assignments evaluated in `analysis/robustness-study/REPORT.md`.
Regenerate their labels, provisional names, metrics, and overlap-matched colours
with `scripts/build-lenses-data.py --robustness-only`. Validate every catalogue
entry with `scripts/validate-lenses-data.py`.

Regenerate nearest-neighbour shards and characteristic-paper rankings from the
frozen normalised SPECTER matrix and audited approximate-neighbour graph with
`scripts/build-exploration-data.py`. The published 30-neighbour graph had mean
exact-neighbour recall of 0.9985 at 15 neighbours and 0.9974 at 50 neighbours on
500 fixed audit queries. Validate all shards, paper indices, memberships, and
rank order with `scripts/validate-exploration-data.py`.

The optional 3D data can be reproduced with `scripts/build-3d-projection.py`.
The script pins the SPECTER model revision, rebuilds the 100D PCA and 30D source
UMAP, evaluates ten 3D UMAP configurations, checks the two finalists across
three seeds, and reports a disjoint 5,000-paper holdout. Intermediate embeddings
and reductions are resumable and ignored by Git. Run
`scripts/validate-3d-projection.py` to check alignment, byte length, checksum,
normalization, selected parameters, and holdout improvement.
