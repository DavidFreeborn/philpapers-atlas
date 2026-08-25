# Generated PhilPapers Atlas data

These files are generated from the saved PhilPapers clustering artefacts.
The application does not refit PCA, UMAP, HDBSCAN, or k-means.

- `map.json`: saved two-dimensional UMAP coordinates and HDBSCAN labels.
- `clusters.json`: reviewed cluster names, sizes, and TF-IDF terms.
- `search.json`: compact title/author/date/URL index used for immediate paper details and search.
- `details/*.json`: abstract-only shards, prefetched on hover and fetched on demand.
- `lenses/catalog.json`: methods, metrics, names, terms, and counts for the available clustering lenses.
- `lenses/labels/*.bin`: compact Int16 assignment arrays, loaded on demand; all use the same paper order.
- `manifest.json`: provenance and validation counts.

Regenerate with `scripts/build-map-data.py`; its source directory must contain
the seven saved inputs named in that script. Regenerate the clustering lenses with
`scripts/build-lenses-data.py` and a source directory containing Prajakta's saved
label arrays, reviewed names, TF-IDF terms, and metric tables. The additional
display-space HDBSCAN labels and scan metrics are stored under `analysis/` and
can be reproduced with `scripts/scan-hdbscan-2d.py`.
