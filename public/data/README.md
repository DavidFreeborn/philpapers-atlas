# Generated PhilPapers Atlas data

These files are generated from the saved `candidate-42` clustering artefacts.
The application does not refit PCA, UMAP, or HDBSCAN.

- `map.json`: saved two-dimensional UMAP coordinates and HDBSCAN labels.
- `clusters.json`: reviewed cluster names, sizes, and TF-IDF terms.
- `search.json`: compact title/author/date search index.
- `details/*.json`: deferred paper metadata and abstract shards.
- `manifest.json`: provenance and validation counts.

Regenerate with `scripts/build-map-data.py`; its source directory must contain
the seven saved inputs named in that script.
