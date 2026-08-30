# Structurally selected finalists

Frozen on 27 August 2026, before held-out semantic validation.

The parameter and seed stages did not identify a single structural winner. Seven finalists were therefore retained to represent the important empirical contrasts rather than to maximise a composite score.

1. **Current 30D / 42 clusters.** The published reference result and best-DBCV cell in the exact canonical 30D parameter grid.
2. **Typical seeded 30D / 49 clusters.** The medoid of the five Linux 30D replication seeds. This separates the behaviour of the originally saved seed-42 array from the typical behaviour of independently refitted 30D UMAPs.
3. **Parameter-stable 20D / 49 clusters.** PCA100→UMAP20 with HDBSCAN(150, 50). DBCV is 0.4323 and mean adjacent-grid ARI is 0.8917.
4. **High-DBCV 20D / 29 clusters.** PCA100→UMAP20 with HDBSCAN(300, 50). This has the highest non-degenerate DBCV in the primary grid (0.4943), with a coarser resolution and mean adjacent-grid ARI 0.7879.
5. **Direct 2D medoid / 70 clusters.** The medoid of the five Linux direct-2D replications, retained to test the claim that clustering after reduction to two dimensions might be advantageous.
6. **Direct 2D with the historical HDBSCAN parameters / 63 clusters.** The medoid of fifteen independent 2D fits clustered with scikit-learn HDBSCAN(150, 50). This protocol clarification was added before semantic validation because the first direct-2D finalist used (200, 15), whereas the strong historical display score had used (150, 50).
7. **Fixed display 2D / 104 clusters.** The atlas’s two-stage display projection clustered with the saved scikit-learn HDBSCAN settings. This is kept separate from direct PCA100→UMAP2 because the two procedures do not estimate the same representation.

The PCA50 branch was not advanced. It produced degenerate three-cluster solutions in two of three 20D seeds and one of three 30D seeds; 8–11 of 24 EOM grid cells per dimension also fell outside the preregistered substantive range. Its apparently very high DBCV values occurred precisely in those collapsed solutions. PCA50→25D was more stable, but its non-degenerate validity did not improve on the retained PCA100 candidates.

The finalists are candidates, not endorsed atlas lenses. They were frozen before original-SPECTER scores, text coherence, permutation-null comparisons, or cluster-level semantic audits were inspected.
