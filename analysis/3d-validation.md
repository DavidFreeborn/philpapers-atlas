# Optional 3D atlas: validation report

## Result

The optional 3D view is ready for publication. It contains all 69,400 papers, preserves the existing paper order and clustering assignments, and leaves the 2D view as the default. The 3D coordinate asset is 813.3 KiB and is not requested until the reader selects 3D.

The display uses a matching reconstruction of the established analytical pipeline:

`SPECTER 768D → PCA 100D → UMAP 30D → display UMAP 3D`

The SPECTER model is pinned to revision `81cbfb43d4fc2e728d5b4201ce14987db8d0854c`. Titles and abstracts are aligned to the public atlas paper order and protected by a corpus SHA-256 hash.

## Parameter selection

Ten 3D UMAP configurations were fitted with a common random seed. Evaluation used the same fixed 5,000-paper sample for every candidate.

| Neighbours | Minimum distance | Trustworthiness@15 | Recall@15 | Recall@50 | Distance Spearman | Composite quality |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.05 | 0.984 | 0.625 | 0.681 | 0.354 | **0.751** |
| 50 | 0.10 | 0.978 | 0.626 | 0.675 | 0.293 | 0.742 |
| 30 | 0.10 | 0.975 | 0.613 | 0.660 | 0.283 | 0.733 |
| 30 | 0.05 | 0.977 | 0.614 | 0.658 | 0.266 | 0.732 |
| 15 | 0.20 | 0.963 | 0.607 | 0.628 | 0.287 | 0.722 |
| 15 | 0.10 | 0.957 | 0.582 | 0.603 | 0.214 | 0.700 |
| 15 | 0.05 | 0.956 | 0.566 | 0.584 | 0.248 | 0.695 |
| 15 | 0.00 | 0.958 | 0.553 | 0.575 | 0.245 | 0.690 |
| 10 | 0.10 | 0.944 | 0.554 | 0.557 | 0.197 | 0.677 |
| 10 | 0.05 | 0.945 | 0.538 | 0.544 | 0.188 | 0.670 |

Trustworthiness penalizes false neighbours introduced by the display. Recall measures how many 30D neighbours remain neighbours in 3D. The sampled Spearman coefficient measures broad agreement in pairwise distance order. The composite gives 35% each to trustworthiness and recall@15, 15% to recall@50, 10% to distance correlation, and 5% to protection against a collapsed axis.

The two leading candidates were fitted at three random seeds. Their mean 15-neighbour agreements were 0.775 for 50 neighbours/minimum distance 0.05 and 0.768 for 50/0.10. Combining projection quality with seed agreement retained 50/0.05.

## Holdout result

The selected configuration was then evaluated on a disjoint 5,000-paper holdout that had not been used to rank the candidates.

| Display | Trustworthiness@15 | Recall@15 | Recall@50 | Distance Spearman | Composite quality |
|---|---:|---:|---:|---:|---:|
| Selected 3D | **0.983** | **0.628** | **0.689** | **0.362** | **0.753** |
| Existing 2D | 0.769 | 0.123 | 0.164 | 0.109 | 0.398 |

This comparison concerns fidelity to the reconstructed 30D source, not clustering quality. The established 2D coordinates were generated previously, so the baseline should not be read as a controlled re-fit of 2D and 3D with identical display parameters.

Orthographic checks of all three axis pairs found no collapsed axis or unusable occlusion. The diagnostic image is `analysis/projection-3d-diagnostics.png`.

## Rendering and interaction

The existing WebGL 2 renderer now accepts either 2D or 3D position buffers. It draws the complete cloud in one call. Camera changes are event-driven; there is no permanent animation loop. Perspective size correction, a CSS-pixel size floor, edge smoothing, and distance fading keep points legible without making the cloud opaque.

3D selection uses an off-screen colour-identity pass with a depth buffer. This returns the frontmost visible paper and avoids testing 69,400 points on the CPU. The buffer is capped at 2,048 pixels on its longest side and is allocated only when picking is first needed.

Controls were checked for rotation, modified/right-drag panning, wheel zoom, point hover, point selection, paper details, cluster focus, search focus, reset, lens changes, repeated 2D/3D switching, and keyboard rotation/zoom/reset. The interaction grammar was informed by [Bible 3D](https://github.com/Macmachi/bible-3D), but the atlas uses an independent raw-WebGL implementation suited to its larger point cloud.

## Performance and responsive checks

Production output changed as follows:

| Asset | Before | After | Change |
|---|---:|---:|---:|
| JavaScript | 219.93 KiB | 235.74 KiB | +15.81 KiB |
| JavaScript, gzip | 68.95 KiB | 73.30 KiB | +4.35 KiB |
| CSS | 16.13 KiB | 18.63 KiB | +2.50 KiB |
| Lazy 3D coordinates | — | 813.3 KiB | requested on first 3D use |

No 3D runtime dependency was added. In the local browser harness, a 41-step sustained rotation completed in 297 ms (7.24 ms per delivered input), a GPU hover pick resolved in 19 ms, and a searched paper's title and abstract were ready in 275 ms. These are local measurements rather than guarantees for every device or network.

Responsive visual checks covered 390×844 phone portrait, 844×390 phone landscape, 1,024×768 tablet/small desktop, 1,280×720 desktop, and 1,440×900 full desktop. The cloud retains consistent framing; panels, map controls, the view switch, markers, and text remain usable at each breakpoint. Physical two-finger hardware was not available in the browser harness; the underlying zoom and pan mathematics are unit-tested, and the multi-pointer path uses standard pointer capture.

## Automated checks

- Seven camera/projection unit tests pass.
- Data validation confirms paper alignment, exact byte length, finite coordinates, centring, unit-radius normalization, bounds, SHA-256 checksum, scan selection, and holdout improvement.
- TypeScript, ESLint, the GitHub Pages production build, and the application build pass.
- The browser console remained free of warnings and errors throughout the interaction and responsive matrix.
- The first live deployment exposed an early-click race before the base map count was available. The view switch now remains disabled until the aligned 2D data has loaded.

The remaining interpretive limitation is intrinsic to UMAP: local neighbourhoods are useful, but global distances, apparent separations, and the orientation of the cloud are not measurements.
