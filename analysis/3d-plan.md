# Optional 3D atlas: implementation and validation plan

## Objective

Add an optional three-dimensional view of the same 69,400 papers without changing the established two-dimensional map, paper order, clustering lenses, colours, search, or metadata. The two-dimensional view remains the default and must not pay the network or rendering cost of the 3D asset until the user requests it.

## Projection

1. Reconstruct the SPECTER input in the existing paper order from each title and abstract.
2. Produce 768-dimensional `allenai/specter` embeddings using the model's title–separator–abstract format.
3. Apply the established reduction pipeline: PCA to 100 dimensions, then UMAP to 30 dimensions with cosine distance, 15 neighbours, and minimum distance 0.
4. Scan three-dimensional display UMAP settings over neighbourhood size and minimum distance. Compare local trustworthiness, neighbourhood recall at 15 and 50 neighbours, sampled distance correlation, shape degeneracy, and agreement across random seeds.
5. Select a result on quantitative and visual-structure grounds, normalize it without changing relative geometry, and export it as a little-endian Float32 binary aligned to the existing paper order.
6. Record the configuration, corpus hash, metrics, bounds, and asset checksum. Validate the binary by round-tripping it and checking its shape, finiteness, variance, and paper count.

The 3D view is a display projection, not a new clustering space. Changing between 2D and 3D changes positions but never paper identities or lens assignments.

## Interaction design

- A compact `2D / 3D` switch sits over the map and is usable by keyboard and touch.
- First use of 3D lazy-loads the binary and reports a small in-place loading state; returning to 3D is immediate.
- Primary drag rotates the cloud. Shift-drag, middle-drag, or right-drag pans. The wheel or trackpad zooms. Two-finger touch pans and pinches; one finger rotates.
- Arrow keys rotate, `+` and `−` zoom, and `R` resets the camera.
- Point hover, selection, paper details, cluster/topic isolation, search focus, and reset work in both views.
- A restrained help control explains the gestures and the projection's interpretive limitation. No semantic axes are drawn because UMAP orientation has no meaning.
- Portrait and narrow screens receive a wider effective field of view and touch targets of at least 44 CSS pixels.

## Rendering architecture

- Extend the existing raw WebGL 2 renderer rather than add a general-purpose 3D engine.
- Keep one GPU point draw for the complete corpus. Lens changes update colour and assignment buffers only.
- Use a perspective view-projection matrix in 3D and preserve the exact orthographic path in 2D.
- Cap device-pixel ratio at 2. Point sprites retain a stable CSS-pixel floor, soften at their edge, and reduce distant-point opacity to avoid shimmer while preserving colour meaning.
- Render only when data, camera, selection, filter, lens, or viewport state changes. There is no permanent animation loop.
- Use an off-screen GPU identity pass for 3D picking, throttled to animation frames, rather than ray-testing 69,400 points in JavaScript.
- Dispose every program, buffer, texture, renderbuffer, framebuffer, and animation request on unmount.

## Systematic tests

### Data and scientific validity

- Exact 69,400-row alignment and deterministic corpus hash.
- Finite coordinates, non-zero variance on all three axes, bounded aspect ratios, and a verified binary checksum.
- Held-out/sample neighbourhood preservation against the 30-dimensional source.
- Comparison with the existing 2D display on the same metrics.
- Multi-seed agreement for the shortlisted 3D configurations.

### Geometry and controls

- Matrix multiplication, look-at and perspective invariants.
- Projection of the camera target to screen centre.
- Rotation radius preservation and pitch limits.
- Zoom and pan bounds.
- Camera fitting for the full corpus, a cluster, and an individual paper.
- Correct treatment of points behind the camera and portrait aspect ratios.

### Application regressions

- Every clustering/topic lens in 2D and 3D.
- Noise visibility, isolated cluster selection, search focus, hover, paper selection, Back and ×, reset, view switching, and repeated switching.
- Mouse, trackpad, keyboard, single-touch and two-touch paths.
- Desktop, tablet, phone, portrait, landscape, coarse-pointer and reduced-motion layouts.
- WebGL context creation failure and 3D asset load failure leave the 2D atlas usable.

### Performance targets

- No increase to the initial 2D data payload.
- 3D coordinate payload below 1 MB before HTTP compression.
- One draw call for the visible cloud and one additional draw only when picking is requested.
- Camera interaction schedules no more than one render per animation frame.
- Lens switching performs two buffer updates without rebuilding geometry or programs.
- Responsive resize does not recreate the renderer.
- Production JavaScript growth remains modest and no runtime 3D dependency is added.

## Iteration procedure

1. Establish baseline asset sizes and production bundle size.
2. Implement the smallest complete 3D path and run scientific, unit, type, lint, and build checks.
3. Measure projection quality, render scheduling, buffer updates, picking, bundle size, and memory allocations.
4. Correct the largest measured weakness, then repeat the complete check set.
5. Run a final regression matrix, publish to GitHub Pages, and verify the deployed metadata and binary assets independently of local files.
