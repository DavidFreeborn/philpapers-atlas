'use client';

import {
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import type { MouseEvent } from 'react';

type PointRow = [number, number, number, number];
type SearchRow = [string, string, string, string];
type DetailRow = [string, string, string, string, string, string, string];

type MapData = {
  version: string;
  count: number;
  bounds: { minX: number; maxX: number; minY: number; maxY: number };
  points: PointRow[];
};

type Cluster = {
  id: number;
  label: string;
  count: number;
  terms: string[];
};

type ClusterData = {
  noiseCount: number;
  clusters: Cluster[];
};

type View = { zoom: number; panX: number; panY: number };

const GRID_SIZE = 128;
const DETAIL_CHUNK_SIZE = 500;

function clusterColor(id: number): string {
  if (id < 0) return '#7f898b';
  const hue = (id * 137.508 + 21) % 360;
  return `hsl(${hue.toFixed(1)} 70% 64%)`;
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-GB').format(value);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function Loader() {
  return (
    <div className="atlas-loader" role="status" aria-live="polite">
      <div className="loader-orbit" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <p className="eyebrow">Loading the atlas</p>
      <h1>Placing 69,400 papers</h1>
      <p>The saved PhilPapers projection is being prepared for exploration.</p>
    </div>
  );
}

export default function Atlas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mapShellRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{
    pointerId: number;
    x: number;
    y: number;
    originX: number;
    originY: number;
    moved: boolean;
  } | null>(null);
  const spatialRef = useRef<number[][]>([]);
  const detailCacheRef = useRef<Map<number, DetailRow[]>>(new Map());

  const [mapData, setMapData] = useState<MapData | null>(null);
  const [clusterData, setClusterData] = useState<ClusterData | null>(null);
  const [searchData, setSearchData] = useState<SearchRow[] | null>(null);
  const [loadError, setLoadError] = useState('');
  const [view, setView] = useState<View>({ zoom: 1, panX: 0, panY: 0 });
  const [activeCluster, setActiveCluster] = useState<number | null>(null);
  const [showNoise, setShowNoise] = useState(true);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<DetailRow | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [showMethod, setShowMethod] = useState(false);
  const [showLegend, setShowLegend] = useState(false);
  const [showInspector, setShowInspector] = useState(false);
  const deferredQuery = useDeferredValue(query.trim().toLowerCase());

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch('/data/map.json').then((response) => {
        if (!response.ok) throw new Error('The saved map could not be loaded.');
        return response.json() as Promise<MapData>;
      }),
      fetch('/data/clusters.json').then((response) => {
        if (!response.ok) throw new Error('The cluster descriptions could not be loaded.');
        return response.json() as Promise<ClusterData>;
      }),
    ])
      .then(([map, clusters]) => {
        if (cancelled) return;
        setMapData(map);
        setClusterData(clusters);
      })
      .catch((error: Error) => {
        if (!cancelled) setLoadError(error.message);
      });

    fetch('/data/search.json')
      .then((response) => {
        if (!response.ok) throw new Error('Search index unavailable');
        return response.json() as Promise<SearchRow[]>;
      })
      .then((search) => {
        if (!cancelled) setSearchData(search);
      })
      .catch(() => {
        // The visual map remains useful before or without the search index.
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const clusterById = useMemo(() => {
    const result = new Map<number, Cluster>();
    clusterData?.clusters.forEach((cluster) => result.set(cluster.id, cluster));
    return result;
  }, [clusterData]);

  const pointGroups = useMemo(() => {
    if (!mapData) return new Map<number, number[]>();
    const groups = new Map<number, number[]>();
    mapData.points.forEach((point, index) => {
      const group = groups.get(point[2]);
      if (group) group.push(index);
      else groups.set(point[2], [index]);
    });
    return groups;
  }, [mapData]);

  useEffect(() => {
    if (!mapData) return;
    const buckets = Array.from({ length: GRID_SIZE * GRID_SIZE }, () => [] as number[]);
    const { minX, maxX, minY, maxY } = mapData.bounds;
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    mapData.points.forEach((point, index) => {
      const cellX = clamp(Math.floor(((point[0] - minX) / spanX) * GRID_SIZE), 0, GRID_SIZE - 1);
      const cellY = clamp(Math.floor(((point[1] - minY) / spanY) * GRID_SIZE), 0, GRID_SIZE - 1);
      buckets[cellY * GRID_SIZE + cellX].push(index);
    });
    spatialRef.current = buckets;
  }, [mapData]);

  const getMetrics = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !mapData) return null;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const { minX, maxX, minY, maxY } = mapData.bounds;
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const baseScale = Math.min((width * 0.88) / spanX, (height * 0.86) / spanY);
    return {
      width,
      height,
      minX,
      minY,
      maxX,
      maxY,
      spanX,
      spanY,
      centerX: (minX + maxX) / 2,
      centerY: (minY + maxY) / 2,
      baseScale,
    };
  }, [mapData]);

  const toScreen = useCallback(
    (x: number, y: number, currentView = view) => {
      const metrics = getMetrics();
      if (!metrics) return { x: 0, y: 0 };
      return {
        x:
          (x - metrics.centerX) * metrics.baseScale * currentView.zoom +
          metrics.width / 2 +
          currentView.panX,
        y:
          -(y - metrics.centerY) * metrics.baseScale * currentView.zoom +
          metrics.height / 2 +
          currentView.panY,
      };
    },
    [getMetrics, view],
  );

  const drawMap = useCallback(() => {
    const canvas = canvasRef.current;
    const metrics = getMetrics();
    if (!canvas || !metrics || !mapData) return;

    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const targetWidth = Math.round(metrics.width * ratio);
    const targetHeight = Math.round(metrics.height * ratio);
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
    }
    const context = canvas.getContext('2d');
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, metrics.width, metrics.height);

    const pointSize = clamp(1.1 + Math.log2(view.zoom + 1) * 0.55, 1.15, 4.4);
    const orderedGroups = [...pointGroups.entries()].sort(([left], [right]) => left - right);

    for (const [clusterId, indices] of orderedGroups) {
      if (clusterId < 0 && !showNoise) continue;
      const isActive = activeCluster === null || activeCluster === clusterId;
      context.globalAlpha = clusterId < 0 ? (isActive ? 0.18 : 0.035) : isActive ? 0.78 : 0.055;
      context.fillStyle = clusterColor(clusterId);
      for (const index of indices) {
        const point = mapData.points[index];
        const screenX =
          (point[0] - metrics.centerX) * metrics.baseScale * view.zoom +
          metrics.width / 2 +
          view.panX;
        const screenY =
          -(point[1] - metrics.centerY) * metrics.baseScale * view.zoom +
          metrics.height / 2 +
          view.panY;
        if (
          screenX < -4 ||
          screenY < -4 ||
          screenX > metrics.width + 4 ||
          screenY > metrics.height + 4
        )
          continue;
        context.fillRect(screenX - pointSize / 2, screenY - pointSize / 2, pointSize, pointSize);
      }
    }

    const emphasized = selectedIndex ?? hoveredIndex;
    if (emphasized !== null) {
      const point = mapData.points[emphasized];
      if (point) {
        const position = toScreen(point[0], point[1]);
        context.globalAlpha = 1;
        context.beginPath();
        context.arc(position.x, position.y, selectedIndex === emphasized ? 7 : 5, 0, Math.PI * 2);
        context.fillStyle = clusterColor(point[2]);
        context.fill();
        context.lineWidth = 2;
        context.strokeStyle = '#ffffff';
        context.stroke();
      }
    }
    context.globalAlpha = 1;
  }, [activeCluster, getMetrics, hoveredIndex, mapData, pointGroups, selectedIndex, showNoise, toScreen, view]);

  useEffect(() => {
    if (!mapData) return;
    const frame = requestAnimationFrame(drawMap);
    return () => cancelAnimationFrame(frame);
  }, [drawMap, mapData]);

  useEffect(() => {
    const shell = mapShellRef.current;
    if (!shell) return;
    const observer = new ResizeObserver(() => drawMap());
    observer.observe(shell);
    return () => observer.disconnect();
  }, [drawMap]);

  useEffect(() => {
    if (selectedIndex === null) return;
    let cancelled = false;
    const chunkIndex = Math.floor(selectedIndex / DETAIL_CHUNK_SIZE);
    const cached = detailCacheRef.current.get(chunkIndex);
    const selectFromChunk = (chunk: DetailRow[]) => {
      if (!cancelled) {
        setSelectedDetail(chunk[selectedIndex % DETAIL_CHUNK_SIZE] ?? null);
        setDetailLoading(false);
      }
    };
    if (cached) {
      selectFromChunk(cached);
      return;
    }
    setDetailLoading(true);
    fetch(`/data/details/${String(chunkIndex).padStart(3, '0')}.json`)
      .then((response) => response.json() as Promise<DetailRow[]>)
      .then((chunk) => {
        detailCacheRef.current.set(chunkIndex, chunk);
        selectFromChunk(chunk);
      })
      .catch(() => {
        if (!cancelled) setDetailLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedIndex]);

  const searchResults = useMemo(() => {
    if (!searchData || deferredQuery.length < 2) return [];
    const results: number[] = [];
    for (let index = 0; index < searchData.length && results.length < 8; index += 1) {
      const row = searchData[index];
      if (
        row[0].toLowerCase().includes(deferredQuery) ||
        row[1].toLowerCase().includes(deferredQuery) ||
        row[2].toLowerCase().includes(deferredQuery)
      ) {
        results.push(index);
      }
    }
    return results;
  }, [deferredQuery, searchData]);

  const pickPoint = useCallback(
    (screenX: number, screenY: number): number | null => {
      const metrics = getMetrics();
      if (!metrics || !mapData) return null;
      const rawX =
        (screenX - metrics.width / 2 - view.panX) /
          (metrics.baseScale * view.zoom) +
        metrics.centerX;
      const rawY =
        -(screenY - metrics.height / 2 - view.panY) /
          (metrics.baseScale * view.zoom) +
        metrics.centerY;
      const cellX = clamp(Math.floor(((rawX - metrics.minX) / metrics.spanX) * GRID_SIZE), 0, GRID_SIZE - 1);
      const cellY = clamp(Math.floor(((rawY - metrics.minY) / metrics.spanY) * GRID_SIZE), 0, GRID_SIZE - 1);
      const cellWidth = (metrics.spanX / GRID_SIZE) * metrics.baseScale * view.zoom;
      const cellHeight = (metrics.spanY / GRID_SIZE) * metrics.baseScale * view.zoom;
      const radiusCells = clamp(Math.ceil(11 / Math.max(1, Math.min(cellWidth, cellHeight))), 1, 7);
      let nearest: number | null = null;
      let nearestDistance = 11 * 11;
      for (let y = cellY - radiusCells; y <= cellY + radiusCells; y += 1) {
        if (y < 0 || y >= GRID_SIZE) continue;
        for (let x = cellX - radiusCells; x <= cellX + radiusCells; x += 1) {
          if (x < 0 || x >= GRID_SIZE) continue;
          for (const index of spatialRef.current[y * GRID_SIZE + x] ?? []) {
            const point = mapData.points[index];
            if (point[2] < 0 && !showNoise) continue;
            if (activeCluster !== null && point[2] !== activeCluster) continue;
            const position = toScreen(point[0], point[1]);
            const distance = (position.x - screenX) ** 2 + (position.y - screenY) ** 2;
            if (distance < nearestDistance) {
              nearestDistance = distance;
              nearest = index;
            }
          }
        }
      }
      return nearest;
    },
    [activeCluster, getMetrics, mapData, showNoise, toScreen, view],
  );

  const focusPoint = useCallback(
    (index: number) => {
      if (!mapData) return;
      const metrics = getMetrics();
      const point = mapData.points[index];
      if (!metrics || !point) return;
      const zoom = Math.max(view.zoom, 6);
      setView({
        zoom,
        panX: -(point[0] - metrics.centerX) * metrics.baseScale * zoom,
        panY: (point[1] - metrics.centerY) * metrics.baseScale * zoom,
      });
      setSelectedIndex(index);
      setActiveCluster(point[2] >= 0 ? point[2] : null);
      setShowInspector(true);
      setQuery('');
    },
    [getMetrics, mapData, view.zoom],
  );

  const handleSearchResultClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      const index = Number(event.currentTarget.dataset.paperIndex);
      if (Number.isInteger(index)) focusPoint(index);
    },
    [focusPoint],
  );

  const focusCluster = useCallback(
    (clusterId: number | null) => {
      setActiveCluster(clusterId);
      setSelectedIndex(null);
      setShowLegend(false);
      if (clusterId === null || !mapData) {
        setView({ zoom: 1, panX: 0, panY: 0 });
        return;
      }
      const indices = pointGroups.get(clusterId) ?? [];
      const metrics = getMetrics();
      if (!indices.length || !metrics) return;
      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      for (const index of indices) {
        const point = mapData.points[index];
        minX = Math.min(minX, point[0]);
        maxX = Math.max(maxX, point[0]);
        minY = Math.min(minY, point[1]);
        maxY = Math.max(maxY, point[1]);
      }
      const spanX = Math.max(maxX - minX, metrics.spanX * 0.02);
      const spanY = Math.max(maxY - minY, metrics.spanY * 0.02);
      const zoom = clamp(
        Math.min((metrics.width * 0.7) / (spanX * metrics.baseScale), (metrics.height * 0.68) / (spanY * metrics.baseScale)),
        1.25,
        16,
      );
      const centerX = (minX + maxX) / 2;
      const centerY = (minY + maxY) / 2;
      setView({
        zoom,
        panX: -(centerX - metrics.centerX) * metrics.baseScale * zoom,
        panY: (centerY - metrics.centerY) * metrics.baseScale * zoom,
      });
    },
    [getMetrics, mapData, pointGroups],
  );

  const selectedPoint = selectedIndex === null ? null : mapData?.points[selectedIndex] ?? null;
  const selectedCluster = selectedPoint ? clusterById.get(selectedPoint[2]) : activeCluster === null ? null : clusterById.get(activeCluster);
  const hoveredRow = hoveredIndex === null ? null : searchData?.[hoveredIndex] ?? null;
  const hoveredPoint = hoveredIndex === null ? null : mapData?.points[hoveredIndex] ?? null;
  const hoveredCluster = hoveredPoint ? clusterById.get(hoveredPoint[2]) : null;

  if (loadError) {
    return (
      <main className="fatal-state">
        <p className="eyebrow">PhilPapers Atlas</p>
        <h1>The map could not be opened</h1>
        <p>{loadError}</p>
      </main>
    );
  }

  return (
    <main className="atlas-app">
      <header className="atlas-header">
        <button className="atlas-brand" onClick={() => focusCluster(null)} aria-label="Reset the PhilPapers Atlas">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>
            <strong>PhilPapers Atlas</strong>
            <small>Candidate map · 42 clusters</small>
          </span>
        </button>

        <div className="atlas-search-wrap">
          <label className="atlas-search">
            <span aria-hidden="true">⌕</span>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={searchData ? 'Search title, author, or paper ID' : 'Preparing paper search…'}
              disabled={!searchData}
              aria-label="Search papers"
            />
            {query && <button onClick={() => setQuery('')} aria-label="Clear search">×</button>}
          </label>
          {deferredQuery.length >= 2 && (
            <div className="search-results" role="listbox">
              {searchResults.length ? (
                searchResults.map((index) => {
                  const row = searchData![index];
                  return (
                    <button
                      key={row[0]}
                      data-paper-index={index}
                      onClick={handleSearchResultClick}
                      role="option"
                      aria-selected={selectedIndex === index}
                    >
                      <strong>{row[1]}</strong>
                      <span>{[row[2], row[3]].filter(Boolean).join(' · ') || row[0]}</span>
                    </button>
                  );
                })
              ) : (
                <p>No matching papers</p>
              )}
            </div>
          )}
        </div>

        <div className="header-actions">
          <button className="mobile-panel-button" onClick={() => setShowLegend(true)}>Clusters</button>
          <button className="text-button" onClick={() => setShowMethod(true)}>Method</button>
        </div>
      </header>

      <section className="atlas-workspace">
        <aside className={`cluster-panel ${showLegend ? 'panel-open' : ''}`}>
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Explore by topic</p>
              <h2>Clusters</h2>
            </div>
            <button className="panel-close" onClick={() => setShowLegend(false)} aria-label="Close cluster panel">×</button>
          </div>
          <button
            className={`cluster-row all-clusters ${activeCluster === null ? 'active' : ''}`}
            onClick={() => focusCluster(null)}
          >
            <span className="cluster-swatch constellation-swatch" />
            <span><strong>All philosophical areas</strong><small>{mapData ? formatNumber(mapData.count) : '—'} papers</small></span>
          </button>
          <div className="cluster-list">
            {clusterData?.clusters
              .slice()
              .sort((left, right) => right.count - left.count)
              .map((cluster) => (
                <button
                  key={cluster.id}
                  className={`cluster-row ${activeCluster === cluster.id ? 'active' : ''}`}
                  onClick={() => focusCluster(cluster.id)}
                  aria-pressed={activeCluster === cluster.id}
                >
                  <span className="cluster-swatch" style={{ background: clusterColor(cluster.id) }} />
                  <span><strong>{cluster.label}</strong><small>{formatNumber(cluster.count)} papers</small></span>
                </button>
              ))}
          </div>
          <label className="noise-toggle">
            <input type="checkbox" checked={showNoise} onChange={(event) => setShowNoise(event.target.checked)} />
            <span />
            Show {clusterData ? formatNumber(clusterData.noiseCount) : '—'} unclustered papers
          </label>
        </aside>

        <div className="map-shell" ref={mapShellRef}>
          {!mapData && <Loader />}
          <canvas
            ref={canvasRef}
            className="atlas-canvas"
            role="application"
            tabIndex={0}
            aria-label="Interactive two-dimensional UMAP of PhilPapers. Drag to pan, scroll to zoom, and select a point to inspect a paper."
            onPointerDown={(event) => {
              event.currentTarget.setPointerCapture(event.pointerId);
              dragRef.current = {
                pointerId: event.pointerId,
                x: event.clientX,
                y: event.clientY,
                originX: view.panX,
                originY: view.panY,
                moved: false,
              };
            }}
            onPointerMove={(event) => {
              const rect = event.currentTarget.getBoundingClientRect();
              const localX = event.clientX - rect.left;
              const localY = event.clientY - rect.top;
              const drag = dragRef.current;
              if (drag) {
                const deltaX = event.clientX - drag.x;
                const deltaY = event.clientY - drag.y;
                if (Math.abs(deltaX) + Math.abs(deltaY) > 3) drag.moved = true;
                setView((current) => ({ ...current, panX: drag.originX + deltaX, panY: drag.originY + deltaY }));
                setHoveredIndex(null);
                return;
              }
              const picked = pickPoint(localX, localY);
              setHoveredIndex(picked);
              setHoverPosition({ x: localX, y: localY });
            }}
            onPointerUp={(event) => {
              const drag = dragRef.current;
              dragRef.current = null;
              if (drag && !drag.moved && hoveredIndex !== null) {
                setSelectedIndex(hoveredIndex);
                setShowInspector(true);
              }
              event.currentTarget.releasePointerCapture(event.pointerId);
            }}
            onPointerCancel={() => {
              dragRef.current = null;
            }}
            onPointerLeave={() => {
              if (!dragRef.current) setHoveredIndex(null);
            }}
            onWheel={(event) => {
              event.preventDefault();
              const rect = event.currentTarget.getBoundingClientRect();
              const pointerX = event.clientX - rect.left;
              const pointerY = event.clientY - rect.top;
              const metrics = getMetrics();
              if (!metrics) return;
              setView((current) => {
                const nextZoom = clamp(current.zoom * Math.exp(-event.deltaY * 0.0012), 0.65, 28);
                const ratio = nextZoom / current.zoom;
                return {
                  zoom: nextZoom,
                  panX: pointerX - metrics.width / 2 - (pointerX - metrics.width / 2 - current.panX) * ratio,
                  panY: pointerY - metrics.height / 2 - (pointerY - metrics.height / 2 - current.panY) * ratio,
                };
              });
            }}
          />

          {hoveredIndex !== null && hoveredPoint && (
            <div className="paper-tooltip" style={{ left: hoverPosition.x, top: hoverPosition.y }}>
              <span style={{ background: clusterColor(hoveredPoint[2]) }} />
              <div>
                <strong>{hoveredRow?.[1] ?? `Paper ${hoveredIndex + 1}`}</strong>
                <small>{hoveredCluster?.label ?? 'Unclustered'}</small>
              </div>
            </div>
          )}

          <div className="map-caption">
            <p className="eyebrow">Two-dimensional UMAP</p>
            <span>{activeCluster === null ? 'All papers' : clusterById.get(activeCluster)?.label}</span>
          </div>

          <div className="map-controls" aria-label="Map controls">
            <button onClick={() => setView((current) => ({ ...current, zoom: clamp(current.zoom * 1.45, 0.65, 28) }))} aria-label="Zoom in">+</button>
            <button onClick={() => setView((current) => ({ ...current, zoom: clamp(current.zoom / 1.45, 0.65, 28) }))} aria-label="Zoom out">−</button>
            <button className="reset-control" onClick={() => focusCluster(null)}>Reset</button>
          </div>
          <div className="zoom-readout">{Math.round(view.zoom * 100)}%</div>
          <button className="mobile-inspector-button" onClick={() => setShowInspector(true)}>Details</button>
        </div>

        <aside className={`paper-panel ${showInspector ? 'panel-open' : ''}`}>
          <div className="panel-heading inspector-heading">
            <div>
              <p className="eyebrow">{selectedDetail ? 'Selected paper' : activeCluster !== null ? 'Selected cluster' : 'About this map'}</p>
              <h2>{selectedDetail ? 'Paper details' : activeCluster !== null ? 'Cluster profile' : 'A map of philosophy'}</h2>
            </div>
            <button className="panel-close" onClick={() => setShowInspector(false)} aria-label="Close details panel">×</button>
          </div>

          {selectedIndex !== null ? (
            detailLoading ? (
              <div className="detail-loading"><span /><p>Retrieving paper details…</p></div>
            ) : selectedDetail ? (
              <article className="paper-detail">
                <div className="detail-cluster-label">
                  <span style={{ background: clusterColor(selectedPoint?.[2] ?? -1) }} />
                  {selectedCluster?.label ?? 'Unclustered'}
                </div>
                <h3>{selectedDetail[1]}</h3>
                <p className="paper-authors">{selectedDetail[2] || 'Authorship not listed'}</p>
                <dl className="paper-meta">
                  <div><dt>Date</dt><dd>{selectedDetail[3] || 'Not listed'}</dd></div>
                  <div><dt>Paper ID</dt><dd>{selectedDetail[0]}</dd></div>
                </dl>
                <section className="abstract-section">
                  <p className="eyebrow">Abstract</p>
                  <p>{selectedDetail[5] || 'No abstract is available in the current PhilPapers metadata.'}</p>
                </section>
                <a className="primary-link" href={selectedDetail[4]} target="_blank" rel="noreferrer">
                  Open on PhilPapers <span aria-hidden="true">↗</span>
                </a>
                <button className="secondary-link" onClick={() => setSelectedIndex(null)}>Back to cluster overview</button>
              </article>
            ) : (
              <p className="empty-detail">Paper metadata could not be retrieved.</p>
            )
          ) : activeCluster !== null && selectedCluster ? (
            <article className="cluster-detail">
              <div className="cluster-orb" style={{ background: clusterColor(selectedCluster.id) }} />
              <h3>{selectedCluster.label}</h3>
              <p><strong>{formatNumber(selectedCluster.count)}</strong> papers in this candidate clustering.</p>
              <section>
                <p className="eyebrow">Characteristic terms</p>
                <div className="term-list">
                  {selectedCluster.terms.map((term) => <span key={term}>{term}</span>)}
                </div>
              </section>
              <p className="panel-note">Select any point to read its title, authorship, date, and abstract.</p>
              <button className="secondary-link" onClick={() => focusCluster(null)}>Return to the full atlas</button>
            </article>
          ) : (
            <article className="atlas-overview">
              <div className="overview-stat-grid">
                <div><strong>69,400</strong><span>English-language papers</span></div>
                <div><strong>42</strong><span>HDBSCAN clusters</span></div>
                <div><strong>48,119</strong><span>Clustered papers</span></div>
                <div><strong>30.7%</strong><span>Retained as noise</span></div>
              </div>
              <p>
                Each point is a paper. Position comes from the saved two-dimensional UMAP of the
                30-dimensional clustering representation; colour indicates Prajakta’s reviewed cluster label.
              </p>
              <div className="instruction-list">
                <span><b>01</b> Drag to travel through the literature.</span>
                <span><b>02</b> Scroll to zoom into dense regions.</span>
                <span><b>03</b> Select a paper to read its metadata.</span>
              </div>
              <p className="panel-note">This is the provisional 42-cluster candidate and can be updated without rebuilding the interface.</p>
            </article>
          )}
        </aside>
      </section>

      {showMethod && (
        <div className="method-backdrop" role="dialog" aria-modal="true" aria-labelledby="method-title" onMouseDown={() => setShowMethod(false)}>
          <article className="method-card" onMouseDown={(event) => event.stopPropagation()}>
            <button className="method-close" onClick={() => setShowMethod(false)} aria-label="Close method description">×</button>
            <p className="eyebrow">How to read the map</p>
            <h2 id="method-title">A projection, not a flat taxonomy</h2>
            <p>
              The source papers were embedded with AllenAI SPECTER, reduced to 100 dimensions with PCA,
              then to 30 dimensions with UMAP. HDBSCAN identified the displayed clusters in that 30-dimensional space.
            </p>
            <div className="pipeline">
              <span><b>768D</b>SPECTER</span><i>→</i><span><b>100D</b>PCA</span><i>→</i><span><b>30D</b>UMAP</span><i>→</i><span><b>42</b>clusters</span><i>→</i><span><b>2D</b>display</span>
            </div>
            <p>
              The final two-dimensional UMAP is a visual aid. Local neighbourhoods are informative, but distances and
              apparent boundaries are distorted by projection. Grey papers were labelled as noise by HDBSCAN rather than
              forced into a topic.
            </p>
            <p className="method-caveat">Cluster membership strengths were not included in the saved preferred-run export, so this version reports the categorical assignment without inventing a confidence value.</p>
          </article>
        </div>
      )}
    </main>
  );
}
