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
type DetailRow = [string, string];

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

type WebGLRenderer = {
  gl: WebGL2RenderingContext;
  program: WebGLProgram;
  vertexArray: WebGLVertexArrayObject;
  positionBuffer: WebGLBuffer;
  colorBuffer: WebGLBuffer;
  clusterBuffer: WebGLBuffer;
  pointCount: number;
  uniforms: {
    center: WebGLUniformLocation;
    baseScale: WebGLUniformLocation;
    zoom: WebGLUniformLocation;
    pan: WebGLUniformLocation;
    viewport: WebGLUniformLocation;
    dpr: WebGLUniformLocation;
    pointSize: WebGLUniformLocation;
    activeCluster: WebGLUniformLocation;
    showNoise: WebGLUniformLocation;
  };
};

const GRID_SIZE = 128;
const DETAIL_CHUNK_SIZE = 200;

const CLUSTER_COLORS = [
  '#82aa3a', '#b262c2', '#5ab22a', '#2aaac2', '#009e73', '#e292ea', '#e26aba',
  '#aa822a', '#ea4292', '#ea6a8a', '#2a8ae2', '#d55e00', '#2ada82', '#ea523a',
  '#baaa6a', '#7a8a52', '#2a9a92', '#eab28a', '#8c72cb', '#e4d34f', '#5a7aea',
  '#5ad2d2', '#aaca82', '#d27a42', '#7a8ac2', '#92922a', '#eab262', '#a2c22a',
  '#56b4e9', '#da52ca', '#ba6a7a', '#ea425a', '#cc79a7', '#e69f00', '#c2c262',
  '#5a9242', '#2aaa4a', '#aa6aea', '#7ad2a2', '#a27a4a', '#82d272', '#ea8a82',
] as const;

function clusterColor(id: number): string {
  if (id < 0) return '#c7cbd1';
  return CLUSTER_COLORS[id % CLUSTER_COLORS.length];
}

function colorChannels(hex: string): [number, number, number] {
  return [
    Number.parseInt(hex.slice(1, 3), 16) / 255,
    Number.parseInt(hex.slice(3, 5), 16) / 255,
    Number.parseInt(hex.slice(5, 7), 16) / 255,
  ];
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('The graphics renderer could not be created.');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) || 'Unknown graphics error';
    gl.deleteShader(shader);
    throw new Error(`The graphics renderer could not be compiled: ${message}`);
  }
  return shader;
}

function requiredUniform(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  name: string,
): WebGLUniformLocation {
  const location = gl.getUniformLocation(program, name);
  if (!location) throw new Error(`The graphics renderer is missing ${name}.`);
  return location;
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
  const webglRef = useRef<WebGLRenderer | null>(null);
  const detailCacheRef = useRef<Map<number, DetailRow[]>>(new Map());
  const detailPromiseRef = useRef<Map<number, Promise<DetailRow[]>>>(new Map());

  const [mapData, setMapData] = useState<MapData | null>(null);
  const [clusterData, setClusterData] = useState<ClusterData | null>(null);
  const [searchData, setSearchData] = useState<SearchRow[] | null>(null);
  const [loadError, setLoadError] = useState('');
  const [viewport, setViewport] = useState({ width: 0, height: 0 });
  const [view, setView] = useState<View>({ zoom: 1, panX: 0, panY: 0 });
  const [activeCluster, setActiveCluster] = useState<number | null>(null);
  const [showNoise, setShowNoise] = useState(true);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [selectedSupplement, setSelectedSupplement] = useState<{
    index: number;
    row: DetailRow | null;
  } | null>(null);
  const [query, setQuery] = useState('');
  const [showMethod, setShowMethod] = useState(false);
  const [showLegend, setShowLegend] = useState(false);
  const [showInspector, setShowInspector] = useState(false);
  const deferredQuery = useDeferredValue(query.trim().toLowerCase());

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch('data/map.json').then((response) => {
        if (!response.ok) throw new Error('The saved map could not be loaded.');
        return response.json() as Promise<MapData>;
      }),
      fetch('data/clusters.json').then((response) => {
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

    fetch('data/search.json')
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
    if (!mapData || viewport.width <= 0 || viewport.height <= 0) return null;
    const { width, height } = viewport;
    const { minX, maxX, minY, maxY } = mapData.bounds;
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const baseScale = Math.min((width * 0.9) / spanX, (height * 0.9) / spanY);
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
  }, [mapData, viewport]);

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
    const renderer = webglRef.current;
    const metrics = getMetrics();
    if (!canvas || !renderer || !metrics) return;

    const { gl, uniforms } = renderer;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const targetWidth = Math.max(1, Math.round(metrics.width * ratio));
    const targetHeight = Math.max(1, Math.round(metrics.height * ratio));
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
    }

    gl.viewport(0, 0, targetWidth, targetHeight);
    gl.clearColor(18 / 255, 18 / 255, 26 / 255, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(renderer.program);
    gl.bindVertexArray(renderer.vertexArray);
    gl.uniform2f(uniforms.center, metrics.centerX, metrics.centerY);
    gl.uniform1f(uniforms.baseScale, metrics.baseScale);
    gl.uniform1f(uniforms.zoom, view.zoom);
    gl.uniform2f(uniforms.pan, view.panX, view.panY);
    gl.uniform2f(uniforms.viewport, metrics.width, metrics.height);
    gl.uniform1f(uniforms.dpr, ratio);
    const pointDiameter = clamp(0.72 + Math.log2(view.zoom + 1) * 0.28, 0.95, 2.7) * 2;
    gl.uniform1f(uniforms.pointSize, pointDiameter);
    gl.uniform1f(uniforms.activeCluster, activeCluster ?? -2);
    gl.uniform1f(uniforms.showNoise, showNoise ? 1 : 0);
    gl.drawArrays(gl.POINTS, 0, renderer.pointCount);
    gl.bindVertexArray(null);
  }, [activeCluster, getMetrics, showNoise, view]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !mapData) return;

    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
    });
    if (!gl) {
      setLoadError('This map requires a browser with WebGL 2 support.');
      return;
    }

    try {
      const vertexShader = compileShader(
        gl,
        gl.VERTEX_SHADER,
        `#version 300 es
        precision highp float;
        layout(location = 0) in vec2 aPosition;
        layout(location = 1) in vec3 aColor;
        layout(location = 2) in float aCluster;
        uniform vec2 uCenter;
        uniform float uBaseScale;
        uniform float uZoom;
        uniform vec2 uPan;
        uniform vec2 uViewport;
        uniform float uDpr;
        uniform float uPointSize;
        uniform float uActiveCluster;
        uniform float uShowNoise;
        out vec3 vColor;
        out float vAlpha;

        void main() {
          bool allClusters = uActiveCluster < -1.5;
          bool isNoise = aCluster < -0.5;
          if (isNoise) {
            vAlpha = uShowNoise > 0.5 ? (allClusters ? 0.50 : 0.085) : 0.0;
          } else {
            vAlpha = allClusters || abs(aCluster - uActiveCluster) < 0.1 ? 0.92 : 0.085;
          }
          if (vAlpha <= 0.0) {
            gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
            gl_PointSize = 0.0;
            vColor = aColor;
            return;
          }
          float x = (aPosition.x - uCenter.x) * uBaseScale * uZoom + uViewport.x * 0.5 + uPan.x;
          float y = -(aPosition.y - uCenter.y) * uBaseScale * uZoom + uViewport.y * 0.5 + uPan.y;
          gl_Position = vec4(x / uViewport.x * 2.0 - 1.0, 1.0 - y / uViewport.y * 2.0, 0.0, 1.0);
          gl_PointSize = uPointSize * uDpr;
          vColor = aColor;
        }`,
      );
      const fragmentShader = compileShader(
        gl,
        gl.FRAGMENT_SHADER,
        `#version 300 es
        precision highp float;
        in vec3 vColor;
        in float vAlpha;
        out vec4 outColor;

        void main() {
          vec2 point = gl_PointCoord * 2.0 - 1.0;
          float radius = dot(point, point);
          if (radius > 1.0) discard;
          float coverage = 1.0 - smoothstep(0.48, 1.0, radius);
          outColor = vec4(vColor, vAlpha * coverage);
        }`,
      );
      const program = gl.createProgram();
      if (!program) throw new Error('The graphics renderer could not be created.');
      gl.attachShader(program, vertexShader);
      gl.attachShader(program, fragmentShader);
      gl.linkProgram(program);
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const message = gl.getProgramInfoLog(program) || 'Unknown graphics error';
        gl.deleteProgram(program);
        throw new Error(`The graphics renderer could not be linked: ${message}`);
      }

      const positions = new Float32Array(mapData.points.length * 2);
      const colors = new Float32Array(mapData.points.length * 3);
      const clusters = new Float32Array(mapData.points.length);
      mapData.points.forEach((point, index) => {
        positions[index * 2] = point[0];
        positions[index * 2 + 1] = point[1];
        clusters[index] = point[2];
        const color = colorChannels(clusterColor(point[2]));
        colors[index * 3] = color[0];
        colors[index * 3 + 1] = color[1];
        colors[index * 3 + 2] = color[2];
      });

      const vertexArray = gl.createVertexArray();
      const positionBuffer = gl.createBuffer();
      const colorBuffer = gl.createBuffer();
      const clusterBuffer = gl.createBuffer();
      if (!vertexArray || !positionBuffer || !colorBuffer || !clusterBuffer) {
        throw new Error('The graphics buffers could not be created.');
      }
      gl.bindVertexArray(vertexArray);
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, clusterBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, clusters, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.disable(gl.DEPTH_TEST);

      const renderer: WebGLRenderer = {
        gl,
        program,
        vertexArray,
        positionBuffer,
        colorBuffer,
        clusterBuffer,
        pointCount: mapData.points.length,
        uniforms: {
          center: requiredUniform(gl, program, 'uCenter'),
          baseScale: requiredUniform(gl, program, 'uBaseScale'),
          zoom: requiredUniform(gl, program, 'uZoom'),
          pan: requiredUniform(gl, program, 'uPan'),
          viewport: requiredUniform(gl, program, 'uViewport'),
          dpr: requiredUniform(gl, program, 'uDpr'),
          pointSize: requiredUniform(gl, program, 'uPointSize'),
          activeCluster: requiredUniform(gl, program, 'uActiveCluster'),
          showNoise: requiredUniform(gl, program, 'uShowNoise'),
        },
      };
      webglRef.current = renderer;
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : 'The graphics renderer could not be prepared.');
      return;
    }

    return () => {
      const renderer = webglRef.current;
      if (!renderer || renderer.gl !== gl) return;
      gl.deleteBuffer(renderer.positionBuffer);
      gl.deleteBuffer(renderer.colorBuffer);
      gl.deleteBuffer(renderer.clusterBuffer);
      gl.deleteVertexArray(renderer.vertexArray);
      gl.deleteProgram(renderer.program);
      webglRef.current = null;
    };
  }, [mapData]);

  useEffect(() => {
    if (!mapData) return;
    const frame = requestAnimationFrame(drawMap);
    return () => cancelAnimationFrame(frame);
  }, [drawMap, mapData]);

  useEffect(() => {
    const shell = mapShellRef.current;
    if (!shell) return;
    const observer = new ResizeObserver(([entry]) => {
      const width = entry?.contentRect.width ?? shell.clientWidth;
      const height = entry?.contentRect.height ?? shell.clientHeight;
      setViewport((current) =>
        current.width === width && current.height === height ? current : { width, height },
      );
    });
    observer.observe(shell);
    return () => observer.disconnect();
  }, []);

  const loadDetailChunk = useCallback((chunkIndex: number): Promise<DetailRow[]> => {
    const cached = detailCacheRef.current.get(chunkIndex);
    if (cached) return Promise.resolve(cached);
    const pending = detailPromiseRef.current.get(chunkIndex);
    if (pending) return pending;

    const request = fetch(`data/details/${String(chunkIndex).padStart(3, '0')}.json`)
      .then((response) => {
        if (!response.ok) throw new Error('Paper details unavailable');
        return response.json() as Promise<DetailRow[]>;
      })
      .then(
        (chunk) => {
          detailCacheRef.current.set(chunkIndex, chunk);
          detailPromiseRef.current.delete(chunkIndex);
          return chunk;
        },
        (error) => {
          detailPromiseRef.current.delete(chunkIndex);
          throw error;
        },
      );
    detailPromiseRef.current.set(chunkIndex, request);
    return request;
  }, []);

  useEffect(() => {
    if (hoveredIndex === null) return;
    const chunkIndex = Math.floor(hoveredIndex / DETAIL_CHUNK_SIZE);
    const timer = window.setTimeout(() => {
      void loadDetailChunk(chunkIndex).catch(() => undefined);
    }, 120);
    return () => window.clearTimeout(timer);
  }, [hoveredIndex, loadDetailChunk]);

  useEffect(() => {
    if (selectedIndex === null) return;
    let cancelled = false;
    const chunkIndex = Math.floor(selectedIndex / DETAIL_CHUNK_SIZE);
    loadDetailChunk(chunkIndex)
      .then((chunk) => {
        if (!cancelled) {
          setSelectedSupplement({
            index: selectedIndex,
            row: chunk[selectedIndex % DETAIL_CHUNK_SIZE] ?? null,
          });
        }
      })
      .catch(() => {
        if (!cancelled) setSelectedSupplement({ index: selectedIndex, row: null });
      });
    return () => {
      cancelled = true;
    };
  }, [loadDetailChunk, selectedIndex]);

  const searchableRows = useMemo(
    () => searchData?.map((row) => `${row[0]}\n${row[1]}`.toLocaleLowerCase()) ?? null,
    [searchData],
  );

  const searchResults = useMemo(() => {
    if (!searchData || !searchableRows || deferredQuery.length < 2) return [];
    const results: number[] = [];
    for (let index = 0; index < searchData.length && results.length < 8; index += 1) {
      if (searchableRows[index].includes(deferredQuery)) results.push(index);
    }
    return results;
  }, [deferredQuery, searchData, searchableRows]);

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
  const selectedSearch = selectedIndex === null ? null : searchData?.[selectedIndex] ?? null;
  const selectedDetail = selectedIndex !== null && selectedSupplement?.index === selectedIndex
    ? selectedSupplement.row
    : null;
  const detailLoading = selectedIndex !== null && selectedSupplement?.index !== selectedIndex;
  const hoveredRow = hoveredIndex === null ? null : searchData?.[hoveredIndex] ?? null;
  const hoveredPoint = hoveredIndex === null ? null : mapData?.points[hoveredIndex] ?? null;
  const hoveredCluster = hoveredPoint ? clusterById.get(hoveredPoint[2]) : null;
  const selectedMarker = selectedPoint ? toScreen(selectedPoint[0], selectedPoint[1]) : null;
  const hoveredMarker = hoveredPoint ? toScreen(hoveredPoint[0], hoveredPoint[1]) : null;

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
          </span>
        </button>

        <div className="atlas-search-wrap">
          <label className="atlas-search">
            <span aria-hidden="true">⌕</span>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={searchData ? 'Search title or author' : 'Preparing paper search…'}
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
                       key={index}
                      data-paper-index={index}
                      onClick={handleSearchResultClick}
                      role="option"
                      aria-selected={selectedIndex === index}
                    >
                      <strong>{row[0]}</strong>
                      <span>{[row[1], row[2]].filter(Boolean).join(' · ') || 'Metadata unavailable'}</span>
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
            <h2>Clusters <span>{clusterData?.clusters.length ?? '—'}</span></h2>
            <button className="panel-close" onClick={() => setShowLegend(false)} aria-label="Close cluster panel">×</button>
          </div>
          <button
            className={`cluster-row all-clusters ${activeCluster === null ? 'active' : ''}`}
            onClick={() => focusCluster(null)}
          >
            <span className="cluster-swatch constellation-swatch" />
            <span><strong>All papers</strong></span>
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
                  title={`${cluster.label} — ${formatNumber(cluster.count)} papers`}
                >
                  <span className="cluster-swatch" style={{ background: clusterColor(cluster.id) }} />
                  <span><strong>{cluster.label}</strong></span>
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
              if (drag && !drag.moved) {
                const rect = event.currentTarget.getBoundingClientRect();
                const picked = pickPoint(event.clientX - rect.left, event.clientY - rect.top);
                if (picked !== null) {
                  setSelectedIndex(picked);
                  setHoveredIndex(picked);
                  setShowInspector(true);
                }
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

          {selectedMarker && selectedPoint && (
            <span
              className="point-marker selected-marker"
              style={{
                left: selectedMarker.x,
                top: selectedMarker.y,
                background: clusterColor(selectedPoint[2]),
              }}
              aria-hidden="true"
            />
          )}
          {hoveredMarker && hoveredPoint && hoveredIndex !== selectedIndex && (
            <span
              className="point-marker hover-marker"
              style={{
                left: hoveredMarker.x,
                top: hoveredMarker.y,
                background: clusterColor(hoveredPoint[2]),
              }}
              aria-hidden="true"
            />
          )}

          {hoveredIndex !== null && hoveredPoint && (
            <div className="paper-tooltip" style={{ left: hoverPosition.x, top: hoverPosition.y }}>
              <span style={{ background: clusterColor(hoveredPoint[2]) }} />
              <div>
                <strong>{hoveredRow?.[0] ?? `Paper ${hoveredIndex + 1}`}</strong>
                <small>{hoveredCluster?.label ?? 'Unclustered'}</small>
              </div>
            </div>
          )}

          <div className="map-controls" aria-label="Map controls">
            <button onClick={() => setView((current) => ({ ...current, zoom: clamp(current.zoom * 1.45, 0.65, 28) }))} aria-label="Zoom in">+</button>
            <button onClick={() => setView((current) => ({ ...current, zoom: clamp(current.zoom / 1.45, 0.65, 28) }))} aria-label="Zoom out">−</button>
            <button className="reset-control" onClick={() => focusCluster(null)}>Reset</button>
          </div>
          <button className="mobile-inspector-button" onClick={() => setShowInspector(true)}>Details</button>
        </div>

        <aside className={`paper-panel ${showInspector ? 'panel-open' : ''}`}>
          <button className="panel-close inspector-close" onClick={() => setShowInspector(false)} aria-label="Close details panel">×</button>

          {selectedIndex !== null ? (
            <article className="paper-detail">
              <div className="detail-cluster-label">
                <span style={{ background: clusterColor(selectedPoint?.[2] ?? -1) }} />
                {selectedCluster?.label ?? 'Unclustered'}
              </div>
              <h3>{selectedSearch?.[0] ?? 'Paper details'}</h3>
              <p className="paper-authors">{selectedSearch?.[1] || 'Authorship not listed'}</p>
              <dl className="paper-meta">
                <div><dt>Date</dt><dd>{selectedSearch?.[2] || 'Not listed'}</dd></div>
              </dl>
              <section className="abstract-section">
                <p className="eyebrow">Abstract</p>
                {detailLoading ? (
                  <div className="abstract-loading" role="status" aria-live="polite">
                    <span />
                    <p>Loading abstract…</p>
                  </div>
                ) : (
                  <p>
                    {selectedDetail?.[0] ||
                      (selectedSupplement?.row === null
                        ? 'The abstract could not be loaded.'
                        : 'No abstract is available in the current PhilPapers metadata.')}
                  </p>
                )}
              </section>
              {selectedSearch?.[3] && (
                <a className="primary-link" href={selectedSearch[3]} target="_blank" rel="noreferrer">
                  Open on PhilPapers <span aria-hidden="true">↗</span>
                </a>
              )}
              <button className="secondary-link" onClick={() => setSelectedIndex(null)}>Back to cluster overview</button>
            </article>
          ) : activeCluster !== null && selectedCluster ? (
            <article className="cluster-detail">
              <div className="cluster-orb" style={{ background: clusterColor(selectedCluster.id) }} />
              <h3>{selectedCluster.label}</h3>
              <p><strong>{formatNumber(selectedCluster.count)}</strong> papers assigned to this cluster.</p>
              <section>
                <p className="eyebrow">Characteristic terms</p>
                <div className="term-list">
                  {selectedCluster.terms.map((term) => <span key={term}>{term}</span>)}
                </div>
              </section>
              <button className="secondary-link" onClick={() => focusCluster(null)}>Return to the full atlas</button>
            </article>
          ) : (
            <article className="atlas-overview">
              <dl className="overview-stats">
                <div><dt>English-language papers</dt><dd>69,400</dd></div>
                <div><dt>HDBSCAN clusters</dt><dd>42</dd></div>
                <div><dt>Clustered papers</dt><dd>48,119</dd></div>
                <div><dt>Unassigned as noise</dt><dd>30.7%</dd></div>
              </dl>
              <p>
                Each point represents one paper. Position comes from the saved projection; colour indicates
                HDBSCAN cluster assignment.
              </p>
            </article>
          )}
        </aside>
      </section>

      {showMethod && (
        <div className="method-backdrop" role="dialog" aria-modal="true" aria-labelledby="method-title" onMouseDown={() => setShowMethod(false)}>
          <article className="method-card" onMouseDown={(event) => event.stopPropagation()}>
            <button className="method-close" onClick={() => setShowMethod(false)} aria-label="Close method description">×</button>
            <h2 id="method-title">Methods</h2>
            <p>
              Each paper is represented by a 768-dimensional SPECTER embedding. The embeddings were reduced
              to 100 dimensions by PCA, then to 30 dimensions by UMAP using cosine distance, 15 neighbours and
              a minimum distance of 0. We then fitted HDBSCAN in this 30-dimensional space, setting the minimum
              cluster size to 200 and the minimum number of samples to 15. This produced 42 clusters and left
              21,281 papers unassigned.
            </p>
            <p>
              The displayed coordinates come from a separate two-dimensional UMAP of the 30-dimensional representation,
              using 15 neighbours and a minimum distance of 0.1. The projection is intended for visual exploration.
              Nearby points are often semantically related, but the map does not preserve every high-dimensional distance,
              cluster shape or boundary relation.
            </p>
            <p>
              Grey points are papers that HDBSCAN treated as noise; they were not assigned to a cluster. The saved export
              contains categorical assignments but not membership probabilities, so confidence scores are not shown.
            </p>
          </article>
        </div>
      )}
    </main>
  );
}
