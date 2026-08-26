'use client';

import {
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import type {
  KeyboardEvent,
  MouseEvent,
  PointerEvent as ReactPointerEvent,
  WheelEvent,
} from 'react';
import {
  fitCameraToSphere,
  orbitCamera,
  panCamera,
  projectPoint,
  resetCamera3D,
  viewProjectionMatrix,
  zoomCamera,
} from './atlas3d';
import type { Camera3D, Vec3 } from './atlas3d';

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
  color: string;
};

type LensMethod = {
  pcaDimensions?: number;
  sourceUmapDimensions?: number;
  umapDimensions?: number;
  clusteringSpace?: 'display2d';
  minClusterSize?: number;
  minSamples?: number;
  selectionMethod?: 'eom' | 'leaf';
  k?: number;
  topicCount?: number;
  vocabularySize?: number;
  minDocumentFrequency?: number;
  maxDocumentFrequency?: number;
  ngramRange?: [number, number];
  docTopicPrior?: number;
  topicWordPrior?: number;
  priorVariant?: string;
  learningMethod?: 'online' | 'batch';
  iterations?: number;
};

type LensMetrics = {
  relativeValidity?: number;
  meanPersistence?: number;
  stabilityWeightedCoverage?: number;
  meanMembership?: number;
  silhouette2d?: number;
  parameterAgreementAri?: number;
  silhouetteCosine?: number;
  daviesBouldin?: number;
  calinskiHarabasz?: number;
  heldOutPerplexity?: number;
  npmiCoherence?: number;
  topicDiversity?: number;
  topicExclusivity?: number;
  meanDominantProbability?: number;
  topicStability?: number;
  assignmentStabilityAri?: number;
  finalScore?: number;
};

type Lens = {
  id: string;
  name: string;
  optionLabel: string;
  algorithm: 'hdbscan' | 'kmeans' | 'lda';
  preferred: boolean;
  labelsFile: string;
  clusterCount: number;
  noiseCount: number;
  noisePct: number;
  method: LensMethod;
  metrics: LensMetrics;
  clusters: Cluster[];
};

type LensCatalog = {
  version: string;
  paperCount: number;
  defaultLens: string;
  projection: {
    sourceLens: string;
    dimensions: number;
    nNeighbors: number;
    minDist: number;
    metric: string;
  };
  lenses: Lens[];
};

type View = { zoom: number; panX: number; panY: number };
type DisplayMode = '2d' | '3d';

type Projection3DMetadata = {
  version: string;
  paperCount: number;
  dimensions: 3;
  file: string;
  format: string;
  binarySha256: string;
  source: {
    embeddingDimensions: number;
    pcaDimensions: number;
    umapDimensions: number;
    umapNNeighbors: number;
    umapMinDist: number;
    umapMetric: string;
  };
  display: {
    nNeighbors: number;
    minDist: number;
    metric: string;
    epochs: number;
    randomState: number;
  };
  metrics: {
    trustworthiness15: number;
    neighbourRecall15: number;
    neighbourRecall50: number;
    distanceSpearman: number;
    seedNeighbourAgreement15: number;
  };
};

type Projection3D = {
  metadata: Projection3DMetadata;
  positions: Float32Array;
};

type PickTarget = {
  framebuffer: WebGLFramebuffer;
  texture: WebGLTexture;
  depth: WebGLRenderbuffer;
  width: number;
  height: number;
  scale: number;
};

type WebGLRenderer = {
  gl: WebGL2RenderingContext;
  program: WebGLProgram;
  vertexArray: WebGLVertexArrayObject;
  positionBuffer: WebGLBuffer;
  colorBuffer: WebGLBuffer;
  clusterBuffer: WebGLBuffer;
  positions2d: Float32Array;
  pickProgram: WebGLProgram;
  pickTarget: PickTarget | null;
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
    is3d: WebGLUniformLocation;
    viewProjection: WebGLUniformLocation;
    cameraDistance: WebGLUniformLocation;
  };
  pickUniforms: {
    viewProjection: WebGLUniformLocation;
    pointSize: WebGLUniformLocation;
    activeCluster: WebGLUniformLocation;
    showNoise: WebGLUniformLocation;
  };
};

const GRID_SIZE = 128;
const DETAIL_CHUNK_SIZE = 200;
const NOISE_COLOR = '#c7cbd1';

const CLUSTER_COLORS = [
  '#82aa3a', '#b262c2', '#5ab22a', '#2aaac2', '#009e73', '#e292ea', '#e26aba',
  '#aa822a', '#ea4292', '#ea6a8a', '#2a8ae2', '#d55e00', '#2ada82', '#ea523a',
  '#baaa6a', '#7a8a52', '#2a9a92', '#eab28a', '#8c72cb', '#e4d34f', '#5a7aea',
  '#5ad2d2', '#aaca82', '#d27a42', '#7a8ac2', '#92922a', '#eab262', '#a2c22a',
  '#56b4e9', '#da52ca', '#ba6a7a', '#ea425a', '#cc79a7', '#e69f00', '#c2c262',
  '#5a9242', '#2aaa4a', '#aa6aea', '#7ad2a2', '#a27a4a', '#82d272', '#ea8a82',
] as const;

function clusterColor(id: number, clusters?: Map<number, Cluster>): string {
  if (id < 0) return NOISE_COLOR;
  return clusters?.get(id)?.color ?? CLUSTER_COLORS[id % CLUSTER_COLORS.length];
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

function linkProgram(
  gl: WebGL2RenderingContext,
  vertexSource: string,
  fragmentSource: string,
): WebGLProgram {
  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
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
  return program;
}

function deletePickTarget(gl: WebGL2RenderingContext, target: PickTarget | null): void {
  if (!target) return;
  gl.deleteFramebuffer(target.framebuffer);
  gl.deleteTexture(target.texture);
  gl.deleteRenderbuffer(target.depth);
}

function ensurePickTarget(
  renderer: WebGLRenderer,
  cssWidth: number,
  cssHeight: number,
): PickTarget {
  const { gl } = renderer;
  const scale = Math.min(1, 2048 / Math.max(cssWidth, cssHeight, 1));
  const width = Math.max(1, Math.ceil(cssWidth * scale));
  const height = Math.max(1, Math.ceil(cssHeight * scale));
  const current = renderer.pickTarget;
  if (current && current.width === width && current.height === height) return current;
  deletePickTarget(gl, current);
  const framebuffer = gl.createFramebuffer();
  const texture = gl.createTexture();
  const depth = gl.createRenderbuffer();
  if (!framebuffer || !texture || !depth) throw new Error('The point picker could not be created.');
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
  gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, width, height);
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    deletePickTarget(gl, { framebuffer, texture, depth, width, height, scale });
    throw new Error('The point picker framebuffer is incomplete.');
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.bindRenderbuffer(gl.RENDERBUFFER, null);
  const target = { framebuffer, texture, depth, width, height, scale };
  renderer.pickTarget = target;
  return target;
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-GB').format(value);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function formatDecimal(value: number | undefined, digits = 3): string {
  return value === undefined ? '—' : value.toFixed(digits);
}

function MethodContent({ lens }: { lens: Lens }) {
  const method = lens.method;
  const usesDisplaySpace = method.clusteringSpace === 'display2d';
  const isLda = lens.algorithm === 'lda';
  return (
    <>
      <p className="method-heading">Parameters</p>
      <dl className="method-table">
        {isLda ? (
          <>
            <div><dt>Input</dt><dd>titles (×2) and abstracts</dd></div>
            <div><dt>Vocabulary</dt><dd>{formatNumber(method.vocabularySize ?? 0)} unigrams and bigrams · min. {method.minDocumentFrequency} papers</dd></div>
            <div><dt>LDA</dt><dd><i>k</i> = {method.topicCount} · {method.learningMethod} · {method.iterations} iterations</dd></div>
            <div><dt>Priors</dt><dd>α = {formatDecimal(method.docTopicPrior, 3)} · η = {formatDecimal(method.topicWordPrior, 3)}</dd></div>
          </>
        ) : (
          <>
            <div><dt>Input</dt><dd>SPECTER embeddings · 768D</dd></div>
            <div><dt>PCA</dt><dd>{method.pcaDimensions}D</dd></div>
            {lens.algorithm === 'hdbscan' ? (
              usesDisplaySpace ? (
                <>
                  <div><dt>UMAP preprocessing</dt><dd>{method.sourceUmapDimensions}D · cosine · 15 neighbours · min. distance 0</dd></div>
                  <div><dt>Clustering space</dt><dd>fixed 2D display UMAP · 15 neighbours · min. distance 0.1</dd></div>
                  <div><dt>HDBSCAN</dt><dd>min. cluster size {method.minClusterSize} · min. samples {method.minSamples} · {method.selectionMethod?.toUpperCase()}</dd></div>
                </>
              ) : (
                <>
                  <div><dt>UMAP</dt><dd>{method.umapDimensions}D · cosine · 15 neighbours · min. distance 0</dd></div>
                  <div><dt>HDBSCAN</dt><dd>min. cluster size {method.minClusterSize} · min. samples {method.minSamples}</dd></div>
                </>
              )
            ) : (
              <div><dt>K-means</dt><dd><i>k</i> = {method.k}</dd></div>
            )}
          </>
        )}
      </dl>
      <p className="method-heading metrics-heading">Metrics</p>
      <dl className="method-metrics">
        {isLda ? (
          <>
            <div title="NPMI coherence of the most probable words in held-out papers; higher is better."><dt>Held-out NPMI coherence</dt><dd>{formatDecimal(lens.metrics.npmiCoherence)}</dd></div>
            <div title="Per-word perplexity on the sealed test set; lower is better and comparisons are meaningful within these LDA models."><dt>Held-out perplexity</dt><dd>{lens.metrics.heldOutPerplexity === undefined ? '—' : formatNumber(Math.round(lens.metrics.heldOutPerplexity))}</dd></div>
            <div title="Mean cosine similarity of matched topic-word distributions across three random seeds; higher is better."><dt>Topic stability</dt><dd>{formatDecimal(lens.metrics.topicStability)}</dd></div>
            <div title="Mean adjusted Rand agreement between dominant-topic paper assignments across three random seeds; higher is better."><dt>Assignment agreement (ARI)</dt><dd>{formatDecimal(lens.metrics.assignmentStabilityAri)}</dd></div>
          </>
        ) : lens.algorithm === 'hdbscan' ? (
          usesDisplaySpace ? (
            <>
              <div title="Mean HDBSCAN membership strength among assigned papers; higher is better."><dt>Mean membership strength</dt><dd>{formatDecimal(lens.metrics.meanMembership)}</dd></div>
              <div title="Silhouette score among assigned papers in the two-dimensional clustering space; higher is better."><dt>Silhouette (2D)</dt><dd>{formatDecimal(lens.metrics.silhouette2d)}</dd></div>
              <div title="Mean adjusted Rand agreement with adjacent parameter settings in the scan; higher indicates a less parameter-sensitive result."><dt>Parameter agreement (ARI)</dt><dd>{formatDecimal(lens.metrics.parameterAgreementAri)}</dd></div>
            </>
          ) : (
            <>
              <div title="Density-based cluster validity; higher is better."><dt>Relative validity (DBCV)</dt><dd>{formatDecimal(lens.metrics.relativeValidity)}</dd></div>
              <div title="Mean persistence of the fitted clusters; higher indicates greater stability."><dt>Mean cluster persistence</dt><dd>{formatDecimal(lens.metrics.meanPersistence)}</dd></div>
              <div title="Mean cluster persistence weighted by the proportion of papers assigned; higher is better."><dt>Persistence-weighted coverage</dt><dd>{formatDecimal(lens.metrics.stabilityWeightedCoverage)}</dd></div>
            </>
          )
        ) : (
          <>
            <div title="Cosine silhouette score in the 100-dimensional PCA space; higher is better."><dt>Silhouette (cosine)</dt><dd>{formatDecimal(lens.metrics.silhouetteCosine)}</dd></div>
            <div title="Davies–Bouldin index in the clustering space; lower is better."><dt>Davies–Bouldin</dt><dd>{formatDecimal(lens.metrics.daviesBouldin)}</dd></div>
            <div title="Calinski–Harabasz index in the clustering space; higher is better."><dt>Calinski–Harabasz</dt><dd>{formatDecimal(lens.metrics.calinskiHarabasz, 0)}</dd></div>
          </>
        )}
      </dl>
    </>
  );
}

function MethodSummary({ lens, collapsed }: { lens: Lens; collapsed: boolean }) {
  if (collapsed) {
    return (
      <details className="method-summary">
        <summary>Parameters and metrics</summary>
        <MethodContent lens={lens} />
      </details>
    );
  }
  return (
    <section className="method-summary method-summary-open">
      <MethodContent lens={lens} />
    </section>
  );
}

function Loader() {
  return (
    <div className="atlas-loader" role="status" aria-live="polite">
      <div className="loader-orbit" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <p>Loading…</p>
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
  const pointersRef = useRef<Map<number, { x: number; y: number }>>(new Map());
  const cameraGestureRef = useRef<{
    kind: 'rotate' | 'pan' | 'pinch';
    startX: number;
    startY: number;
    startSpan: number;
    origin: Camera3D;
    moved: boolean;
  } | null>(null);
  const hoverPickFrameRef = useRef<number | null>(null);
  const pendingHoverPickRef = useRef<{ x: number; y: number } | null>(null);
  const spatialRef = useRef<number[][]>([]);
  const webglRef = useRef<WebGLRenderer | null>(null);
  const detailCacheRef = useRef<Map<number, DetailRow[]>>(new Map());
  const detailPromiseRef = useRef<Map<number, Promise<DetailRow[]>>>(new Map());
  const lensLabelCacheRef = useRef<Map<string, Int16Array>>(new Map());
  const lensRequestRef = useRef(0);
  const projectionRequestRef = useRef<Promise<Projection3D> | null>(null);

  const [mapData, setMapData] = useState<MapData | null>(null);
  const [lensCatalog, setLensCatalog] = useState<LensCatalog | null>(null);
  const [activeLensId, setActiveLensId] = useState('');
  const [labelState, setLabelState] = useState<{ lensId: string; labels: Int16Array } | null>(null);
  const [lensLoadingId, setLensLoadingId] = useState<string | null>(null);
  const [searchData, setSearchData] = useState<SearchRow[] | null>(null);
  const [loadError, setLoadError] = useState('');
  const [lensError, setLensError] = useState('');
  const [viewport, setViewport] = useState({ width: 0, height: 0 });
  const [view, setView] = useState<View>({ zoom: 1, panX: 0, panY: 0 });
  const [displayMode, setDisplayMode] = useState<DisplayMode>('2d');
  const [projection3d, setProjection3d] = useState<Projection3D | null>(null);
  const [projectionLoading, setProjectionLoading] = useState(false);
  const [projectionError, setProjectionError] = useState('');
  const [camera3d, setCamera3d] = useState<Camera3D>(() => resetCamera3D());
  const [show3dHelp, setShow3dHelp] = useState(false);
  const [activeCluster, setActiveCluster] = useState<number | null>(null);
  const [showNoise, setShowNoise] = useState(true);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [selectedSupplement, setSelectedSupplement] = useState<{
    index: number;
    row: DetailRow | null;
  } | null>(null);
  const [query, setQuery] = useState('');
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
      fetch('data/lenses/catalog.json').then((response) => {
        if (!response.ok) throw new Error('The clustering lenses could not be loaded.');
        return response.json() as Promise<LensCatalog>;
      }),
    ])
      .then(([map, catalog]) => {
        if (cancelled) return;
        if (map.count !== catalog.paperCount || map.points.length !== catalog.paperCount) {
          throw new Error('The map and clustering lenses do not describe the same papers.');
        }
        const initialLabels = Int16Array.from(map.points, (point) => point[2]);
        lensLabelCacheRef.current.set(catalog.defaultLens, initialLabels);
        setMapData(map);
        setLensCatalog(catalog);
        setActiveLensId(catalog.defaultLens);
        setLabelState({ lensId: catalog.defaultLens, labels: initialLabels });
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

  const activeLens = useMemo(
    () => lensCatalog?.lenses.find((lens) => lens.id === activeLensId) ?? null,
    [activeLensId, lensCatalog],
  );
  const currentLabels = labelState?.lensId === activeLensId ? labelState.labels : null;

  const clusterById = useMemo(() => {
    const result = new Map<number, Cluster>();
    activeLens?.clusters.forEach((cluster) => result.set(cluster.id, cluster));
    return result;
  }, [activeLens]);
  const sortedClusters = useMemo(
    () => activeLens?.clusters.slice().sort((left, right) => right.count - left.count) ?? [],
    [activeLens],
  );

  const changeLens = useCallback(async (lensId: string) => {
    if (!lensCatalog || lensId === activeLensId || lensLoadingId) return;
    const lens = lensCatalog.lenses.find((candidate) => candidate.id === lensId);
    if (!lens) return;
    const requestId = lensRequestRef.current + 1;
    lensRequestRef.current = requestId;
    setLensError('');
    setLensLoadingId(lensId);
    try {
      let labels = lensLabelCacheRef.current.get(lensId);
      if (!labels) {
        const response = await fetch(lens.labelsFile);
        if (!response.ok) throw new Error('The selected clustering lens could not be loaded.');
        const buffer = await response.arrayBuffer();
        if (buffer.byteLength !== lensCatalog.paperCount * Int16Array.BYTES_PER_ELEMENT) {
          throw new Error('The selected clustering lens does not match the map.');
        }
        labels = new Int16Array(buffer);
        lensLabelCacheRef.current.set(lensId, labels);
      }
      if (lensRequestRef.current !== requestId) return;
      setActiveLensId(lensId);
      setLabelState({ lensId, labels });
      setActiveCluster(null);
      setHoveredIndex(null);
      setShowNoise(true);
    } catch (error) {
      if (lensRequestRef.current === requestId) {
        setLensError(error instanceof Error ? error.message : 'The selected lens could not be loaded.');
      }
    } finally {
      if (lensRequestRef.current === requestId) setLensLoadingId(null);
    }
  }, [activeLensId, lensCatalog, lensLoadingId]);

  const loadProjection3d = useCallback(async (): Promise<Projection3D> => {
    if (projection3d) return projection3d;
    if (projectionRequestRef.current) return projectionRequestRef.current;
    const request = fetch('data/projection-3d.json')
      .then(async (response) => {
        if (!response.ok) throw new Error('The three-dimensional projection could not be loaded.');
        const metadata = await response.json() as Projection3DMetadata;
        if (metadata.paperCount !== mapData?.count || metadata.dimensions !== 3) {
          throw new Error('The three-dimensional projection does not match this atlas.');
        }
        const binaryResponse = await fetch(metadata.file);
        if (!binaryResponse.ok) throw new Error('The three-dimensional coordinates could not be loaded.');
        const buffer = await binaryResponse.arrayBuffer();
        const expectedBytes = metadata.paperCount * 3 * Float32Array.BYTES_PER_ELEMENT;
        if (buffer.byteLength !== expectedBytes) {
          throw new Error('The three-dimensional coordinate file has an unexpected size.');
        }
        const positions = new Float32Array(buffer);
        for (let index = 0; index < positions.length; index += 1) {
          if (!Number.isFinite(positions[index])) {
            throw new Error('The three-dimensional coordinate file contains invalid values.');
          }
        }
        return { metadata, positions };
      })
      .finally(() => {
        projectionRequestRef.current = null;
      });
    projectionRequestRef.current = request;
    return request;
  }, [mapData?.count, projection3d]);

  const changeDisplayMode = useCallback(async (nextMode: DisplayMode) => {
    if (!mapData || nextMode === displayMode || projectionLoading) return;
    setProjectionError('');
    setHoveredIndex(null);
    if (nextMode === '2d') {
      setDisplayMode('2d');
      setShow3dHelp(false);
      return;
    }
    setProjectionLoading(true);
    try {
      const loaded = await loadProjection3d();
      setProjection3d(loaded);
      setCamera3d(resetCamera3D());
      setDisplayMode('3d');
    } catch (error) {
      setProjectionError(error instanceof Error ? error.message : 'The three-dimensional view could not be loaded.');
    } finally {
      setProjectionLoading(false);
    }
  }, [displayMode, loadProjection3d, mapData, projectionLoading]);

  const pointGroups = useMemo(() => {
    if (!currentLabels) return new Map<number, number[]>();
    const groups = new Map<number, number[]>();
    currentLabels.forEach((clusterId, index) => {
      const group = groups.get(clusterId);
      if (group) group.push(index);
      else groups.set(clusterId, [index]);
    });
    return groups;
  }, [currentLabels]);

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

  const toScreenIndex = useCallback((index: number) => {
    const metrics = getMetrics();
    const point = mapData?.points[index];
    if (!metrics || !point) return null;
    if (displayMode === '2d') return { ...toScreen(point[0], point[1]), visible: true };
    const positions = projection3d?.positions;
    if (!positions) return null;
    const offset = index * 3;
    return projectPoint(
      [positions[offset], positions[offset + 1], positions[offset + 2]],
      viewProjectionMatrix(camera3d, metrics.width / Math.max(metrics.height, 1)),
      metrics.width,
      metrics.height,
    );
  }, [camera3d, displayMode, getMetrics, mapData, projection3d, toScreen]);

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

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, targetWidth, targetHeight);
    gl.clearColor(18 / 255, 18 / 255, 26 / 255, 1);
    gl.clearDepth(1);
    gl.depthMask(false);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(renderer.program);
    gl.bindVertexArray(renderer.vertexArray);
    gl.uniform2f(uniforms.center, metrics.centerX, metrics.centerY);
    gl.uniform1f(uniforms.baseScale, metrics.baseScale);
    gl.uniform1f(uniforms.zoom, view.zoom);
    gl.uniform2f(uniforms.pan, view.panX, view.panY);
    gl.uniform2f(uniforms.viewport, metrics.width, metrics.height);
    gl.uniform1f(uniforms.dpr, ratio);
    gl.uniform1f(
      uniforms.pointSize,
      displayMode === '3d'
        ? clamp(2.45 + Math.log2(3.25 / camera3d.distance + 1) * 0.38, 2.55, 5.4)
        : clamp(1.75 + Math.log2(view.zoom + 1) * 0.72, 2.35, 7),
    );
    gl.uniform1f(uniforms.activeCluster, activeCluster ?? -2);
    gl.uniform1f(uniforms.showNoise, showNoise ? 1 : 0);
    gl.uniform1f(uniforms.is3d, displayMode === '3d' ? 1 : 0);
    gl.uniformMatrix4fv(
      uniforms.viewProjection,
      false,
      viewProjectionMatrix(camera3d, metrics.width / Math.max(metrics.height, 1)),
    );
    gl.uniform1f(uniforms.cameraDistance, camera3d.distance);
    gl.drawArrays(gl.POINTS, 0, renderer.pointCount);
    gl.bindVertexArray(null);
  }, [activeCluster, camera3d, displayMode, getMetrics, showNoise, view]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !mapData) return;

    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: true,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
    });
    if (!gl) {
      setLoadError('This map requires a browser with WebGL 2 support.');
      return;
    }

    try {
      const program = linkProgram(
        gl,
        `#version 300 es
        precision highp float;
        layout(location = 0) in vec3 aPosition;
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
        uniform float uIs3d;
        uniform mat4 uViewProjection;
        uniform float uCameraDistance;
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
          if (uIs3d > 0.5) {
            gl_Position = uViewProjection * vec4(aPosition, 1.0);
            float depth = max(gl_Position.w, 0.001);
            float wanted = uPointSize * uDpr * uCameraDistance / depth;
            float floorSize = 2.15 * uDpr;
            float drawn = max(wanted, floorSize);
            float ratio = min(1.0, wanted / max(drawn, 0.001));
            float farFade = clamp(uCameraDistance / depth, 0.20, 1.0);
            float nearFade = smoothstep(0.09, 0.27, depth / max(uCameraDistance, 0.001));
            vAlpha *= ratio * ratio * farFade * nearFade;
            gl_PointSize = drawn;
          } else {
            float x = (aPosition.x - uCenter.x) * uBaseScale * uZoom + uViewport.x * 0.5 + uPan.x;
            float y = -(aPosition.y - uCenter.y) * uBaseScale * uZoom + uViewport.y * 0.5 + uPan.y;
            gl_Position = vec4(x / uViewport.x * 2.0 - 1.0, 1.0 - y / uViewport.y * 2.0, 0.0, 1.0);
            gl_PointSize = uPointSize * uDpr;
          }
          vColor = aColor;
        }`,
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
          float alpha = vAlpha * coverage;
          if (!(alpha >= 0.004)) discard;
          outColor = vec4(vColor, clamp(alpha, 0.0, 1.0));
        }`,
      );
      const pickProgram = linkProgram(
        gl,
        `#version 300 es
        precision highp float;
        layout(location = 0) in vec3 aPosition;
        layout(location = 2) in float aCluster;
        uniform mat4 uViewProjection;
        uniform float uPointSize;
        uniform float uActiveCluster;
        uniform float uShowNoise;
        flat out vec3 vId;

        void main() {
          bool allClusters = uActiveCluster < -1.5;
          bool isNoise = aCluster < -0.5;
          bool visible = isNoise ? uShowNoise > 0.5 : (allClusters || abs(aCluster - uActiveCluster) < 0.1);
          if (!visible) {
            gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
            gl_PointSize = 0.0;
          } else {
            gl_Position = uViewProjection * vec4(aPosition, 1.0);
            gl_PointSize = uPointSize;
          }
          int identifier = gl_VertexID + 1;
          vId = vec3(
            float(identifier & 255),
            float((identifier >> 8) & 255),
            float((identifier >> 16) & 255)
          ) / 255.0;
        }`,
        `#version 300 es
        precision highp float;
        flat in vec3 vId;
        out vec4 outColor;

        void main() {
          vec2 point = gl_PointCoord * 2.0 - 1.0;
          if (dot(point, point) > 1.0) discard;
          outColor = vec4(vId, 1.0);
        }`,
      );

      const positions = new Float32Array(mapData.points.length * 3);
      const colors = new Float32Array(mapData.points.length * 3);
      const clusters = new Float32Array(mapData.points.length);
      mapData.points.forEach((point, index) => {
        positions[index * 3] = point[0];
        positions[index * 3 + 1] = point[1];
        positions[index * 3 + 2] = 0;
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
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, clusterBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, clusters, gl.DYNAMIC_DRAW);
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
        positions2d: positions,
        pickProgram,
        pickTarget: null,
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
          is3d: requiredUniform(gl, program, 'uIs3d'),
          viewProjection: requiredUniform(gl, program, 'uViewProjection'),
          cameraDistance: requiredUniform(gl, program, 'uCameraDistance'),
        },
        pickUniforms: {
          viewProjection: requiredUniform(gl, pickProgram, 'uViewProjection'),
          pointSize: requiredUniform(gl, pickProgram, 'uPointSize'),
          activeCluster: requiredUniform(gl, pickProgram, 'uActiveCluster'),
          showNoise: requiredUniform(gl, pickProgram, 'uShowNoise'),
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
      deletePickTarget(gl, renderer.pickTarget);
      gl.deleteProgram(renderer.pickProgram);
      gl.deleteProgram(renderer.program);
      webglRef.current = null;
    };
  }, [mapData]);

  useEffect(() => {
    const renderer = webglRef.current;
    if (!renderer) return;
    const positions = displayMode === '3d' ? projection3d?.positions : renderer.positions2d;
    if (!positions || positions.length !== renderer.pointCount * 3) return;
    const { gl } = renderer;
    gl.bindBuffer(gl.ARRAY_BUFFER, renderer.positionBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, positions);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    const frame = requestAnimationFrame(drawMap);
    return () => cancelAnimationFrame(frame);
  }, [displayMode, drawMap, projection3d]);

  useEffect(() => {
    const renderer = webglRef.current;
    if (!renderer || !currentLabels || currentLabels.length !== renderer.pointCount) return;
    const clusters = new Float32Array(currentLabels.length);
    const colors = new Float32Array(currentLabels.length * 3);
    currentLabels.forEach((clusterId, index) => {
      clusters[index] = clusterId;
      const color = colorChannels(clusterColor(clusterId, clusterById));
      colors[index * 3] = color[0];
      colors[index * 3 + 1] = color[1];
      colors[index * 3 + 2] = color[2];
    });
    const { gl } = renderer;
    gl.bindBuffer(gl.ARRAY_BUFFER, renderer.clusterBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, clusters);
    gl.bindBuffer(gl.ARRAY_BUFFER, renderer.colorBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, colors);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    const frame = requestAnimationFrame(drawMap);
    return () => cancelAnimationFrame(frame);
    // Buffer uploads should only occur when a clustering lens changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clusterById, currentLabels]);

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

  const pickPoint2d = useCallback(
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
            const clusterId = currentLabels?.[index] ?? point[2];
            if (clusterId < 0 && !showNoise) continue;
            if (activeCluster !== null && clusterId !== activeCluster) continue;
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
    [activeCluster, currentLabels, getMetrics, mapData, showNoise, toScreen, view],
  );

  const pickPoint3d = useCallback((screenX: number, screenY: number): number | null => {
    const renderer = webglRef.current;
    const metrics = getMetrics();
    if (!renderer || !metrics || displayMode !== '3d' || !projection3d) return null;
    const { gl } = renderer;
    try {
      const target = ensurePickTarget(renderer, metrics.width, metrics.height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
      gl.viewport(0, 0, target.width, target.height);
      gl.disable(gl.BLEND);
      gl.enable(gl.DEPTH_TEST);
      gl.depthMask(true);
      gl.clearColor(0, 0, 0, 0);
      gl.clearDepth(1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(renderer.pickProgram);
      gl.bindVertexArray(renderer.vertexArray);
      gl.uniformMatrix4fv(
        renderer.pickUniforms.viewProjection,
        false,
        viewProjectionMatrix(camera3d, metrics.width / Math.max(metrics.height, 1)),
      );
      gl.uniform1f(renderer.pickUniforms.pointSize, Math.max(8, 14 * target.scale));
      gl.uniform1f(renderer.pickUniforms.activeCluster, activeCluster ?? -2);
      gl.uniform1f(renderer.pickUniforms.showNoise, showNoise ? 1 : 0);
      gl.drawArrays(gl.POINTS, 0, renderer.pointCount);
      const pixel = new Uint8Array(4);
      const x = clamp(Math.floor(screenX * target.scale), 0, target.width - 1);
      const y = clamp(target.height - 1 - Math.floor(screenY * target.scale), 0, target.height - 1);
      gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
      const identifier = pixel[0] + (pixel[1] << 8) + (pixel[2] << 16);
      return identifier > 0 && identifier <= renderer.pointCount ? identifier - 1 : null;
    } catch {
      return null;
    } finally {
      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.depthMask(false);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
    }
  }, [activeCluster, camera3d, displayMode, getMetrics, projection3d, showNoise]);

  const schedule3dHoverPick = useCallback((x: number, y: number) => {
    pendingHoverPickRef.current = { x, y };
    if (hoverPickFrameRef.current !== null) return;
    hoverPickFrameRef.current = requestAnimationFrame(() => {
      hoverPickFrameRef.current = null;
      const pending = pendingHoverPickRef.current;
      pendingHoverPickRef.current = null;
      if (!pending || pointersRef.current.size > 0) return;
      setHoveredIndex(pickPoint3d(pending.x, pending.y));
    });
  }, [pickPoint3d]);

  useEffect(() => () => {
    if (hoverPickFrameRef.current !== null) cancelAnimationFrame(hoverPickFrameRef.current);
  }, []);

  const begin3dGesture = useCallback((event: ReactPointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    pointersRef.current.set(event.pointerId, {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    });
    const points = [...pointersRef.current.values()];
    if (points.length >= 2) {
      const first = points[0];
      const second = points[1];
      cameraGestureRef.current = {
        kind: 'pinch',
        startX: (first.x + second.x) / 2,
        startY: (first.y + second.y) / 2,
        startSpan: Math.max(1, Math.hypot(first.x - second.x, first.y - second.y)),
        origin: camera3d,
        moved: true,
      };
    } else {
      cameraGestureRef.current = {
        kind: event.shiftKey || event.button === 1 || event.button === 2 ? 'pan' : 'rotate',
        startX: points[0].x,
        startY: points[0].y,
        startSpan: 0,
        origin: camera3d,
        moved: false,
      };
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    setHoveredIndex(null);
  }, [camera3d]);

  const move3dGesture = useCallback((event: ReactPointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (!pointersRef.current.has(event.pointerId)) {
      if (event.pointerType === 'mouse') schedule3dHoverPick(x, y);
      return;
    }
    pointersRef.current.set(event.pointerId, { x, y });
    const gesture = cameraGestureRef.current;
    const points = [...pointersRef.current.values()];
    if (!gesture) return;
    if (points.length >= 2) {
      const first = points[0];
      const second = points[1];
      const centerX = (first.x + second.x) / 2;
      const centerY = (first.y + second.y) / 2;
      const span = Math.max(1, Math.hypot(first.x - second.x, first.y - second.y));
      if (gesture.kind !== 'pinch') {
        cameraGestureRef.current = {
          kind: 'pinch',
          startX: centerX,
          startY: centerY,
          startSpan: span,
          origin: camera3d,
          moved: true,
        };
        return;
      }
      const zoomed = zoomCamera(gesture.origin, gesture.startSpan / span);
      setCamera3d(panCamera(
        zoomed,
        centerX - gesture.startX,
        centerY - gesture.startY,
        viewport.height,
        viewport.width / Math.max(viewport.height, 1),
      ));
      gesture.moved = true;
      return;
    }
    const deltaX = x - gesture.startX;
    const deltaY = y - gesture.startY;
    if (Math.abs(deltaX) + Math.abs(deltaY) > 3) gesture.moved = true;
    setCamera3d(
      gesture.kind === 'pan'
        ? panCamera(
          gesture.origin,
          deltaX,
          deltaY,
          viewport.height,
          viewport.width / Math.max(viewport.height, 1),
        )
        : orbitCamera(gesture.origin, deltaX, deltaY),
    );
  }, [camera3d, schedule3dHoverPick, viewport]);

  const end3dGesture = useCallback((event: ReactPointerEvent<HTMLCanvasElement>) => {
    const gesture = cameraGestureRef.current;
    const wasSinglePointer = pointersRef.current.size === 1;
    pointersRef.current.delete(event.pointerId);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (wasSinglePointer && gesture && !gesture.moved && event.button === 0) {
      const rect = event.currentTarget.getBoundingClientRect();
      const picked = pickPoint3d(event.clientX - rect.left, event.clientY - rect.top);
      if (picked !== null) {
        setSelectedIndex(picked);
        setHoveredIndex(picked);
        setShowInspector(true);
      }
    }
    const remaining = [...pointersRef.current.values()];
    if (remaining.length === 1) {
      cameraGestureRef.current = {
        kind: 'rotate',
        startX: remaining[0].x,
        startY: remaining[0].y,
        startSpan: 0,
        origin: camera3d,
        moved: true,
      };
    } else if (remaining.length === 0) {
      cameraGestureRef.current = null;
    }
  }, [camera3d, pickPoint3d]);

  const cancel3dGesture = useCallback((event: ReactPointerEvent<HTMLCanvasElement>) => {
    pointersRef.current.delete(event.pointerId);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (pointersRef.current.size === 0) cameraGestureRef.current = null;
  }, []);

  const zoom3dBy = useCallback((factor: number) => {
    setCamera3d((current) => zoomCamera(current, factor));
  }, []);

  const handleCanvasWheel = useCallback((event: WheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    if (displayMode === '3d') {
      zoom3dBy(Math.exp(event.deltaY * 0.0011));
      return;
    }
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
  }, [displayMode, getMetrics, zoom3dBy]);

  const handleCanvasKeyDown = useCallback((event: KeyboardEvent<HTMLCanvasElement>) => {
    const key = event.key.toLowerCase();
    if (displayMode === '3d') {
      if (key === 'arrowleft' || key === 'arrowright' || key === 'arrowup' || key === 'arrowdown') {
        event.preventDefault();
        const horizontal = key === 'arrowleft' ? -24 : key === 'arrowright' ? 24 : 0;
        const vertical = key === 'arrowup' ? -24 : key === 'arrowdown' ? 24 : 0;
        setCamera3d((current) => orbitCamera(current, horizontal, vertical));
      } else if (key === '+' || key === '=') {
        event.preventDefault(); zoom3dBy(0.78);
      } else if (key === '-' || key === '_') {
        event.preventDefault(); zoom3dBy(1.28);
      } else if (key === 'r') {
        event.preventDefault();
        setActiveCluster(null);
        setSelectedIndex(null);
        setCamera3d(resetCamera3D());
      }
      return;
    }
    if (key === '+' || key === '=') {
      event.preventDefault(); setView((current) => ({ ...current, zoom: clamp(current.zoom * 1.35, 0.65, 28) }));
    } else if (key === '-' || key === '_') {
      event.preventDefault(); setView((current) => ({ ...current, zoom: clamp(current.zoom / 1.35, 0.65, 28) }));
    } else if (key === 'r') {
      event.preventDefault();
      setActiveCluster(null);
      setSelectedIndex(null);
      setView({ zoom: 1, panX: 0, panY: 0 });
    }
  }, [displayMode, zoom3dBy]);

  const focusPoint = useCallback(
    (index: number) => {
      if (!mapData) return;
      const metrics = getMetrics();
      const point = mapData.points[index];
      if (!metrics || !point) return;
      const clusterId = currentLabels?.[index] ?? point[2];
      if (displayMode === '3d' && projection3d) {
        const offset = index * 3;
        const center: Vec3 = [
          projection3d.positions[offset],
          projection3d.positions[offset + 1],
          projection3d.positions[offset + 2],
        ];
        setCamera3d((current) => fitCameraToSphere(
          current,
          center,
          0.11,
          metrics.width / Math.max(metrics.height, 1),
        ));
      } else {
        const zoom = Math.max(view.zoom, 6);
        setView({
          zoom,
          panX: -(point[0] - metrics.centerX) * metrics.baseScale * zoom,
          panY: (point[1] - metrics.centerY) * metrics.baseScale * zoom,
        });
      }
      setSelectedIndex(index);
      setActiveCluster(clusterId >= 0 ? clusterId : null);
      setShowInspector(true);
      setQuery('');
    },
    [currentLabels, displayMode, getMetrics, mapData, projection3d, view.zoom],
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
      if (clusterId !== null) setShowInspector(true);
      if (clusterId === null || !mapData) {
        if (displayMode === '3d') setCamera3d(resetCamera3D());
        else setView({ zoom: 1, panX: 0, panY: 0 });
        return;
      }
      const indices = pointGroups.get(clusterId) ?? [];
      const metrics = getMetrics();
      if (!indices.length || !metrics) return;
      if (displayMode === '3d' && projection3d) {
        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;
        let minZ = Infinity;
        let maxZ = -Infinity;
        for (const index of indices) {
          const offset = index * 3;
          const x = projection3d.positions[offset];
          const y = projection3d.positions[offset + 1];
          const z = projection3d.positions[offset + 2];
          minX = Math.min(minX, x); maxX = Math.max(maxX, x);
          minY = Math.min(minY, y); maxY = Math.max(maxY, y);
          minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
        }
        const center: Vec3 = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
        let radius = 0;
        for (const index of indices) {
          const offset = index * 3;
          radius = Math.max(
            radius,
            Math.hypot(
              projection3d.positions[offset] - center[0],
              projection3d.positions[offset + 1] - center[1],
              projection3d.positions[offset + 2] - center[2],
            ),
          );
        }
        setCamera3d((current) => fitCameraToSphere(
          current,
          center,
          radius,
          metrics.width / Math.max(metrics.height, 1),
        ));
        return;
      }
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
    [displayMode, getMetrics, mapData, pointGroups, projection3d],
  );

  const selectedPoint = selectedIndex === null ? null : mapData?.points[selectedIndex] ?? null;
  const selectedClusterId = selectedIndex === null ? null : currentLabels?.[selectedIndex] ?? selectedPoint?.[2] ?? -1;
  const selectedCluster = selectedPoint ? clusterById.get(selectedClusterId ?? -1) : activeCluster === null ? null : clusterById.get(activeCluster);
  const selectedSearch = selectedIndex === null ? null : searchData?.[selectedIndex] ?? null;
  const selectedDetail = selectedIndex !== null && selectedSupplement?.index === selectedIndex
    ? selectedSupplement.row
    : null;
  const detailLoading = selectedIndex !== null && selectedSupplement?.index !== selectedIndex;
  const hoveredRow = hoveredIndex === null ? null : searchData?.[hoveredIndex] ?? null;
  const hoveredPoint = hoveredIndex === null ? null : mapData?.points[hoveredIndex] ?? null;
  const hoveredClusterId = hoveredIndex === null ? null : currentLabels?.[hoveredIndex] ?? hoveredPoint?.[2] ?? -1;
  const hoveredCluster = hoveredPoint ? clusterById.get(hoveredClusterId ?? -1) : null;
  const selectedMarker = selectedIndex === null ? null : toScreenIndex(selectedIndex);
  const hoveredMarker = hoveredIndex === null ? null : toScreenIndex(hoveredIndex);

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
          <button className="mobile-panel-button" onClick={() => setShowLegend(true)}>{activeLens?.algorithm === 'lda' ? 'Topics' : 'Clusters'}</button>
        </div>
      </header>

      <section className="atlas-workspace">
        <aside className={`cluster-panel ${showLegend ? 'panel-open' : ''}`}>
          <p className="corpus-stat"><strong>69,400</strong> English-language papers</p>
          <div className="lens-controls">
            <div className="lens-control-heading">
              <label htmlFor="lens-select">Clustering lens</label>
            </div>
            <select
              id="lens-select"
              value={lensLoadingId ?? activeLensId}
              onChange={(event) => void changeLens(event.target.value)}
              disabled={!lensCatalog || lensLoadingId !== null}
            >
              <optgroup label="HDBSCAN lenses">
                {lensCatalog?.lenses.filter((lens) => lens.algorithm === 'hdbscan').map((lens) => (
                  <option key={lens.id} value={lens.id}>
                    {lens.preferred ? `${lens.optionLabel} (selected)` : lens.optionLabel}
                  </option>
                ))}
              </optgroup>
              <optgroup label="K-means lenses">
                {lensCatalog?.lenses.filter((lens) => lens.algorithm === 'kmeans').map((lens) => (
                  <option key={lens.id} value={lens.id}>{lens.optionLabel}</option>
                ))}
              </optgroup>
              <optgroup label="LDA topic lenses">
                {lensCatalog?.lenses.filter((lens) => lens.algorithm === 'lda').map((lens) => (
                  <option key={lens.id} value={lens.id}>{lens.optionLabel}</option>
                ))}
              </optgroup>
            </select>
            {lensLoadingId && <p className="lens-message">Loading lens…</p>}
            {lensError && <p className="lens-message lens-error">{lensError}</p>}
          </div>
          <div className="panel-heading">
            <h2>{activeLens?.algorithm === 'lda' ? 'Topics' : 'Clusters'} <span>{activeLens?.clusterCount ?? '—'}</span></h2>
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
            {sortedClusters.map((cluster) => (
                <button
                  key={cluster.id}
                  className={`cluster-row ${activeCluster === cluster.id ? 'active' : ''}`}
                  onClick={() => focusCluster(cluster.id)}
                  aria-pressed={activeCluster === cluster.id}
                  title={`${cluster.label} — ${formatNumber(cluster.count)} papers`}
                >
                  <span className="cluster-swatch" style={{ background: clusterColor(cluster.id, clusterById) }} />
                  <span><strong>{cluster.label}</strong></span>
                </button>
              ))}
          </div>
          {activeLens && activeLens.noiseCount > 0 ? (
            <label className="noise-toggle">
              <input type="checkbox" checked={showNoise} onChange={(event) => setShowNoise(event.target.checked)} />
              <span />
              Show unclustered papers
            </label>
          ) : (
            <p className="assignment-note">All papers are assigned in this lens.</p>
          )}
        </aside>

        <div className="map-shell" ref={mapShellRef}>
          {!mapData && <Loader />}
          <div className="view-control-wrap">
            <div className="view-control" role="group" aria-label="Projection view">
              <button
                className={displayMode === '2d' ? 'active' : ''}
                onClick={() => void changeDisplayMode('2d')}
                aria-pressed={displayMode === '2d'}
                disabled={!mapData || projectionLoading}
              >2D</button>
              <button
                className={displayMode === '3d' ? 'active' : ''}
                onClick={() => void changeDisplayMode('3d')}
                aria-pressed={displayMode === '3d'}
                disabled={!mapData || projectionLoading}
              >{projectionLoading ? 'Loading…' : '3D'}</button>
            </div>
            {displayMode === '3d' && (
              <button
                className="view-help-button"
                onClick={() => setShow3dHelp((current) => !current)}
                aria-label="How to use the three-dimensional view"
                aria-expanded={show3dHelp}
              >?</button>
            )}
          </div>

          {projectionError && (
            <div className="projection-message projection-error" role="alert">
              {projectionError}
            </div>
          )}

          {displayMode === '3d' && show3dHelp && (
            <section className="view-help" aria-label="Three-dimensional view controls">
              <button onClick={() => setShow3dHelp(false)} aria-label="Close three-dimensional view help">×</button>
              <h2>Explore in 3D</h2>
              <dl>
                <div><dt>Rotate</dt><dd>Drag</dd></div>
                <div><dt>Move</dt><dd>Shift-drag or right-drag</dd></div>
                <div><dt>Zoom</dt><dd>Scroll or pinch</dd></div>
                <div><dt>Select</dt><dd>Click a paper</dd></div>
                <div><dt>Reset</dt><dd>R</dd></div>
              </dl>
              <p>
                This separate UMAP uses a matching reconstruction of the 30-dimensional analysis pipeline.
                It preserves more local structure than the flat display, but global distances and orientation remain approximate.
              </p>
            </section>
          )}

          <canvas
            ref={canvasRef}
            className={`atlas-canvas mode-${displayMode}`}
            role="application"
            tabIndex={0}
            aria-label={displayMode === '3d'
              ? 'Interactive three-dimensional UMAP of PhilPapers. Drag to rotate, shift-drag to move, scroll to zoom, and select a point to inspect a paper.'
              : 'Interactive two-dimensional UMAP of PhilPapers. Drag to pan, scroll to zoom, and select a point to inspect a paper.'}
            onPointerDown={(event) => {
              if (displayMode === '3d') {
                begin3dGesture(event);
                return;
              }
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
              if (displayMode === '3d') {
                move3dGesture(event);
                return;
              }
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
              const picked = pickPoint2d(localX, localY);
              setHoveredIndex((current) => current === picked ? current : picked);
            }}
            onPointerUp={(event) => {
              if (displayMode === '3d') {
                end3dGesture(event);
                return;
              }
              const drag = dragRef.current;
              dragRef.current = null;
              if (drag && !drag.moved) {
                const rect = event.currentTarget.getBoundingClientRect();
                const picked = pickPoint2d(event.clientX - rect.left, event.clientY - rect.top);
                if (picked !== null) {
                  setSelectedIndex(picked);
                  setHoveredIndex(picked);
                  setShowInspector(true);
                }
              }
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
              }
            }}
            onPointerCancel={(event) => {
              if (displayMode === '3d') {
                cancel3dGesture(event);
                return;
              }
              dragRef.current = null;
            }}
            onPointerLeave={() => {
              if (displayMode === '3d') {
                if (pointersRef.current.size === 0) setHoveredIndex(null);
              } else if (!dragRef.current) setHoveredIndex(null);
            }}
            onWheel={handleCanvasWheel}
            onKeyDown={handleCanvasKeyDown}
            onContextMenu={(event) => {
              if (displayMode === '3d') event.preventDefault();
            }}
          />

          {selectedMarker?.visible && selectedPoint && (
            <span
              className="point-marker selected-marker"
              style={{
                left: selectedMarker.x,
                top: selectedMarker.y,
                background: clusterColor(selectedClusterId ?? -1, clusterById),
              }}
              aria-hidden="true"
            />
          )}
          {hoveredMarker?.visible && hoveredPoint && hoveredIndex !== selectedIndex && (
            <span
              className="point-marker hover-marker"
              style={{
                left: hoveredMarker.x,
                top: hoveredMarker.y,
                background: clusterColor(hoveredClusterId ?? -1, clusterById),
              }}
              aria-hidden="true"
            />
          )}

          {hoveredIndex !== null && hoveredPoint && hoveredMarker?.visible && (
            <div className="paper-tooltip" style={{ left: hoveredMarker.x, top: hoveredMarker.y }}>
              <span style={{ background: clusterColor(hoveredClusterId ?? -1, clusterById) }} />
              <div>
                <strong>{hoveredRow?.[0] ?? `Paper ${hoveredIndex + 1}`}</strong>
                <small>{hoveredCluster?.label ?? 'Unclustered'}</small>
              </div>
            </div>
          )}

          <div className="map-controls" aria-label="Map controls">
            <button
              onClick={() => displayMode === '3d'
                ? zoom3dBy(0.72)
                : setView((current) => ({ ...current, zoom: clamp(current.zoom * 1.45, 0.65, 28) }))}
              aria-label="Zoom in"
            >+</button>
            <button
              onClick={() => displayMode === '3d'
                ? zoom3dBy(1.38)
                : setView((current) => ({ ...current, zoom: clamp(current.zoom / 1.45, 0.65, 28) }))}
              aria-label="Zoom out"
            >−</button>
            <button className="reset-control" onClick={() => focusCluster(null)}>Reset</button>
          </div>
          <button className="mobile-inspector-button" onClick={() => setShowInspector(true)}>Details</button>
        </div>

        <aside className={`paper-panel ${showInspector ? 'panel-open' : ''}`}>
          <button
            className={`panel-close inspector-close ${selectedIndex !== null ? 'paper-selected-close' : ''}`}
            onClick={() => selectedIndex !== null ? setSelectedIndex(null) : setShowInspector(false)}
            aria-label={selectedIndex !== null ? 'Back to cluster overview' : 'Close details panel'}
          >×</button>

          <div className="paper-panel-content">
            {selectedIndex !== null ? (
              <article className="paper-detail">
                <div className="detail-cluster-label">
                  <span style={{ background: clusterColor(selectedClusterId ?? -1, clusterById) }} />
                  {selectedCluster?.label ?? 'Unclustered in this lens'}
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
                <button className="secondary-link" onClick={() => setSelectedIndex(null)}>Back</button>
              </article>
            ) : activeCluster !== null && selectedCluster ? (
              <article className="cluster-detail">
                <div className="cluster-orb" style={{ background: clusterColor(selectedCluster.id, clusterById) }} />
                <h3>{selectedCluster.label}</h3>
                <p><strong>{formatNumber(selectedCluster.count)}</strong> papers assigned to this {activeLens?.algorithm === 'lda' ? 'topic' : 'cluster'}.</p>
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
                <h3>
                  {activeLens?.algorithm === 'hdbscan'
                    ? activeLens.method.clusteringSpace === 'display2d'
                      ? 'HDBSCAN · display UMAP 2D'
                      : `HDBSCAN · UMAP ${activeLens.method.umapDimensions}D`
                    : activeLens?.algorithm === 'kmeans'
                      ? `K-means · PCA ${activeLens.method.pcaDimensions}D`
                      : activeLens?.algorithm === 'lda'
                        ? `LDA · ${activeLens.method.topicCount} topics`
                      : 'Clustering lens'}
                </h3>
                {activeLens && (
                  <p className="lens-outcome">
                    <strong>{activeLens.clusterCount}</strong> {activeLens.algorithm === 'lda' ? 'topics' : 'clusters'}
                    <span>·</span>
                    <strong>{activeLens.noisePct.toFixed(1)}%</strong> unassigned
                  </p>
                )}
              </article>
            )}
          </div>
          {activeLens && lensCatalog && (
            <MethodSummary
              lens={activeLens}
              collapsed={selectedIndex !== null || activeCluster !== null}
            />
          )}
        </aside>
      </section>
    </main>
  );
}
