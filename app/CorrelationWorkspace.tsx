'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  associationCellKey,
  buildAssociation,
  type AssociationCell,
  type AssociationLens,
  type AssociationOverview,
  type AssociationOverviewPair,
  type AssociationResult,
} from './associationAnalysis';

type Props = {
  lenses: AssociationLens[];
  defaultLensId: string;
  loadLabels: (lensId: string) => Promise<Int16Array>;
};

function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-GB').format(value);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function pairKey(left: string, right: string): string {
  return left < right ? `${left}::${right}` : `${right}::${left}`;
}

function overviewColour(value: number): string {
  const level = clamp(value, 0, 1);
  const lightness = 20 + level * 58;
  const saturation = 18 + level * 60;
  return `hsl(226 ${saturation}% ${lightness}%)`;
}

function residualColour(value: number, scale: number): string {
  const strength = Math.sqrt(clamp(Math.abs(value) / Math.max(scale, 0.01), 0, 1));
  if (value >= 0) {
    return `rgb(${Math.round(38 + strength * 210)} ${Math.round(43 + strength * 66)} ${Math.round(58 + strength * 42)})`;
  }
  return `rgb(${Math.round(34 + strength * 46)} ${Math.round(42 + strength * 110)} ${Math.round(58 + strength * 188)})`;
}

function cellDescription(cell: AssociationCell, total: number): string {
  const difference = cell.observed - cell.expected;
  const relation = difference >= 0 ? 'more' : 'fewer';
  return `${formatNumber(cell.observed)} papers; ${formatNumber(Math.round(cell.expected))} expected; ${formatNumber(Math.abs(Math.round(difference)))} ${relation} than expected.`;
}

export default function CorrelationWorkspace({ lenses, defaultLensId, loadLabels }: Props) {
  const [overview, setOverview] = useState<AssociationOverview | null>(null);
  const [overviewError, setOverviewError] = useState('');
  const [leftId, setLeftId] = useState(defaultLensId);
  const [rightId, setRightId] = useState(
    lenses.find((lens) => lens.id === 'lda_k20_default')?.id
      ?? lenses.find((lens) => lens.id !== defaultLensId)?.id
      ?? defaultLensId,
  );
  const [result, setResult] = useState<AssociationResult | null>(null);
  const [detailError, setDetailError] = useState<{ key: string; message: string } | null>(null);
  const [visibleCount, setVisibleCount] = useState(10);
  const [hoveredCell, setHoveredCell] = useState<{
    cell: AssociationCell;
    rowLabel: string;
    columnLabel: string;
  } | null>(null);
  const requestRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    fetch('data/lenses/associations.json')
      .then((response) => {
        if (!response.ok) throw new Error('The association overview could not be loaded.');
        return response.json() as Promise<AssociationOverview>;
      })
      .then((data) => {
        if (!cancelled) setOverview(data);
      })
      .catch((error: Error) => {
        if (!cancelled) setOverviewError(error.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const lensById = useMemo(
    () => new Map(lenses.map((lens) => [lens.id, lens])),
    [lenses],
  );
  const overviewPairs = useMemo(() => {
    const pairs = new Map<string, AssociationOverviewPair>();
    overview?.pairs.forEach((pair) => pairs.set(pairKey(pair.left, pair.right), pair));
    return pairs;
  }, [overview]);
  const selectionKey = pairKey(leftId, rightId);
  const detailLoading = !result || pairKey(result.left, result.right) !== selectionKey;
  const currentDetailError = detailError?.key === selectionKey ? detailError.message : '';

  useEffect(() => {
    if (!leftId || !rightId || leftId === rightId) return;
    const leftLens = lensById.get(leftId);
    const rightLens = lensById.get(rightId);
    if (!leftLens || !rightLens) return;
    const requestId = requestRef.current + 1;
    requestRef.current = requestId;
    Promise.all([loadLabels(leftId), loadLabels(rightId)])
      .then(([leftLabels, rightLabels]) => {
        if (requestRef.current !== requestId) return;
        setResult(buildAssociation(leftLens, rightLens, leftLabels, rightLabels));
        setDetailError(null);
      })
      .catch((error: Error) => {
        if (requestRef.current === requestId) {
          setDetailError({ key: pairKey(leftId, rightId), message: error.message });
        }
      });
  }, [leftId, lensById, loadLabels, rightId]);

  const selectOverviewPair = (left: string, right: string) => {
    if (left === right) return;
    setLeftId(left);
    setRightId(right);
  };

  const leftLens = lensById.get(leftId);
  const rightLens = lensById.get(rightId);
  const rows = result?.rows.slice(0, visibleCount) ?? [];
  const columns = result?.columns.slice(0, visibleCount) ?? [];
  const maximumResidual = useMemo(() => {
    if (!result) return 1;
    let maximum = 1;
    for (const row of rows) {
      for (const column of columns) {
        maximum = Math.max(
          maximum,
          Math.abs(result.cells.get(associationCellKey(row.id, column.id))?.residual ?? 0),
        );
      }
    }
    return maximum;
  }, [columns, result, rows]);

  return (
    <section className="correlation-workspace" aria-label="Correlation matrices">
      <div className="association-overview-panel">
        <div className="correlation-heading">
          <p className="eyebrow">All lenses</p>
          <h1>Pairwise association</h1>
          <p>Bias-corrected Cramér’s V measures how strongly each pair of categorical assignments is associated.</p>
        </div>
        {overviewError ? (
          <p className="lens-error" role="alert">{overviewError}</p>
        ) : !overview ? (
          <p className="correlation-loading">Loading…</p>
        ) : (
          <>
            <div className="association-overview-scroll">
              <div
                className="association-overview-grid"
                style={{ gridTemplateColumns: `116px repeat(${overview.lenses.length}, 34px)` }}
              >
                <span />
                {overview.lenses.map((lens) => (
                  <span className="association-column-label" key={lens.id} title={lens.label}>
                    {lens.shortLabel}
                  </span>
                ))}
                {overview.lenses.map((rowLens) => (
                  <div className="association-overview-row" key={rowLens.id}>
                    <span title={rowLens.label}>{rowLens.shortLabel}</span>
                    {overview.lenses.map((columnLens) => {
                      const diagonal = rowLens.id === columnLens.id;
                      const pair = overviewPairs.get(pairKey(rowLens.id, columnLens.id));
                      const value = diagonal ? 1 : pair?.cramersV ?? 0;
                      const selected = !diagonal
                        && ((leftId === rowLens.id && rightId === columnLens.id)
                          || (rightId === rowLens.id && leftId === columnLens.id));
                      return (
                        <button
                          key={columnLens.id}
                          className={selected ? 'selected' : ''}
                          style={{ background: overviewColour(value) }}
                          disabled={diagonal}
                          onClick={() => selectOverviewPair(rowLens.id, columnLens.id)}
                          aria-label={diagonal
                            ? `${rowLens.label} compared with itself`
                            : `${rowLens.label} and ${columnLens.label}: Cramér’s V ${value.toFixed(3)}`}
                          title={diagonal ? 'Same lens' : `V = ${value.toFixed(3)}`}
                        >
                          {value.toFixed(2)}
                        </button>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
            <div className="association-scale" aria-label="Cramér’s V scale from zero to one">
              <span>0</span><i /><span>1</span>
            </div>
            <p className="association-note">{overview.inclusion}</p>
          </>
        )}
      </div>

      <div className="association-detail-panel">
        <div className="association-selectors">
          <label>
            <span>Rows</span>
            <select value={leftId} onChange={(event) => setLeftId(event.target.value)}>
              {lenses.map((lens) => (
                <option key={lens.id} value={lens.id} disabled={lens.id === rightId}>{lens.optionLabel}</option>
              ))}
            </select>
          </label>
          <button
            onClick={() => {
              setLeftId(rightId);
              setRightId(leftId);
            }}
            aria-label="Swap rows and columns"
            title="Swap rows and columns"
          >⇄</button>
          <label>
            <span>Columns</span>
            <select value={rightId} onChange={(event) => setRightId(event.target.value)}>
              {lenses.map((lens) => (
                <option key={lens.id} value={lens.id} disabled={lens.id === leftId}>{lens.optionLabel}</option>
              ))}
            </select>
          </label>
        </div>

        <div className="association-detail-heading">
          <div>
            <p className="eyebrow">Pearson residuals</p>
            <h2>{leftLens?.name} × {rightLens?.name}</h2>
          </div>
          <div className="association-summary">
            <span><strong>{result?.cramersV.toFixed(3) ?? '—'}</strong>Cramér’s V</span>
            <span><strong>{result ? formatNumber(result.eligiblePapers) : '—'}</strong>papers</span>
          </div>
        </div>

        <div className="association-toolbar">
          <span>Largest categories</span>
          <div role="group" aria-label="Number of categories shown">
            {[10, 20, 30].map((count) => (
              <button
                key={count}
                className={visibleCount === count ? 'active' : ''}
                onClick={() => setVisibleCount(count)}
                aria-pressed={visibleCount === count}
              >{count}</button>
            ))}
          </div>
        </div>

        {currentDetailError ? (
          <p className="lens-error" role="alert">{currentDetailError}</p>
        ) : detailLoading || !result ? (
          <p className="correlation-loading">Loading…</p>
        ) : (
          <>
            <div className="residual-matrix-scroll">
              <div
                className="residual-matrix"
                style={{ gridTemplateColumns: `minmax(140px, 210px) repeat(${columns.length}, 48px)` }}
              >
                <span className="residual-corner">Observed relative to independence</span>
                {columns.map((column) => (
                  <span className="residual-column-label" key={column.id} title={column.label}>
                    <i>{column.label}</i>
                  </span>
                ))}
                {rows.map((row) => (
                  <div className="residual-row" key={row.id}>
                    <span className="residual-row-label" title={row.label}>
                      <strong>{row.label}</strong><small>{formatNumber(row.count)}</small>
                    </span>
                    {columns.map((column) => {
                      const cell = result.cells.get(associationCellKey(row.id, column.id));
                      if (!cell) return <span key={column.id} />;
                      return (
                        <button
                          key={column.id}
                          style={{ background: residualColour(cell.residual, maximumResidual) }}
                          onPointerEnter={() => setHoveredCell({
                            cell,
                            rowLabel: row.label,
                            columnLabel: column.label,
                          })}
                          onPointerLeave={() => setHoveredCell(null)}
                          onFocus={() => setHoveredCell({
                            cell,
                            rowLabel: row.label,
                            columnLabel: column.label,
                          })}
                          onBlur={() => setHoveredCell(null)}
                          aria-label={`${row.label} and ${column.label}: ${cellDescription(cell, result.eligiblePapers)}`}
                          title={`${cell.residual >= 0 ? '+' : ''}${cell.residual.toFixed(1)} residual`}
                        >
                          {cell.residual >= 0 ? '+' : ''}{cell.residual.toFixed(0)}
                        </button>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
            <div className="residual-footer">
              <div className="residual-scale">
                <span>Fewer</span><i /><span>As expected</span><b /><span>More</span>
              </div>
              <p aria-live="polite">
                {hoveredCell
                  ? <><strong>{hoveredCell.rowLabel} × {hoveredCell.columnLabel}.</strong> {cellDescription(hoveredCell.cell, result.eligiblePapers)}</>
                  : 'Select or hover over a cell for observed and expected counts.'}
              </p>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
