export type AssociationLens = {
  id: string;
  name: string;
  optionLabel: string;
  algorithm: 'hdbscan' | 'kmeans' | 'lda' | 'metadata';
  clusters: Array<{ id: number; label: string; count: number }>;
};

export type AssociationOverviewLens = {
  id: string;
  label: string;
  shortLabel: string;
};

export type AssociationOverviewPair = {
  left: string;
  right: string;
  cramersV: number;
  eligiblePapers: number;
};

export type AssociationOverview = {
  version: string;
  paperCount: number;
  measure: 'bias-corrected Cramers V';
  lenses: AssociationOverviewLens[];
  pairs: AssociationOverviewPair[];
  inclusion: string;
};

export type AssociationCategory = {
  id: number;
  label: string;
  count: number;
};

export type AssociationCell = {
  row: number;
  column: number;
  observed: number;
  expected: number;
  residual: number;
};

export type AssociationResult = {
  left: string;
  right: string;
  cramersV: number;
  eligiblePapers: number;
  rows: AssociationCategory[];
  columns: AssociationCategory[];
  cells: Map<string, AssociationCell>;
};

const CELL_SEPARATOR = ':';

function includeLabel(lens: AssociationLens, label: number): boolean {
  // HDBSCAN noise is an informative assignment. A negative metadata label means
  // that the source field is unavailable and is therefore excluded pairwise.
  return lens.algorithm !== 'metadata' || label >= 0;
}

function categoryLabel(lens: AssociationLens, category: number): string {
  if (category < 0) return 'Unclustered';
  return lens.clusters.find((cluster) => cluster.id === category)?.label
    ?? `Category ${category}`;
}

export function associationCellKey(row: number, column: number): string {
  return `${row}${CELL_SEPARATOR}${column}`;
}

export function correctedCramersV(
  cellCounts: ReadonlyMap<string, number>,
  rowMargins: ReadonlyMap<number, number>,
  columnMargins: ReadonlyMap<number, number>,
  total: number,
): number {
  const rowCount = rowMargins.size;
  const columnCount = columnMargins.size;
  if (total <= 1 || rowCount <= 1 || columnCount <= 1) return 0;

  let weightedSquares = 0;
  for (const [key, observed] of cellCounts) {
    const separator = key.indexOf(CELL_SEPARATOR);
    const row = Number(key.slice(0, separator));
    const column = Number(key.slice(separator + 1));
    weightedSquares += (observed * observed)
      / ((rowMargins.get(row) ?? 1) * (columnMargins.get(column) ?? 1));
  }

  const chiSquare = Math.max(0, total * weightedSquares - total);
  const phiSquared = chiSquare / total;
  const correction = ((columnCount - 1) * (rowCount - 1)) / (total - 1);
  const correctedPhiSquared = Math.max(0, phiSquared - correction);
  const correctedRows = rowCount - ((rowCount - 1) ** 2) / (total - 1);
  const correctedColumns = columnCount - ((columnCount - 1) ** 2) / (total - 1);
  const denominator = Math.min(correctedRows - 1, correctedColumns - 1);
  return denominator > 0 ? Math.sqrt(correctedPhiSquared / denominator) : 0;
}

export function buildAssociation(
  leftLens: AssociationLens,
  rightLens: AssociationLens,
  leftLabels: Int16Array,
  rightLabels: Int16Array,
): AssociationResult {
  if (leftLens.id === rightLens.id) {
    throw new Error('Choose two different lenses.');
  }
  if (leftLabels.length !== rightLabels.length) {
    throw new Error('The selected lenses describe different paper cohorts.');
  }

  const cellCounts = new Map<string, number>();
  const rowMargins = new Map<number, number>();
  const columnMargins = new Map<number, number>();
  let eligiblePapers = 0;

  for (let index = 0; index < leftLabels.length; index += 1) {
    const row = leftLabels[index];
    const column = rightLabels[index];
    if (!includeLabel(leftLens, row) || !includeLabel(rightLens, column)) continue;
    const key = associationCellKey(row, column);
    cellCounts.set(key, (cellCounts.get(key) ?? 0) + 1);
    rowMargins.set(row, (rowMargins.get(row) ?? 0) + 1);
    columnMargins.set(column, (columnMargins.get(column) ?? 0) + 1);
    eligiblePapers += 1;
  }

  const rows = [...rowMargins]
    .map(([id, count]) => ({ id, count, label: categoryLabel(leftLens, id) }))
    .sort((left, right) => right.count - left.count || left.label.localeCompare(right.label));
  const columns = [...columnMargins]
    .map(([id, count]) => ({ id, count, label: categoryLabel(rightLens, id) }))
    .sort((left, right) => right.count - left.count || left.label.localeCompare(right.label));
  const cells = new Map<string, AssociationCell>();

  for (const row of rows) {
    for (const column of columns) {
      const key = associationCellKey(row.id, column.id);
      const observed = cellCounts.get(key) ?? 0;
      const expected = eligiblePapers
        ? (row.count * column.count) / eligiblePapers
        : 0;
      cells.set(key, {
        row: row.id,
        column: column.id,
        observed,
        expected,
        residual: expected > 0 ? (observed - expected) / Math.sqrt(expected) : 0,
      });
    }
  }

  return {
    left: leftLens.id,
    right: rightLens.id,
    cramersV: correctedCramersV(
      cellCounts,
      rowMargins,
      columnMargins,
      eligiblePapers,
    ),
    eligiblePapers,
    rows,
    columns,
    cells,
  };
}
