import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import {
  associationCellKey,
  buildAssociation,
  correctedCramersV,
  type AssociationLens,
  type AssociationOverview,
} from '../app/associationAnalysis.ts';

const category = (id: number, label: string, count: number) => ({ id, label, count });

test('bias-corrected Cramér’s V is one for identical non-degenerate assignments', () => {
  const counts = new Map([
    [associationCellKey(0, 0), 50],
    [associationCellKey(1, 1), 50],
  ]);
  assert.equal(
    correctedCramersV(counts, new Map([[0, 50], [1, 50]]), new Map([[0, 50], [1, 50]]), 100),
    1,
  );
});

test('metadata missing values are excluded while HDBSCAN noise is retained', () => {
  const densityLens: AssociationLens = {
    id: 'density',
    name: 'Density',
    optionLabel: 'Density',
    algorithm: 'hdbscan',
    clusters: [category(0, 'A', 2), category(1, 'B', 1)],
  };
  const metadataLens: AssociationLens = {
    id: 'metadata',
    name: 'Metadata',
    optionLabel: 'Metadata',
    algorithm: 'metadata',
    clusters: [category(0, 'Article', 2), category(1, 'Book', 1)],
  };
  const result = buildAssociation(
    densityLens,
    metadataLens,
    Int16Array.from([-1, 0, 0, 1]),
    Int16Array.from([0, 0, -1, 1]),
  );
  assert.equal(result.eligiblePapers, 3);
  assert.deepEqual(result.rows.map((row) => row.label), ['A', 'B', 'Unclustered']);
  assert.equal(result.cells.get(associationCellKey(-1, 0))?.observed, 1);
});

test('browser calculation matches the published overview for a real lens pair', async () => {
  const catalog = JSON.parse(await readFile('public/data/lenses/catalog.json', 'utf8')) as {
    lenses: Array<AssociationLens & { labelsFile: string }>;
  };
  const overview = JSON.parse(
    await readFile('public/data/lenses/associations.json', 'utf8'),
  ) as AssociationOverview;
  const left = catalog.lenses.find((lens) => lens.id === 'pca100_u30_mcs200_ms15');
  const right = catalog.lenses.find((lens) => lens.id === 'metadata_publication_period');
  assert.ok(left && right);
  const load = async (file: string) => {
    const buffer = await readFile(file);
    return new Int16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
  };
  const result = buildAssociation(
    left,
    right,
    await load(`public/${left.labelsFile}`),
    await load(`public/${right.labelsFile}`),
  );
  const expected = overview.pairs.find((pair) => (
    pair.left === left.id && pair.right === right.id
  ));
  assert.ok(expected);
  assert.equal(result.eligiblePapers, expected.eligiblePapers);
  assert.ok(Math.abs(result.cramersV - expected.cramersV) < 1e-6);
});
