import assert from 'node:assert/strict';
import test from 'node:test';

import {
  normalizePaperSearch,
  preparePaperSearch,
  searchPreparedPapers,
  type PaperSearchRow,
} from '../app/paperSearch.ts';

const rows: PaperSearchRow[] = [
  ['Machine Consciousness', 'Helliwell, Alice', '2024', 'https://example.test/1'],
  ['Alice and the philosophy of science', 'Morgan, Beth', '2020', 'https://example.test/2'],
  ['Embodiment and agency', 'García Márquez, José', '2019', 'https://example.test/3'],
  ['Consciousness in machines', 'Alice H. Helliwell', '2023', 'https://example.test/4'],
];
const prepared = preparePaperSearch(rows);

test('normalisation removes diacritics and punctuation without joining words', () => {
  assert.equal(normalizePaperSearch('  García-Márquez, José '), 'garcia marquez jose');
});

test('author names match independently of display order', () => {
  assert.deepEqual(searchPreparedPapers(prepared, 'Alice Helliwell', 4), [0, 3]);
});

test('author search handles diacritics and reversed names', () => {
  assert.deepEqual(searchPreparedPapers(prepared, 'Jose Garcia Marquez', 4), [2]);
});

test('contiguous title matches rank above scattered token matches', () => {
  assert.deepEqual(searchPreparedPapers(prepared, 'machine consciousness', 4).slice(0, 2), [0, 3]);
});

test('short single-character queries do not scan', () => {
  assert.deepEqual(searchPreparedPapers(prepared, 'a', 4), []);
});
