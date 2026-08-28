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

test('author initials match full given names when anchored by a surname', () => {
  assert.deepEqual(searchPreparedPapers(prepared, 'A H Helliwell', 4), [3]);
});

test('a single adjacent transposition is tolerated', () => {
  assert.deepEqual(searchPreparedPapers(prepared, 'Helliwlel Alice', 4), [0, 3]);
});

test('singular and plural title terms are treated as equivalent', () => {
  assert.deepEqual(searchPreparedPapers(prepared, 'machines consciousness', 4).slice(0, 2), [0, 3]);
});

test('more than one spelling error does not produce an unrelated match', () => {
  assert.deepEqual(searchPreparedPapers(prepared, 'Hellawoll', 4), []);
});

test('author-initial matches rank above papers that only mention the author', () => {
  const chalmersRows = preparePaperSearch([
    ['A response to David Chalmers', 'Bishop, John', '', ''],
    ['The Conscious Mind', 'Chalmers, David J.', '', ''],
  ]);
  assert.deepEqual(searchPreparedPapers(chalmersRows, 'D Chalmers', 2), [1, 0]);
});

test('order-independent author matches rank above title mentions of that author', () => {
  const chalmersRows = preparePaperSearch([
    ['A response to David Chalmers', 'Bishop, John', '', ''],
    ['The Conscious Mind', 'Chalmers, David', '', ''],
  ]);
  assert.deepEqual(searchPreparedPapers(chalmersRows, 'David Chalmers', 2), [1, 0]);
});
