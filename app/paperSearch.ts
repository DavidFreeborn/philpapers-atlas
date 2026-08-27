export type PaperSearchRow = [string, string, string, string];

type PreparedPaperSearchRow = {
  title: string;
  authors: string;
  titleTokens: string[];
  authorTokens: string[];
};

export type PreparedPaperSearch = {
  rows: PreparedPaperSearchRow[];
  tokenIndex: Map<string, number[]>;
  vocabulary: string[];
};

export function normalizePaperSearch(value: string): string {
  return value
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[’'`]/g, '')
    .toLocaleLowerCase()
    .replace(/&/g, ' and ')
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim()
    .replace(/\s+/g, ' ');
}

export function preparePaperSearch(rows: PaperSearchRow[]): PreparedPaperSearch {
  const tokenIndex = new Map<string, number[]>();
  const preparedRows = rows.map((row, index) => {
    const title = normalizePaperSearch(row[0]);
    const authors = normalizePaperSearch(row[1]);
    const prepared = {
      title,
      authors,
      titleTokens: title.split(' ').filter(Boolean),
      authorTokens: authors.split(' ').filter(Boolean),
    };
    new Set(prepared.titleTokens.concat(prepared.authorTokens)).forEach((token) => {
      const postings = tokenIndex.get(token);
      if (postings) postings.push(index);
      else tokenIndex.set(token, [index]);
    });
    return prepared;
  });
  return { rows: preparedRows, tokenIndex, vocabulary: [...tokenIndex.keys()] };
}

function tokensMatch(queryTokens: string[], candidateTokens: string[], allowPrefix: boolean): boolean {
  return queryTokens.every((queryToken) => candidateTokens.some((candidateToken) => (
    candidateToken === queryToken
      || (allowPrefix && queryToken.length >= 2 && candidateToken.startsWith(queryToken))
  )));
}

function scoreRow(row: PreparedPaperSearchRow, query: string, queryTokens: string[]): number {
  if (row.title === query) return 1_400;
  if (row.authors === query) return 1_350;
  if (row.title.startsWith(query)) return 1_250;
  if (row.authors.startsWith(query)) return 1_200;
  if (row.title.includes(query)) return 1_100;
  if (row.authors.includes(query)) return 1_050;
  if (tokensMatch(queryTokens, row.authorTokens, false)) return 950 + queryTokens.length;
  if (tokensMatch(queryTokens, row.titleTokens, false)) return 900 + queryTokens.length;
  if (tokensMatch(queryTokens, row.authorTokens, true)) return 800 + queryTokens.length;
  if (tokensMatch(queryTokens, row.titleTokens, true)) return 750 + queryTokens.length;
  const combinedTokens = row.titleTokens.concat(row.authorTokens);
  if (tokensMatch(queryTokens, combinedTokens, false)) return 650 + queryTokens.length;
  if (tokensMatch(queryTokens, combinedTokens, true)) return 550 + queryTokens.length;
  return -1;
}

export function searchPreparedPapers(
  prepared: PreparedPaperSearch,
  rawQuery: string,
  limit = 8,
): number[] {
  const query = normalizePaperSearch(rawQuery);
  if (query.length < 2 || limit <= 0) return [];
  const queryTokens = query.split(' ').filter(Boolean);
  const candidateLists: number[][] = [];
  for (const queryToken of queryTokens) {
    if (queryToken.length < 2) return [];
    const prefixed: number[] = [];
    for (const token of prepared.vocabulary) {
      if (token.startsWith(queryToken)) prefixed.push(...(prepared.tokenIndex.get(token) ?? []));
    }
    if (!prefixed.length) return [];
    candidateLists.push([...new Set(prefixed)]);
  }
  const candidates = candidateLists.reduce((smallest, current) => (
    current.length < smallest.length ? current : smallest
  ));
  const ranked: Array<{ index: number; score: number }> = [];
  for (const index of candidates) {
    const score = scoreRow(prepared.rows[index], query, queryTokens);
    if (score < 0) continue;
    ranked.push({ index, score });
    ranked.sort((left, right) => right.score - left.score || left.index - right.index);
    if (ranked.length > limit) ranked.pop();
  }
  return ranked.map((result) => result.index);
}
