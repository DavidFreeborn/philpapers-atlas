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
  tokenBuckets: Map<string, string[]>;
  lengthBuckets: Map<number, string[]>;
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
  const vocabulary = [...tokenIndex.keys()];
  const tokenBuckets = new Map<string, string[]>();
  const lengthBuckets = new Map<number, string[]>();
  for (const token of vocabulary) {
    const prefix = token.slice(0, 2);
    const prefixed = tokenBuckets.get(prefix);
    if (prefixed) prefixed.push(token);
    else tokenBuckets.set(prefix, [token]);
    const sameLength = lengthBuckets.get(token.length);
    if (sameLength) sameLength.push(token);
    else lengthBuckets.set(token.length, [token]);
  }
  return { rows: preparedRows, tokenIndex, tokenBuckets, lengthBuckets };
}

function singularForm(token: string): string {
  if (token.length >= 5 && token.endsWith('ies')) return `${token.slice(0, -3)}y`;
  if (token.length >= 5 && /(sses|shes|ches|xes|zes)$/.test(token)) return token.slice(0, -2);
  if (
    token.length >= 4
    && token.endsWith('s')
    && !/(ss|us|is|ics)$/.test(token)
  ) return token.slice(0, -1);
  return token;
}

function tokensEquivalent(left: string, right: string): boolean {
  return left === right || singularForm(left) === singularForm(right);
}

function withinOneEdit(left: string, right: string): boolean {
  if (left === right) return true;
  if (Math.abs(left.length - right.length) > 1) return false;
  if (left.length === right.length) {
    const differences: number[] = [];
    for (let index = 0; index < left.length; index += 1) {
      if (left[index] !== right[index]) differences.push(index);
      if (differences.length > 2) return false;
    }
    if (differences.length === 1) return true;
    return differences.length === 2
      && differences[1] === differences[0] + 1
      && left[differences[0]] === right[differences[1]]
      && left[differences[1]] === right[differences[0]];
  }
  const [shorter, longer] = left.length < right.length ? [left, right] : [right, left];
  let shortIndex = 0;
  let longIndex = 0;
  let skipped = false;
  while (shortIndex < shorter.length && longIndex < longer.length) {
    if (shorter[shortIndex] === longer[longIndex]) {
      shortIndex += 1;
      longIndex += 1;
    } else if (skipped) {
      return false;
    } else {
      skipped = true;
      longIndex += 1;
    }
  }
  return true;
}

type TokenMatchOptions = {
  allowPrefix?: boolean;
  allowInitial?: boolean;
  allowFuzzy?: boolean;
};

function tokenMatches(
  queryToken: string,
  candidateToken: string,
  options: TokenMatchOptions,
): boolean {
  if (tokensEquivalent(queryToken, candidateToken)) return true;
  if (options.allowInitial && queryToken.length === 1 && candidateToken.startsWith(queryToken)) return true;
  if (options.allowPrefix && queryToken.length >= 2 && candidateToken.startsWith(queryToken)) return true;
  return Boolean(
    options.allowFuzzy
    && queryToken.length >= 4
    && candidateToken.length >= 4
    && withinOneEdit(queryToken, candidateToken)
  );
}

function tokensMatch(
  queryTokens: string[],
  candidateTokens: string[],
  options: TokenMatchOptions = {},
): boolean {
  const candidateOptions = queryTokens.map((queryToken) => (
    candidateTokens.flatMap((candidateToken, index) => (
      tokenMatches(queryToken, candidateToken, options) ? [index] : []
    ))
  ));
  if (candidateOptions.some((matches) => matches.length === 0)) return false;
  candidateOptions.sort((left, right) => left.length - right.length);
  const used = new Set<number>();
  const assign = (queryIndex: number): boolean => {
    if (queryIndex === candidateOptions.length) return true;
    for (const candidateIndex of candidateOptions[queryIndex]) {
      if (used.has(candidateIndex)) continue;
      used.add(candidateIndex);
      if (assign(queryIndex + 1)) return true;
      used.delete(candidateIndex);
    }
    return false;
  };
  return assign(0);
}

function scoreRow(row: PreparedPaperSearchRow, query: string, queryTokens: string[]): number {
  const containsPhrase = (value: string) => (
    value === query
    || value.startsWith(`${query} `)
    || value.endsWith(` ${query}`)
    || value.includes(` ${query} `)
  );
  if (row.title === query) return 1_400;
  if (row.authors === query) return 1_350;
  if (row.title.startsWith(query)) return 1_250;
  if (row.authors.startsWith(query)) return 1_200;
  if (containsPhrase(row.authors)) return 1_175;
  if (tokensMatch(queryTokens, row.authorTokens)) return 1_150 + queryTokens.length;
  if (containsPhrase(row.title)) return 1_100;
  if (tokensMatch(queryTokens, row.titleTokens)) return 1_050 + queryTokens.length;
  if (tokensMatch(queryTokens, row.authorTokens, { allowPrefix: true, allowInitial: true })) {
    return 950 + queryTokens.length;
  }
  if (tokensMatch(queryTokens, row.titleTokens, { allowPrefix: true })) return 900 + queryTokens.length;
  if (tokensMatch(queryTokens, row.authorTokens, { allowPrefix: true, allowInitial: true, allowFuzzy: true })) {
    return 850 + queryTokens.length;
  }
  if (tokensMatch(queryTokens, row.titleTokens, { allowPrefix: true, allowFuzzy: true })) {
    return 800 + queryTokens.length;
  }
  const combinedTokens = row.titleTokens.concat(row.authorTokens);
  if (tokensMatch(queryTokens, combinedTokens)) return 750 + queryTokens.length;
  if (tokensMatch(queryTokens, combinedTokens, { allowPrefix: true, allowInitial: true })) {
    return 650 + queryTokens.length;
  }
  if (tokensMatch(queryTokens, combinedTokens, { allowPrefix: true, allowInitial: true, allowFuzzy: true })) {
    return 600 + queryTokens.length;
  }
  return -1;
}

function vocabularyMatches(prepared: PreparedPaperSearch, queryToken: string): string[] {
  if (queryToken.length === 1) return [];
  const direct: string[] = [];
  for (const token of prepared.tokenBuckets.get(queryToken.slice(0, 2)) ?? []) {
    if (tokensEquivalent(queryToken, token) || token.startsWith(queryToken)) direct.push(token);
  }
  if (direct.length || queryToken.length < 4) return direct;
  const fuzzy: string[] = [];
  for (let length = queryToken.length - 1; length <= queryToken.length + 1; length += 1) {
    for (const token of prepared.lengthBuckets.get(length) ?? []) {
      if (token.length >= 4 && withinOneEdit(queryToken, token)) fuzzy.push(token);
    }
  }
  return fuzzy;
}

export function searchPreparedPapers(
  prepared: PreparedPaperSearch,
  rawQuery: string,
  limit = 8,
): number[] {
  const query = normalizePaperSearch(rawQuery);
  if (query.length < 2 || limit <= 0) return [];
  const queryTokens = query.split(' ').filter(Boolean);
  if (!queryTokens.some((token) => token.length >= 2)) return [];
  const candidateLists: number[][] = [];
  for (const queryToken of queryTokens) {
    if (queryToken.length === 1) continue;
    const matches = vocabularyMatches(prepared, queryToken);
    if (!matches.length) return [];
    const postings = matches.flatMap((token) => prepared.tokenIndex.get(token) ?? []);
    candidateLists.push([...new Set(postings)]);
  }
  if (!candidateLists.length) return [];
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
