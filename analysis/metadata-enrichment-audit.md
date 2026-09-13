# Bibliographic metadata audit

## Conclusion

Publication type and publication period can be used as atlas lenses now. Journal,
publisher, and co-authorship cannot yet be represented with adequate coverage and
reliability. The current PhilPapers workbook has no journal, publisher, source, or
contributor values, and its `creator` field is effectively a single display name
rather than a complete author list.

The atlas therefore adds only the two fields supported by the source data. Venue
and co-authorship remain a proposed enrichment project, not inferred facts.

## Source-field audit

The workbook was aligned to the fixed 69,400-paper cohort through
`english_indices.npy` and `document_ids.txt`, using the same order as the map.
Duplicate metadata rows were resolved by retaining the row with the longest
abstract, matching the existing map-data build.

| Field | Usable records | Assessment |
| --- | ---: | --- |
| Publication type | 69,397 | Suitable after normalising article, book, and review values |
| Publication period | 69,009 | Suitable after grouping years and manuscript/forthcoming records |
| Creator | 69,390 | Suitable for display and first-author search; not a complete author list |
| Journal/source | 0 | Absent |
| Publisher | 0 | Absent |
| Contributor | 0 | Absent |

Among 69,390 non-empty creator strings, 69,278 contain exactly one comma, only
four contain “and”, and none uses a semicolon-separated author list. A creator
string therefore cannot be converted into a defensible co-author count.

## Published metadata lenses

The publication-type lens contains 62,843 articles, 3,364 books, and 3,190
reviews; three papers have no usable value. The publication-period lens contains
1,234 papers before 1990, 2,327 from the 1990s, 7,322 from the 2000s, 29,488 from
the 2010s, 21,000 from 2020 onward, and 7,638 marked manuscript or forthcoming;
391 papers have no usable period.

These are descriptive metadata categories, not topic models. Missing metadata is
shown in grey on the map and excluded pairwise from association calculations.

## External-enrichment pilot

A deterministic 108-paper sample was drawn with six papers from each available
publication-type × period stratum. Crossref was queried first; records without an
accepted Crossref match were then searched in OpenAlex. Acceptance required:

- title similarity of at least 0.94;
- matching author surname when the candidate supplied authors;
- dates within two years when both sources supplied a year;
- a compatible publication type; and
- exclusion of circular OpenAlex records whose source was PhilPapers itself.

Only 23 of 108 records (21.3%) passed: 18 through Crossref and five through
OpenAlex. The Wilson 95% interval for this deliberately balanced pilot is
14.6–29.9%. Results differed sharply by source type:

| Source type | Accepted | Sample | Rate | Wilson 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Article | 16 | 36 | 44.4% | 29.5–60.4% |
| Book | 1 | 36 | 2.8% | 0.5–14.2% |
| Review | 6 | 36 | 16.7% | 7.9–31.9% |

Every accepted candidate supplied a venue and publisher, but that is conditional
on the small matched subset and is not corpus coverage. Only two accepted records
had more than one listed author. The detailed, reproducible pilot output is stored
in `metadata-enrichment-pilot.json`.

The result is strong evidence against immediately publishing journal, publisher,
or co-authorship lenses. A partial lens would chiefly distinguish records that
Crossref and OpenAlex happen to index, conflating bibliographic coverage with the
phenomenon being studied.

## A defensible route to richer metadata

1. Seek an authorised PhilPapers export containing DOI, venue, and complete author
   lists, or written permission for a bulk API/OAI workflow. PhilPapers’ current
   [API documentation](https://philpapers.org/help/api),
   [OAI documentation](https://philpapers.org/help/oai.html), and
   [terms](https://philpapers.org/help/terms.html) do not support an unapproved
   scripted harvest of the full corpus.
2. Resolve DOI matches first, then use conservative title/author/year matching for
   records without DOI. Crossref documents its open
   [REST API](https://support.crossref.org/hc/en-us/articles/214320426-REST-API)
   and [bibliographic query](https://support.crossref.org/hc/en-us/articles/213942866-Author-article-title-query);
   OpenAlex provides a downloadable CC0 snapshot and
   [API access](https://help.openalex.org/api/).
3. Conduct a blinded manual audit of accepted and rejected matches, stratified by
   source type and period. Proposed publication thresholds are at least 98%
   precision and 80% coverage overall, with no major stratum below 70% coverage.
4. Version the resolved identifiers, match evidence, source dates, and rejection
   reasons. Re-run the association matrices only after the enriched cohort passes
   those checks.

The external sources are technically usable and legally preferable to scraping,
but the pilot shows that title search alone is not sufficient. DOI-bearing or
authorised source metadata is the key missing input.
