# LDA model review

## Design

The input is each paper's title (included twice) and abstract. The count vocabulary contains 14,104 unigrams and bigrams occurring in at least 20 papers and no more than half of the corpus. English stop words, short and numeric tokens, and recurring publication language were removed. An initial run was rejected because bibliographic boilerplate formed several apparent topics; the filtering was revised before the reported scan.

The 69,400 papers were split deterministically into 55,520 training, 6,940 validation, and 6,940 test papers. The test set did not contribute to fitting or model selection. The first stage compared 20 combinations of topic count (12, 20, 30, 40, 50, 60, 75, or 100) and symmetric or sparse prior settings. Six validation finalists—two at each of broad, intermediate, and detailed resolution—were fitted at three random seeds. The selection criteria were held-out perplexity and NPMI coherence, topic diversity and exclusivity, mean dominant-topic probability, matched topic-word stability, and adjusted Rand agreement between dominant-topic assignments. Selected configurations were finally fitted to the complete corpus for display.

## Selected configurations and sealed-test results

| Topics | Test perplexity ↓ | NPMI coherence ↑ | Diversity ↑ | Exclusivity ↑ | Dominant probability ↑ | Topic stability ↑ | Assignment ARI ↑ | Composite score ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 6,467 | 0.203 | 0.927 | 0.732 | 0.469 | 0.534 | 0.203 | 0.553 |
| 60 | 7,586 | 0.220 | 0.931 | 0.790 | 0.375 | 0.425 | 0.225 | 0.467 |
| 100 | 8,344 | 0.235 | 0.941 | 0.827 | 0.336 | 0.416 | 0.219 | 0.581 |

The 20-topic model provides the best broad summary. It has the lowest perplexity, the highest topic-word stability, and the strongest dominant-topic probabilities. Its lower coherence and exclusivity reflect the breadth of its topics.

The 60-topic model is retained as a more detailed exploratory view. It improves coherence and exclusivity over the 20-topic model and has the highest assignment agreement of the three selected configurations, although its individual paper assignments remain soft and seed-sensitive.

The 100-topic model is not included in the atlas. Its marginal gain in coherence and exclusivity comes with worse perplexity, weaker dominant-topic probabilities, lower topic stability, many small semantically mixed topics, and several large generic catch-alls. Its high composite score does not outweigh those substantive defects.

## Interpretation

The low assignment ARI values (0.20–0.22) are the main limitation. The topics are useful summaries of recurring vocabulary, but they do not form a robust hard partition of the literature. LDA is a mixed-membership model: every paper has a distribution over topics. The atlas necessarily colours a paper by its largest topic weight, so the two LDA views should be read as exploratory thematic overlays rather than alternative taxonomies.

The full first-stage results are in `lda-scan.csv`; model metadata, topic terms, counts, and test metrics are in `lda-selected.json`.
