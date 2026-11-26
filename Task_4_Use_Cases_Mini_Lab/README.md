# Task 4 — Use Cases Mini-Lab

## Run
`pip install -r requirements.txt`, then open `notebook.ipynb` and run all cells.

## Included
1. Semantic search over 15 FAQs with 3 queries; top 3 results per query.
2. Near-duplicate detection across 12 complaint texts using cosine threshold `0.78`.
3. K-Means clustering over 30 headline embeddings into 4 groups.
4. Top 3 recommendations from 15 products for a product description.
5. Bonus precision/recall calculation on a small illustrative labelled sample.

## Important
- Embedding similarities and cluster assignments are computed at runtime and can vary slightly by model/library version.
- K-Means cluster numbers are arbitrary. Inspect the printed headlines and assign a meaningful topic name to each cluster.
- The bonus labels are a starter evaluation sample, not a statistically representative benchmark. Replace them with human-reviewed labels for a reliable threshold evaluation.
