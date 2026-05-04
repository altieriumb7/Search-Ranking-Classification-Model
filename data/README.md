This folder contains a small synthetic/demo search ranking dataset.

The dataset is intentionally compact so the full learning-to-rank pipeline can
be trained, evaluated, tested, and demonstrated locally without downloading a
large external benchmark. It is not MS MARCO, LETOR, or production traffic.

Files:
- `queries.csv`: query text and train/test split.
- `documents.csv`: candidate document titles and bodies.
- `qrels.csv`: query-document relevance judgments on a 0-3 ordinal scale.

Relevance scale:
- 3: highly relevant
- 2: relevant
- 1: marginally relevant
- 0: not relevant
