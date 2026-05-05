# Analisi dello stato attuale del progetto e possibili sviluppi futuri

## Executive summary

Il repository è un **prototipo LTR (Learning-to-Rank) ben strutturato** e già utilizzabile per demo/portfolio: ha pipeline training+evaluation, baseline BM25, ranker supervisionato con fallback robusto, reportistica e UI Streamlit.

Allo stato attuale, però, è ancora **non pronto per produzione**: dataset molto piccolo/sintetico, assenza di retrieval scalabile a due stadi, niente tracking esperimenti e governance MLOps.

Dato più importante emerso dai risultati correnti: **il baseline BM25 supera il ranker learned** su NDCG@10 e MAP nel run salvato (2026-05-03), quindi la priorità è migliorare qualità del segnale e protocollo sperimentale prima di evoluzioni architetturali complesse.

---

## 1) Stato attuale (cosa funziona bene)

## 1.1 Architettura e codice

- Struttura repository ordinata (`src/`, `tests/`, `reports/`, `data/`, `models/`).
- Pipeline separata in moduli chiari: caricamento dati, feature extraction, training, evaluation e ranking.
- Presenza di fallback deterministico (`PairwiseLogisticRanker`) quando XGBoost non è disponibile: scelta utile per riproducibilità locale.

## 1.2 Qualità ingegneristica

- Report automatici in JSON/Markdown per metriche offline.
- Test unitari presenti su componenti core (data loading, feature, metriche, output ranking).
- App Streamlit pronta per confronto side-by-side tra baseline e modello learned.

## 1.3 Valutazione offline disponibile

Dal file `reports/metrics.json` (timestamp `2026-05-03T15:57:36Z`):

| Modello | NDCG@10 | MAP | MRR |
|---|---:|---:|---:|
| BM25 baseline | 0.9877 | 0.9514 | 1.0000 |
| Learned ranker | 0.9783 | 0.8958 | 1.0000 |

Interpretazione pratica:
- MRR uguale (1.0) indica che il primo risultato è spesso corretto in entrambi i sistemi.
- La differenza su MAP/NDCG suggerisce che il learned ranker **ordina peggio i risultati oltre la primissima posizione**.

---

## 2) Limiti attuali (gap da chiudere)

## 2.1 Dati e validità sperimentale

1. Dataset demo molto piccolo (`train_queries=8`, `test_queries=4`, `documents=36` nel report corrente).
2. Alto rischio di varianza e overfitting; una singola run può non essere rappresentativa.
3. Mancanza di split multipli/cross-validation per misurare stabilità delle metriche.

## 2.2 Modellazione ranking

1. Feature set prevalentemente lessicale (BM25/overlap/coverage/proximity semplice).
2. In assenza di XGBoost si usa fallback logistico pairwise, ottimo per demo ma limitato per performance massime.
3. Nessuna analisi sistematica “query-level wins/losses” tra baseline e learned.

## 2.3 MLOps e operatività

1. Assenza di experiment tracking strutturato (config + metriche + artefatti versionati).
2. Nessun controllo formale qualità dati in ingresso (schema/consistenza qrels).
3. Nessun endpoint di serving con SLA/monitoraggio latenza.

---

## 3) Priorità consigliate (ordine reale di impatto)

## Priorità 1 — Stabilizzare misurazione e qualità (subito)

- Aggiungere test di regressione metriche su fixture fisse.
- Introdurre più split di valutazione (o query-group CV) e riportare media + deviazione standard.
- Salvare in report: configurazione completa modello, seed, backend usato, hash dati.

**Outcome atteso:** capire con confidenza se il learned ranker migliora davvero BM25.

## Priorità 2 — Portare il learned ranker sopra la baseline

- Hyperparameter tuning strutturato (anche semplice random search).
- Espansione feature ingegnerizzate (fielded BM25, segnali title/body separati, feature di prossimità più ricche).
- Analisi errori per classi di query (corte/lunghe, navigazionali/informative).

**Outcome atteso:** superare BM25 su NDCG@10 e MAP in modo consistente.

## Priorità 3 — Hardening architetturale

- Pipeline a due stadi: retrieval candidati (stage-1) + reranking (stage-2).
- Definizione di un contratto API (`rank(query, top_k)`), con versione modello esplicita.
- Monitoraggio minimo: latenza p50/p95, failure rate, drift feature base.

**Outcome atteso:** base tecnica per PoC “production-like”.

## Priorità 4 — Evoluzione semantica (dopo i fondamentali)

- Ranking ibrido lexical+dense.
- Reranker neurale leggero su top-N candidati.
- Successiva validazione online (CTR/success rate) in ambienti reali.

**Outcome atteso:** miglioramento su query ambigue, sinonimia e casi long-tail.

---

## 4) Roadmap proposta (8-12 settimane)

### Fase A (settimane 1-2): affidabilità sperimentale
- test regressione metriche
- validazione schema dati + coerenza qrels
- report run completo (config/seed/backend)

### Fase B (settimane 3-5): quality uplift
- tuning modello
- arricchimento feature
- dashboard query-level error analysis

### Fase C (settimane 6-8): packaging e serving
- API ranking
- packaging modello/versioning
- metriche operative (latenza, errori)

### Fase D (settimane 9-12): semantica e hybrid
- embedding retrieval o segnali dense
- hybrid scoring + valutazione comparativa

---

## 5) KPI di avanzamento suggeriti

### Offline relevance
- NDCG@10, MAP, MRR (media + deviazione standard su split multipli)

### Stabilità e robustezza
- varianza run-to-run
- percentuale di query dove learned > BM25

### Efficienza
- latenza p50/p95 ranking
- throughput query/sec

### Operativo
- tasso fallimento pipeline train/eval
- completezza metadati esperimento (config, seed, backend, artefatti)

---

## 6) Decisione strategica consigliata

Nel breve termine, la scelta migliore è: **ottimizzare prima la qualità sperimentale e il modello attuale**, non introdurre subito complessità neurale.

Motivo: i numeri correnti mostrano che il baseline BM25 è ancora superiore; senza una base sperimentale robusta, qualsiasi upgrade avanzato rischia di aumentare complessità senza ROI misurabile.
