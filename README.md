# Text Encoding for NLP (Assignment 2)

Compares one-hot, TF-IDF and Word2Vec for finding semantically similar sentences in the emotion multigenre corpus (news headlines, movie reviews and blogs). For each query sentence, the 10 most similar sentences are retrieved with cosine similarity and compared across methods.

## Files

| File | Description |
|---|---|
| `text_vectorizing_and_semantic_similarity.ipynb` | Main notebook. The provided sample code is at the top, followed by preprocessing, the three encoders, and both experiments with all top-10 lists. |
| `report_analysis.ipynb` | Results used in the report (Figures A1, C1, C2, E1, E2, F5). |
| `additional_results.ipynb` | Supporting results (all other figures and Table T1). |
| `figures/` | Figures and tables saved by the two analysis notebooks. |
| `main.tex`, `references.bib` | LaTeX report (Overleaf). |
| `emotion_multigenre_corpus_setences.txt` | Corpus, one sentence per line (used). |
| `emotion_multigenre_corpus_clauses.txt` | Corpus split into clauses (only used to compare lengths). |
| `mallet_en_stoplist.txt` | Mallet stopword list. |

## Setup

```bash
conda install anaconda::gensim
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter
```

## How to run

Put the notebooks and data files in the same folder, then run:

1. `text_vectorizing_and_semantic_similarity.ipynb` (about 1 minute)
2. `report_analysis.ipynb` (about 5 minutes)
3. `additional_results.ipynb` (about 5 minutes)

The two analysis notebooks download pretrained GloVe vectors (`glove-wiki-gigaword-100`, about 128 MB) on the first run. All results use seed 42 and a single Word2Vec worker, so they are reproducible.

## Experiments

- **Experiment 1:** all 16,822 sentences are encoded, and 3 queries per genre are compared with the whole corpus.
- **Experiment 2:** 100 random sentences per genre, each genre encoded separately. Each query is compared with the other 99 sentences of its genre. A pooled version (all 300 sentences together) is used for the genre question.
- **Additional checks:** 908 queries with bootstrap confidence intervals, 30 repeated Experiment 2 samples, a learning curve, and GloVe as a reference.

## Main findings

- TF-IDF gave the most similar results when a rare word or name defined the topic. One-hot was close behind.
- Word2Vec trained on this corpus (about 120,000 tokens) mostly learned genre and was less precise than TF-IDF. Pretrained GloVe was much better.
- On the full corpus, most similar sentences came from the query's genre. With 100 sentences per genre, this effect mostly disappeared.
- No method preserved sentiment.
