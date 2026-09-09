# GNN-Based BERT for Understanding Context from Music

Implementation of the project spec: a hybrid **BERT + Graph
Neural Network** system for music context understanding (multi-label tagging,
emotion regression, cross-modal alignment), covering all four tasks in the
project roadmap.

| Task | Model | File(s) |
|---|---|---|
| 1 (Easy) | BERT multi-label tag classifier | `src/bert_encoder.py` |
| 2 (Medium) | GraphSAGE/GAT on chord/segment graphs | `src/gnn_model.py` |
| 3 (Hard) | Cross-attention GNN-BERT fusion + multi-task loss | `src/fusion_model.py` |
| 4 (Advanced) | Contrastive dual-encoder (InfoNCE) MusicCaps alignment | `src/contrastive.py` |

## Setup

```bash
pip install -r requirements.txt
```

`torch`, `torch-geometric`, and `transformers` require network access; if the
HuggingFace Hub isn't reachable in your environment, pre-download
`distilbert-base-uncased` / `bert-base-uncased` and point `text.bert_model_name`
in `config.yaml` at the local cache path.

## Data

Download at least one **audio** dataset and one **text/tag** dataset from
Table 1 of the spec (FMA, MagnaTagATune, GTZAN, DEAM, MusicCaps, MSD, Lakh
MIDI, EmoMusic) into `data/raw/`. This repository
does not ship any licensed/copyrighted audio.


### Using real FMA data 

1. Download and unzip into `data/raw/`:
   ```bash
   wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
   wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
   unzip fma_metadata.zip -d data/raw/fma_metadata
   unzip fma_small.zip   -d data/raw/fma_small
   ```
2. Check `fma:` in `config.yaml` matches your paths/subset (`small`) and
   set `max_tracks` for how many tracks to process on a first run.
3. Run with `--loader fma`:
   ```bash
   python train.py --task 2 --loader fma
   python train.py --task 3 --loader fma --fusion_mode cross_attention
   ```

`src/load_fma.py` reads `tracks.csv`, extracts 12-D chroma segment features
per track (cached to `data/processed/fma_cache/*.npy` so re-runs are fast),
builds a multi-hot genre vector from `genre_top` as the tag target, and a
short pseudo-caption from title/artist/genre metadata as BERT input (FMA has
no MusicCaps-style natural-language captions — Table 1 lists it as a
"top tags / metadata" text source). `train.py` now infers `num_tags` and the
GNN's `node_in_dim` directly from whatever corpus is loaded, so the FMA
genre vocabulary and 12-D chroma features plug in without touching model code.
FMA has no valence/arousal labels; those are set to a neutral placeholder —
layer DEAM on top if you want real emotion regression (Section 3's "Emotion
extension"). For any other dataset in Table 1, write an equivalent
`load_<dataset>.py` that returns the same corpus schema and pass it via
`prepare_corpus(cfg, loader=...)`.

### Preprocessing pipeline (`src/audio_features.py`, `src/graph_builder.py`)
1. Resample audio to 22,050 Hz.
2. Extract log-mel spectrogram (128 bins) or chroma (12 bins); normalize per track.
3. Segment into fixed 5-10s windows or beat-synchronous segments (`librosa`).
4. Build graphs:
   - **Chord-transition graph**: nodes = unique chords (template-matched from
     chroma), edges = observed transitions weighted by count.
   - **Segment graph**: nodes = time segments, edges = temporal adjacency +
     cosine similarity of MFCC/chroma above `graph.similarity_threshold`.
5. Tokenize lyrics/tags/captions with the BERT tokenizer (max length 128-256).
6. Split train/val/test with no artist leakage (`train_val_test_split`, splits saved to `data/splits/`).

## Training

```bash
python train.py --task 1                                   # BERT tag baseline
python train.py --task 2                                   # GraphSAGE/GAT on structure graphs
python train.py --task 3 --fusion_mode cross_attention      # full fusion model
python train.py --task 3 --fusion_mode concat               # ablation: early concat
python train.py --task 3 --fusion_mode bert_only            # ablation: BERT-only
python train.py --task 3 --fusion_mode gnn_only             # ablation: GNN-only
python train.py --task 4                                    # contrastive MusicCaps alignment
```

Each run: prepares the corpus + splits, exports 20 example graphs (`.pt` +
`.json`) to `data/processed/sample_graphs/`, trains, evaluates on the held-out
split every epoch, and writes checkpoints to `results/checkpoints/`, plots to
`results/plots/`, and metrics to `results/metrics.json`.

## Evaluation (`src/evaluate.py`)
- Tag/genre classification: per-tag Precision/Recall/F1, Macro-F1, Micro-F1, mean AUC-PR.
- Emotion regression (DEAM): MAE, R².
- Optional graph coherence score `S_graph`.
- Retrieval: Caption->Audio / Audio->Caption R@1, R@5, R@10 (`src/contrastive.py: retrieval_eval`).
- Baselines (Section 8): B1 majority-class, B2 CNN-on-mel-spectrogram
  (`gnn_model.CNNBaseline`), B3 = Task 1 BERT-only, B4 (optional) PCA+MLP —
  all in `src/evaluate.py`.
- Plotting helpers: F1-vs-epoch curves, t-SNE of fused embeddings `z`, retrieval bar charts.

## Demo

`notebooks/demo_context.ipynb` loads a trained Task 3 checkpoint (or falls
back to random init if none exists yet) and runs one end-to-end inference:
audio structure graph + caption -> predicted tags + valence/arousal, plus a
cross-attention visualization. `notebooks/eda.ipynb` explores tag frequency,
valence/arousal spread, and segment-graph statistics.

## Project structure

```
gnn-bert-music-context/
  README.md
  requirements.txt
  config.yaml
  data/
    raw/            # FMA, MagnaTagATune, MusicCaps downloads (not included)
    processed/       # graphs, mel-spec, BERT caches (sample_graphs/ pre-populated)
    splits/          # train/val/test JSON
  notebooks/
    eda.ipynb
    demo_context.ipynb
  src/
    audio_features.py   # mel, chroma, segmentation
    graph_builder.py     # chord + segment graphs
    bert_encoder.py
    gnn_model.py          # GraphSAGE / GAT
    fusion_model.py        # cross-attention GNN-BERT
    contrastive.py          # Task 4 InfoNCE
    dataset.py               # PyTorch Datasets + synthetic corpus generator
    load_fma.py               # real FMA data loader (--loader fma)
    evaluate.py               # metrics, baselines, plots
  train.py
  results/
    metrics.json
    plots/
    retrieval_examples/
  report/
    final_report.pdf   # write-up (NeurIPS/IEEE/ICML template)
```

