import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from graph_builder import build_segment_graph, save_graph_pt, save_graph_json

NODE_FEAT_DIM = 20  

SAMPLE_TAGS = [
    "jazz", "rock", "electronic", "pop", "hiphop", "classical", "acoustic",
    "melancholic", "energetic", "calm", "guitar", "piano", "synth", "drums",
    "vocal", "instrumental", "1960s", "1980s", "lofi", "ambient",
]

SAMPLE_CAPTIONS = [
    "a melancholic piano ballad with soft vocals and slow tempo",
    "an upbeat electronic track with driving synth bass and four-on-the-floor drums",
    "acoustic guitar fingerpicking with warm mellow tone, folk influence",
    "energetic rock song with distorted electric guitar and powerful drums",
    "ambient soundscape with sustained pads and no percussion",
    "jazz trio with walking bass, brushed drums, and improvised saxophone",
    "lo-fi hip hop beat with vinyl crackle and mellow piano chords",
    "classical string quartet performing a slow expressive passage",
]


# ---------------------------------------------------------------------------
# Synthetic sample generation (segment graph + caption/tags + emotion)
# ---------------------------------------------------------------------------
def _synthetic_track(seed: int, num_segments: int = 12, num_tags: int = len(SAMPLE_TAGS)):
    rng = np.random.RandomState(seed)
    segment_features = [rng.normal(0, 1, size=NODE_FEAT_DIM).astype(np.float32)for _ in range(num_segments)]

    active_tags = rng.choice(num_tags, size=rng.randint(1, 5), replace=False)
    tag_vector = np.zeros(num_tags, dtype=np.float32)
    tag_vector[active_tags] = 1.0

    caption = SAMPLE_CAPTIONS[seed % len(SAMPLE_CAPTIONS)]
    valence = float(np.clip(rng.normal(5.0, 2.0), 1, 9))
    arousal = float(np.clip(rng.normal(5.0, 2.0), 1, 9))

    return segment_features, tag_vector, caption, valence, arousal


def generate_synthetic_corpus(n_tracks: int = 200, seed: int = 42):
    random.seed(seed)
    corpus = []
    for i in range(n_tracks):
        seg_feats, tags, caption, valence, arousal = _synthetic_track(seed + i)
        corpus.append({
            "track_id": f"synth_{i:04d}",
            "segment_features": seg_feats,
            "tags": tags,
            "caption": caption,
            "valence": valence,
            "arousal": arousal,
        })
    return corpus


def generate_synthetic_mel_for_corpus(corpus, n_mels: int = 128, max_frames: int = 130, seed: int = 42):
    mel_features = {}
    for i, item in enumerate(corpus):
        rng = np.random.RandomState(seed + i + 99991)  # offset so it differs from segment_features' rng
        mel_features[item["track_id"]] = rng.normal(0, 1, size=(n_mels, max_frames)).astype(np.float32)
    return mel_features


def export_sample_graphs(corpus, out_dir: str, n_samples: int = 20,similarity_threshold: float = 0.8):
    os.makedirs(out_dir, exist_ok=True)
    for i, item in enumerate(corpus[:n_samples]):
        graph = build_segment_graph(item["segment_features"], similarity_threshold)
        save_graph_pt(graph, os.path.join(out_dir, f"{item['track_id']}.pt"))
        save_graph_json(graph, os.path.join(out_dir, f"{item['track_id']}.json"))


# ---------------------------------------------------------------------------
# Task 1: BERT tag/caption classifier dataset
# ---------------------------------------------------------------------------
class MusicTagTextDataset(Dataset):

    def __init__(self, corpus):
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        item = self.corpus[idx]
        return item["caption"], torch.tensor(item["tags"], dtype=torch.float)


def collate_tag_text(batch, tokenizer, max_length=128):
    texts, tags = zip(*batch)
    enc = tokenizer(list(texts), padding=True, truncation=True,
                     max_length=max_length, return_tensors="pt")
    tag_tensor = torch.stack(tags)
    return enc["input_ids"], enc["attention_mask"], tag_tensor


# ---------------------------------------------------------------------------
# Task 2: Graph-only dataset (audio structure graphs)
# ---------------------------------------------------------------------------
class MusicGraphDataset(Dataset):

    def __init__(self, corpus, similarity_threshold: float = 0.8):
        self.corpus = corpus
        self.similarity_threshold = similarity_threshold

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        item = self.corpus[idx]
        graph = build_segment_graph(item["segment_features"], self.similarity_threshold)
        graph.y = torch.tensor(item["tags"], dtype=torch.float).unsqueeze(0)
        return graph


def collate_graphs(batch):
    graph_batch = Batch.from_data_list(batch)
    targets = torch.cat([g.y for g in batch], dim=0)
    return graph_batch, targets


# ---------------------------------------------------------------------------
# Task 3: Fusion dataset (graph + text + tags + valence/arousal)
# ---------------------------------------------------------------------------
class MusicFusionDataset(Dataset):
    def __init__(self, corpus, similarity_threshold: float = 0.8):
        self.corpus = corpus
        self.similarity_threshold = similarity_threshold

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        item = self.corpus[idx]
        graph = build_segment_graph(item["segment_features"], self.similarity_threshold)
        return {
            "graph": graph,
            "caption": item["caption"],
            "tags": torch.tensor(item["tags"], dtype=torch.float),
            "valence": torch.tensor(item["valence"], dtype=torch.float),
            "arousal": torch.tensor(item["arousal"], dtype=torch.float),
        }


def collate_fusion(batch, tokenizer, max_length=128):
    graphs = [b["graph"] for b in batch]
    graph_batch = Batch.from_data_list(graphs)

    texts = [b["caption"] for b in batch]
    enc = tokenizer(texts, padding=True, truncation=True,
                     max_length=max_length, return_tensors="pt")

    tags = torch.stack([b["tags"] for b in batch])
    valence = torch.stack([b["valence"] for b in batch])
    arousal = torch.stack([b["arousal"] for b in batch])

    return {
        "graph_batch": graph_batch,
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "tags": tags,
        "valence": valence,
        "arousal": arousal,
    }


# ---------------------------------------------------------------------------
# Task 4: Contrastive (graph, caption) pair dataset
# ---------------------------------------------------------------------------
class MusicCaptionPairDataset(Dataset):
    def __init__(self, corpus, similarity_threshold: float = 0.8):
        self.corpus = corpus
        self.similarity_threshold = similarity_threshold

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        item = self.corpus[idx]
        graph = build_segment_graph(item["segment_features"], self.similarity_threshold)
        return graph, item["caption"], item["track_id"]


def collate_pairs(batch, tokenizer, max_length=128):
    graphs, captions, track_ids = zip(*batch)
    graph_batch = Batch.from_data_list(graphs)
    enc = tokenizer(list(captions), padding=True, truncation=True,
                     max_length=max_length, return_tensors="pt")
    return graph_batch, enc["input_ids"], enc["attention_mask"], list(track_ids), list(captions)


# ---------------------------------------------------------------------------
# Train/val/test split helper 
# ---------------------------------------------------------------------------
def train_val_test_split(corpus, val_frac=0.15, test_frac=0.15, seed=42):
    rng = random.Random(seed)
    items = list(corpus)
    rng.shuffle(items)
    n = len(items)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val = items[:n_val]
    test = items[n_val:n_val + n_test]
    train = items[n_val + n_test:]
    return train, val, test


def save_splits(train, val, test, splits_dir: str):
    os.makedirs(splits_dir, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        ids = [item["track_id"] for item in split]
        with open(os.path.join(splits_dir, f"{name}.json"), "w") as f:
            json.dump(ids, f, indent=2)
