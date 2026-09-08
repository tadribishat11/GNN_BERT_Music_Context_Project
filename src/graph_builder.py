import json
import numpy as np
import torch
from torch_geometric.data import Data

from audio_features import chroma_to_chord_sequence


# ---------------------------------------------------------------------------
# Chord-transition graph
# ---------------------------------------------------------------------------
def build_chord_transition_graph(chord_sequence: list) -> Data:
    unique_chords = sorted(set(chord_sequence))
    chord_to_idx = {c: i for i, c in enumerate(unique_chords)}

    edge_counts = {}
    for a, b in zip(chord_sequence[:-1], chord_sequence[1:]):
        key = (chord_to_idx[a], chord_to_idx[b])
        edge_counts[key] = edge_counts.get(key, 0) + 1

    if len(edge_counts) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float)
    else:
        src, dst, weight = [], [], []
        for (a, b), count in edge_counts.items():
            src.append(a)
            dst.append(b)
            weight.append(float(count))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.tensor(weight, dtype=torch.float)
        edge_weight = edge_weight / edge_weight.sum()  # normalize transition weights

    x = _chord_node_features(unique_chords)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight.unsqueeze(-1))
    data.node_labels = unique_chords
    data.graph_type = "chord_transition"
    return data


def _chord_node_features(chords: list) -> torch.Tensor:
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    feats = []
    for c in chords:
        vec = np.zeros(25, dtype=np.float32)
        if c == "N":
            vec[24] = 1.0
        else:
            is_minor = c.endswith("m")
            root = c[:-1] if is_minor else c
            if root in pitch_classes:
                idx = pitch_classes.index(root) + (12 if is_minor else 0)
                vec[idx] = 1.0
        feats.append(vec)
    return torch.tensor(np.stack(feats), dtype=torch.float)


# ---------------------------------------------------------------------------
# Segment-similarity graph
# ---------------------------------------------------------------------------
def build_segment_graph(segment_features: list, similarity_threshold: float = 0.8) -> Data:
    n = len(segment_features)
    feats = np.stack(segment_features) if n > 0 else np.zeros((1, 1), dtype=np.float32)
    x = torch.tensor(feats, dtype=torch.float)

    src, dst, weight = [], [], []

    
    for i in range(n - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
        weight += [1.0, 1.0]

    
    if n > 1:
        norm_feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        sim_matrix = norm_feats @ norm_feats.T
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > similarity_threshold:
                    src += [i, j]
                    dst += [j, i]
                    weight += [float(sim_matrix[i, j]), float(sim_matrix[i, j])]

    if len(src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.tensor(weight, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight.unsqueeze(-1))
    data.graph_type = "segment_similarity"
    return data


def build_graph_from_chroma(chroma: np.ndarray, graph_type: str = "segment", similarity_threshold: float = 0.8, smoothing: int = 4):
    if graph_type == "chord":
        chord_seq = chroma_to_chord_sequence(chroma, smoothing=smoothing)
        return build_chord_transition_graph(chord_seq)
    else:
        segments = [chroma[:, i:i + smoothing].mean(axis=1)
                    for i in range(0, chroma.shape[1], smoothing)
                    if chroma[:, i:i + smoothing].shape[1] > 0]
        return build_segment_graph(segments, similarity_threshold=similarity_threshold)


# ---------------------------------------------------------------------------
# I/O helpers (.pt native + .json human-readable export)
# ---------------------------------------------------------------------------
def save_graph_pt(data: Data, path: str):
    torch.save(data, path)


def load_graph_pt(path: str) -> Data:
    return torch.load(path, weights_only=False)


def save_graph_json(data: Data, path: str):
    obj = {
        "graph_type": getattr(data, "graph_type", "unknown"),
        "num_nodes": int(data.x.shape[0]),
        "node_features": data.x.tolist(),
        "edge_index": data.edge_index.tolist(),
        "edge_weight": data.edge_attr.squeeze(-1).tolist() if data.edge_attr is not None else [],
    }
    if hasattr(data, "node_labels"):
        obj["node_labels"] = data.node_labels
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
