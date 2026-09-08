import ast
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from audio_features import (
    load_audio, extract_chroma, fixed_window_segments,
    extract_log_mel_spectrogram, pad_or_crop_mel,
)


def _track_id_to_path(audio_dir: str, track_id: int) -> str:
    tid_str = f"{track_id:06d}"
    return os.path.join(audio_dir, tid_str[:3], tid_str + ".mp3")


def _load_tracks_csv(metadata_dir: str) -> pd.DataFrame:
    path = os.path.join(metadata_dir, "tracks.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Download+unzip fma_metadata.zip into {metadata_dir} first "
            f"(see README.md for the exact commands)."
        )
    return pd.read_csv(path, index_col=0, header=[0, 1])


def _build_genre_vocab(tracks: pd.DataFrame) -> list:
    genres = tracks[("track", "genre_top")].dropna().unique().tolist()
    return sorted(genres)


def _pseudo_caption(row) -> str:
    title = row.get(("track", "title"), "")
    artist = row.get(("artist", "name"), "")
    parts = [p for p in [str(title) if pd.notna(title) else "",
                          str(artist) if pd.notna(artist) else ""] if p]
    return ", ".join(parts) if parts else "unknown track"


def load_fma_corpus(cfg: dict):
    fma_cfg = cfg["fma"]
    audio_cfg = cfg["audio"]

    tracks = _load_tracks_csv(fma_cfg["metadata_dir"])
    subset_mask = tracks[("set", "subset")].isin(
        {"small": ["small"], "medium": ["small", "medium"], "large": ["small", "medium", "large"]}[fma_cfg["subset"]]
    )
    tracks = tracks[subset_mask]
    tracks = tracks[tracks[("track", "genre_top")].notna()]

    tag_vocab = _build_genre_vocab(tracks)
    genre_to_idx = {g: i for i, g in enumerate(tag_vocab)}

    os.makedirs(fma_cfg["cache_dir"], exist_ok=True)

    corpus = []
    track_ids = tracks.index.tolist()[: fma_cfg["max_tracks"]]

    for track_id in tqdm(track_ids, desc="Loading FMA tracks"):
        cache_path = os.path.join(fma_cfg["cache_dir"], f"{track_id}.npy")

        if os.path.exists(cache_path):
            segment_features = list(np.load(cache_path))
        else:
            audio_path = _track_id_to_path(fma_cfg["audio_dir"], track_id)
            if not os.path.exists(audio_path):
                continue  
            try:
                y, sr = load_audio(audio_path, sr=audio_cfg["sample_rate"])
                y = y[: int(fma_cfg["clip_seconds"] * sr)]
                chroma = extract_chroma(y, sr=sr, hop_length=audio_cfg["hop_length"])
                segment_features = fixed_window_segments(
                    chroma, sr=sr, hop_length=audio_cfg["hop_length"],
                    seg_len_sec=audio_cfg["segment_length_sec"],
                )
            except Exception as e:
                print(f"Skipping track {track_id}: {e}")
                continue
            if len(segment_features) == 0:
                continue
            np.save(cache_path, np.stack(segment_features))

        row = tracks.loc[track_id]
        genre = row[("track", "genre_top")]
        tag_vector = np.zeros(len(tag_vocab), dtype=np.float32)
        tag_vector[genre_to_idx[genre]] = 1.0

        corpus.append({
            "track_id": f"fma_{track_id}",
            "segment_features": segment_features,
            "tags": tag_vector,
            "caption": _pseudo_caption(row),
            "valence": 5.0,   
            "arousal": 5.0,
        })

    if len(corpus) == 0:
        raise RuntimeError(
            "Loaded 0 tracks. Check fma.metadata_dir / fma.audio_dir in config.yaml and that "
            "fma_medium.zip (or fma_small.zip) was unzipped alongside fma_metadata.zip."
        )

    print(f"Loaded {len(corpus)} FMA tracks across {len(tag_vocab)} genres: {tag_vocab}")
    return corpus, tag_vocab


def load_fma_mel_for_corpus(cfg: dict, corpus_items: list) -> dict:
    fma_cfg = cfg["fma"]
    audio_cfg = cfg["audio"]
    mel_cache_dir = fma_cfg.get("mel_cache_dir") or os.path.join(fma_cfg["cache_dir"], "..", "fma_mel_cache")
    os.makedirs(mel_cache_dir, exist_ok=True)
    max_frames = audio_cfg.get("cnn_max_frames", 130)

    mel_features = {}
    for item in tqdm(corpus_items, desc="Loading FMA mel-spectrograms"):
        track_id_str = item["track_id"]
        if not track_id_str.startswith("fma_"):
            continue  

        raw_id = int(track_id_str.split("_", 1)[1])
        cache_path = os.path.join(mel_cache_dir, f"{raw_id}.npy")

        if os.path.exists(cache_path):
            mel = np.load(cache_path)
        else:
            audio_path = _track_id_to_path(fma_cfg["audio_dir"], raw_id)
            if not os.path.exists(audio_path):
                continue
            try:
                y, sr = load_audio(audio_path, sr=audio_cfg["sample_rate"])
                y = y[: int(fma_cfg["clip_seconds"] * sr)]
                mel = extract_log_mel_spectrogram(
                    y, sr=sr, n_mels=audio_cfg["n_mels"],
                    n_fft=audio_cfg["n_fft"], hop_length=audio_cfg["hop_length"],
                )
                mel = pad_or_crop_mel(mel, max_frames)
            except Exception as e:
                print(f"Skipping mel-spectrogram for track {raw_id}: {e}")
                continue
            np.save(cache_path, mel)

        mel_features[track_id_str] = mel

    return mel_features
