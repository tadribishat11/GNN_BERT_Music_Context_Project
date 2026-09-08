import numpy as np
import librosa


# ---------------------------------------------------------------------------
# Loading / resampling
# ---------------------------------------------------------------------------
def load_audio(path: str, sr: int = 22050):
    y, orig_sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_log_mel_spectrogram(y: np.ndarray, sr: int = 22050, n_mels: int = 128,n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                          n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return normalize_features(log_mel)


def extract_chroma(y: np.ndarray, sr: int = 22050, n_chroma: int = 12,
                    hop_length: int = 512) -> np.ndarray:
    
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_chroma=n_chroma)
    return normalize_features(chroma)


def extract_mfcc(y: np.ndarray, sr: int = 22050, n_mfcc: int = 20,
                  hop_length: int = 512) -> np.ndarray:
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return normalize_features(mfcc)


def normalize_features(feat: np.ndarray) -> np.ndarray:
    
    mean = feat.mean(axis=1, keepdims=True)
    std = feat.std(axis=1, keepdims=True) + 1e-8
    return (feat - mean) / std


def pad_or_crop_mel(mel: np.ndarray, max_frames: int) -> np.ndarray:
    
    n_mels, t = mel.shape
    if t >= max_frames:
        return mel[:, :max_frames]
    pad = np.zeros((n_mels, max_frames - t), dtype=mel.dtype)
    return np.concatenate([mel, pad], axis=1)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def fixed_window_segments(feat: np.ndarray, sr: int, hop_length: int,seg_len_sec: float = 5.0):
    
    frames_per_sec = sr / hop_length
    seg_len_frames = max(1, int(round(seg_len_sec * frames_per_sec)))
    segments = []
    for start in range(0, feat.shape[1], seg_len_frames):
        window = feat[:, start:start + seg_len_frames]
        if window.shape[1] == 0:
            continue
        segments.append(window.mean(axis=1))
    return segments


def beat_synchronous_segments(y: np.ndarray, sr: int, feat: np.ndarray, hop_length: int = 512):
    
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_frames = librosa.util.fix_frames(beat_frames, x_min=0, x_max=feat.shape[1])
    segments = []
    for i in range(len(beat_frames) - 1):
        window = feat[:, beat_frames[i]:beat_frames[i + 1]]
        if window.shape[1] == 0:
            continue
        segments.append(window.mean(axis=1))
    return segments


def segment_track(y: np.ndarray, sr: int, feat: np.ndarray, hop_length: int = 512, seg_len_sec: float = 5.0, use_beat_sync: bool = False):
   
    if use_beat_sync:
        return beat_synchronous_segments(y, sr, feat, hop_length)
    return fixed_window_segments(feat, sr, hop_length, seg_len_sec)


# ---------------------------------------------------------------------------
# Simple template-based chord estimation (used by graph_builder.py)
# ---------------------------------------------------------------------------
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


_MAJOR_TEMPLATE = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
_MINOR_TEMPLATE = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])


def chroma_frame_to_chord_label(chroma_frame: np.ndarray) -> str:
    
    best_label, best_score = "N", -1.0
    for shift in range(12):
        root = _PITCH_CLASSES[shift]
        maj = np.roll(_MAJOR_TEMPLATE, shift)
        minr = np.roll(_MINOR_TEMPLATE, shift)
        for template, suffix in [(maj, ""), (minr, "m")]:
            denom = (np.linalg.norm(chroma_frame) * np.linalg.norm(template) + 1e-8)
            score = float(np.dot(chroma_frame, template) / denom)
            if score > best_score:
                best_score = score
                best_label = root + suffix
    return best_label


def chroma_to_chord_sequence(chroma: np.ndarray, smoothing: int = 4) -> list:
    
    chroma_pos = chroma - chroma.min()
    chords = []
    for start in range(0, chroma_pos.shape[1], smoothing):
        block = chroma_pos[:, start:start + smoothing]
        if block.shape[1] == 0:
            continue
        frame = block.mean(axis=1)
        chords.append(chroma_frame_to_chord_label(frame))
    return chords
