import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_fscore_support, average_precision_score, f1_score
)


# ---------------------------------------------------------------------------
# Tag / genre classification metrics
# ---------------------------------------------------------------------------
def multilabel_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    auc_prs = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() > 0:
            auc_prs.append(average_precision_score(y_true[:, k], y_prob[:, k]))
    mean_auc_pr = float(np.mean(auc_prs)) if auc_prs else 0.0

    return {
        "precision_per_tag": precision.tolist(),
        "recall_per_tag": recall.tolist(),
        "f1_per_tag": f1.tolist(),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "mean_auc_pr": mean_auc_pr,
    }


# ---------------------------------------------------------------------------
# Emotion regression metrics (DEAM valence/arousal)
# ---------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Graph coherence score 
# ---------------------------------------------------------------------------
def graph_coherence_score(node_embeddings: torch.Tensor, edge_index: torch.Tensor, tau: float = 0.5) -> float:
    if edge_index.shape[1] == 0:
        return 0.0
    h = torch.nn.functional.normalize(node_embeddings, dim=-1)
    src, dst = edge_index[0], edge_index[1]
    cos_sim = (h[src] * h[dst]).sum(dim=-1)
    coherent = (cos_sim > tau).float().mean().item()
    return coherent


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_f1_curve(history: dict, out_path: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["macro_f1"], label="Macro-F1", marker="o")
    plt.plot(history["epoch"], history["micro_f1"], label="Micro-F1", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")
    plt.title("Tag Classification F1 vs. Training Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_tsne(z: np.ndarray, labels: list, out_path: str, title: str = "t-SNE of fused embeddings z"):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    n = z.shape[0]
    perplexity = max(2, min(30, n // 3))
    z_2d = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=42).fit_transform(z)

    plt.figure(figsize=(6, 6))
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(z_2d[idx, 0], z_2d[idx, 1], label=str(lab), s=20, alpha=0.7)
    plt.legend(fontsize=7, markerscale=1.5, loc="best")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Baselines (Section 8): B1 majority-class, B4 PCA+MLP 
# ---------------------------------------------------------------------------
def majority_class_baseline(y_train: np.ndarray, n_test: int) -> np.ndarray:
    tag_freq = y_train.mean(axis=0, keepdims=True)   # (1, K)
    return np.repeat(tag_freq, n_test, axis=0)


def pca_mlp_baseline(X_train, y_train, X_test, n_components: int = 32, hidden_dim: int = 64, epochs: int = 30, lr: float = 1e-3, seed: int = 42):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(n_components, X_train.shape[1]), random_state=seed)
    Xtr = pca.fit_transform(X_train)
    Xte = pca.transform(X_test)

    torch.manual_seed(seed)
    mlp = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, y_train.shape[1]),
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float)
    ytr_t = torch.tensor(y_train, dtype=torch.float)

    for _ in range(epochs):
        opt.zero_grad()
        logits = mlp(Xtr_t)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ytr_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        probs = torch.sigmoid(mlp(torch.tensor(Xte, dtype=torch.float))).numpy()
    return probs


def build_results_table(model_results: dict, out_path: str = None) -> "pandas.DataFrame":
    import pandas as pd
    df = pd.DataFrame(model_results).T
    df.index.name = "Model"
    if out_path:
        df.to_csv(out_path)
    return df


def plot_retrieval_bar(retrieval_results: dict, out_path: str):
    import matplotlib.pyplot as plt
    keys = list(retrieval_results.keys())
    values = [retrieval_results[k] for k in keys]
    plt.figure(figsize=(7, 4))
    plt.bar(keys, values, color="#4C72B0")
    plt.ylabel("Recall")
    plt.title("Retrieval Recall@K")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
