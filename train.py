import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from dataset import (
    generate_synthetic_corpus, SAMPLE_TAGS, NODE_FEAT_DIM,
    export_sample_graphs, train_val_test_split, save_splits,
    MusicTagTextDataset, collate_tag_text,
    MusicGraphDataset, collate_graphs,
    MusicFusionDataset, collate_fusion,
    MusicCaptionPairDataset, collate_pairs,
)
from bert_encoder import BERTTagClassifier, bce_tag_loss
from gnn_model import GNNTagClassifier, CNNBaseline
from fusion_model import GNNBERTFusionModel, multi_task_loss
from contrastive import ContrastiveGNNBERT, info_nce_loss, retrieval_eval
from evaluate import (
    multilabel_classification_metrics, regression_metrics,
    majority_class_baseline, plot_f1_curve, plot_tsne, plot_retrieval_bar,
)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    want = cfg["train"].get("device", "cuda")
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dirs(cfg: dict):
    for key in ["results_dir", "checkpoints_dir"]:
        os.makedirs(cfg["paths"][key], exist_ok=True)
    os.makedirs(os.path.join(cfg["paths"]["results_dir"], "plots"), exist_ok=True)
    os.makedirs(os.path.join(cfg["paths"]["results_dir"], "retrieval_examples"), exist_ok=True)
    os.makedirs(cfg["paths"]["processed_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["splits_dir"], exist_ok=True)


def prepare_corpus(cfg: dict, loader: str, n_synthetic: int = 200):
    """Returns (train, val, test, tag_vocab, node_in_dim)."""
    if loader == "fma":
        from load_fma import load_fma_corpus
        corpus, tag_vocab = load_fma_corpus(cfg)
    else:
        corpus = generate_synthetic_corpus(n_tracks=n_synthetic, seed=cfg["train"]["seed"])
        tag_vocab = SAMPLE_TAGS

    node_in_dim = len(corpus[0]["segment_features"][0]) if loader == "fma" else NODE_FEAT_DIM

    train, val, test = train_val_test_split(corpus, seed=cfg["train"]["seed"])
    save_splits(train, val, test, cfg["paths"]["splits_dir"])

    sample_dir = os.path.join(cfg["paths"]["processed_dir"], "sample_graphs")
    export_sample_graphs(corpus, sample_dir, n_samples=20,
                          similarity_threshold=cfg["graph"]["similarity_threshold"])

    print(f"[data] loader={loader}  train={len(train)} val={len(val)} test={len(test)}  "
          f"tags={len(tag_vocab)}  node_in_dim={node_in_dim}")
    return train, val, test, tag_vocab, node_in_dim


def save_metrics(cfg: dict, run_name: str, metrics: dict):
    path = os.path.join(cfg["paths"]["results_dir"], "metrics.json")
    all_metrics = {}
    if os.path.exists(path):
        with open(path) as f:
            all_metrics = json.load(f)
    all_metrics[run_name] = metrics
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[metrics] wrote results/metrics.json['{run_name}']")


def y_to_numpy(*tensors):
    return [t.detach().cpu().numpy() for t in tensors]


# ---------------------------------------------------------------------------
# Task 1: BERT tag classifier
# ---------------------------------------------------------------------------
def run_task1(cfg, args, train, val, test, tag_vocab, device):
    model = BERTTagClassifier(
        num_tags=len(tag_vocab), model_name=cfg["text"]["bert_model_name"],
        max_length=cfg["text"]["max_length"], freeze_bert=cfg["text"]["freeze_bert"],
    ).to(device)

    collate = partial(collate_tag_text, tokenizer=model.encoder.tokenizer,
                       max_length=cfg["text"]["max_length"])
    train_loader = DataLoader(MusicTagTextDataset(train), batch_size=cfg["train"]["batch_size"],
                               shuffle=True, collate_fn=collate)
    val_loader = DataLoader(MusicTagTextDataset(val), batch_size=cfg["train"]["batch_size"],
                             shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"],
                                   weight_decay=cfg["train"]["weight_decay"])

    history = {"epoch": [], "macro_f1": [], "micro_f1": []}
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for input_ids, attn_mask, tags in train_loader:
            input_ids, attn_mask, tags = input_ids.to(device), attn_mask.to(device), tags.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attn_mask)
            loss = bce_tag_loss(logits, tags)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * tags.size(0)

        model.eval()
        all_probs, all_tags = [], []
        with torch.no_grad():
            for input_ids, attn_mask, tags in val_loader:
                input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
                probs = torch.sigmoid(model(input_ids, attn_mask))
                all_probs.append(probs.cpu())
                all_tags.append(tags)
        y_prob = torch.cat(all_probs).numpy()
        y_true = torch.cat(all_tags).numpy()
        m = multilabel_classification_metrics(y_true, y_prob)
        history["epoch"].append(epoch)
        history["macro_f1"].append(m["macro_f1"])
        history["micro_f1"].append(m["micro_f1"])
        print(f"[task1][epoch {epoch}] loss={total_loss/len(train):.4f} "
              f"macro_f1={m['macro_f1']:.4f} micro_f1={m['micro_f1']:.4f} auc_pr={m['mean_auc_pr']:.4f}")

    ckpt = os.path.join(cfg["paths"]["checkpoints_dir"], "task1_bert.pt")
    torch.save(model.state_dict(), ckpt)
    plot_f1_curve(history, os.path.join(cfg["paths"]["results_dir"], "plots", "task1_f1_curve.png"))

    y_train = np.stack([t["tags"] for t in train])
    y_test_true = np.stack([t["tags"] for t in test])
    b1_probs = majority_class_baseline(y_train, len(test))
    b1_metrics = multilabel_classification_metrics(y_test_true, b1_probs)

    final_test_metrics = _eval_task1_on(model, test, model.encoder.tokenizer, cfg, device)
    save_metrics(cfg, "task1_bert", {
        "history": history, "test": final_test_metrics,
        "baseline_majority_class": {"macro_f1": b1_metrics["macro_f1"], "micro_f1": b1_metrics["micro_f1"]},
        "checkpoint": ckpt,
    })


def _eval_task1_on(model, split, tokenizer, cfg, device):
    collate = partial(collate_tag_text, tokenizer=tokenizer, max_length=cfg["text"]["max_length"])
    loader = DataLoader(MusicTagTextDataset(split), batch_size=cfg["train"]["batch_size"],
                         shuffle=False, collate_fn=collate)
    model.eval()
    all_probs, all_tags = [], []
    with torch.no_grad():
        for input_ids, attn_mask, tags in loader:
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            probs = torch.sigmoid(model(input_ids, attn_mask))
            all_probs.append(probs.cpu())
            all_tags.append(tags)
    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_tags).numpy()
    return multilabel_classification_metrics(y_true, y_prob)


# ---------------------------------------------------------------------------
# Task 2: GNN on structure graphs
# ---------------------------------------------------------------------------
def run_task2(cfg, args, train, val, test, tag_vocab, node_in_dim, device):
    model = GNNTagClassifier(
        in_dim=node_in_dim, num_tags=len(tag_vocab), gnn_type=cfg["model"]["gnn_type"],
        hidden_dim=cfg["model"]["gnn_hidden_dim"], out_dim=cfg["model"]["gnn_out_dim"],
        num_layers=cfg["model"]["gnn_num_layers"], dropout=cfg["model"]["gnn_dropout"],
    ).to(device)

    sim_thresh = cfg["graph"]["similarity_threshold"]
    train_loader = DataLoader(MusicGraphDataset(train, sim_thresh), batch_size=cfg["train"]["batch_size"],
                               shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(MusicGraphDataset(val, sim_thresh), batch_size=cfg["train"]["batch_size"],
                             shuffle=False, collate_fn=collate_graphs)
    test_loader = DataLoader(MusicGraphDataset(test, sim_thresh), batch_size=cfg["train"]["batch_size"],
                              shuffle=False, collate_fn=collate_graphs)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["gnn_learning_rate"],
                                  weight_decay=cfg["train"]["weight_decay"])

    history = {"epoch": [], "macro_f1": [], "micro_f1": []}
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for graph_batch, targets in train_loader:
            graph_batch, targets = graph_batch.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)

        m = _eval_task2_loader(model, val_loader, device)
        history["epoch"].append(epoch)
        history["macro_f1"].append(m["macro_f1"])
        history["micro_f1"].append(m["micro_f1"])
        print(f"[task2][epoch {epoch}] loss={total_loss/len(train):.4f} "
              f"macro_f1={m['macro_f1']:.4f} micro_f1={m['micro_f1']:.4f}")

    ckpt = os.path.join(cfg["paths"]["checkpoints_dir"], f"task2_gnn_{cfg['model']['gnn_type']}.pt")
    torch.save(model.state_dict(), ckpt)
    plot_f1_curve(history, os.path.join(cfg["paths"]["results_dir"], "plots", "task2_f1_curve.png"))

    y_train = np.stack([t["tags"] for t in train])
    y_test_true = np.stack([t["tags"] for t in test])
    b1_probs = majority_class_baseline(y_train, len(test))
    b1_metrics = multilabel_classification_metrics(y_test_true, b1_probs)

    test_metrics = _eval_task2_loader(model, test_loader, device)
    cnn_baseline_results = run_cnn_baseline(cfg, args, train, val, test, tag_vocab, device)
    save_metrics(cfg, f"task2_gnn_{cfg['model']['gnn_type']}", {
        "history": history, "test": test_metrics,
        "baseline_majority_class": {"macro_f1": b1_metrics["macro_f1"], "micro_f1": b1_metrics["micro_f1"]},
        "baseline_cnn_mel": cnn_baseline_results,
        "checkpoint": ckpt,
    })


# ---------------------------------------------------------------------------
# B2 baseline (Section 8): CNN on log-mel spectrogram, no graph, no text
# ---------------------------------------------------------------------------
def _build_mel_dataset(items, mel_features):
    X, y = [], []
    for item in items:
        mel = mel_features.get(item["track_id"])
        if mel is None:
            continue
        X.append(mel)
        y.append(item["tags"])
    if len(X) == 0:
        return None
    X = torch.tensor(np.stack(X), dtype=torch.float).unsqueeze(1)  # (N, 1, n_mels, T)
    y = torch.tensor(np.stack(y), dtype=torch.float)
    return TensorDataset(X, y)


def _eval_cnn_loader(model, loader, device):
    model.eval()
    all_probs, all_tags = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            probs = torch.sigmoid(model(X))
            all_probs.append(probs.cpu())
            all_tags.append(y)
    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_tags).numpy()
    return multilabel_classification_metrics(y_true, y_prob)


def run_cnn_baseline(cfg, args, train, val, test, tag_vocab, device):
    if args.loader == "fma":
        from load_fma import load_fma_mel_for_corpus
        mel_features = load_fma_mel_for_corpus(cfg, train + val + test)
    else:
        from dataset import generate_synthetic_mel_for_corpus
        mel_features = generate_synthetic_mel_for_corpus(
            train + val + test, n_mels=cfg["audio"]["n_mels"],
            max_frames=cfg["audio"].get("cnn_max_frames", 130), seed=cfg["train"]["seed"],
        )

    train_ds = _build_mel_dataset(train, mel_features)
    val_ds = _build_mel_dataset(val, mel_features)
    test_ds = _build_mel_dataset(test, mel_features)
    if train_ds is None or test_ds is None or len(test_ds) == 0:
        print("[task2:cnn_baseline] no mel-spectrograms available for this corpus, skipping B2 baseline.")
        return None

    model = CNNBaseline(num_tags=len(tag_vocab), n_mels=cfg["audio"]["n_mels"]).to(device)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False) if val_ds and len(val_ds) > 0 else None
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["gnn_learning_rate"],
                                  weight_decay=cfg["train"]["weight_decay"])

    history = {"epoch": [], "macro_f1": [], "micro_f1": []}
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss, n = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            n += X.size(0)

        if val_loader is not None:
            m = _eval_cnn_loader(model, val_loader, device)
            history["epoch"].append(epoch)
            history["macro_f1"].append(m["macro_f1"])
            history["micro_f1"].append(m["micro_f1"])
            print(f"[task2:cnn_baseline][epoch {epoch}] loss={total_loss/max(n,1):.4f} "
                  f"macro_f1={m['macro_f1']:.4f} micro_f1={m['micro_f1']:.4f}")
        else:
            print(f"[task2:cnn_baseline][epoch {epoch}] loss={total_loss/max(n,1):.4f}")

    ckpt = os.path.join(cfg["paths"]["checkpoints_dir"], "task2_cnn_baseline.pt")
    torch.save(model.state_dict(), ckpt)
    if history["epoch"]:
        plot_f1_curve(history, os.path.join(cfg["paths"]["results_dir"], "plots", "task2_cnn_baseline_f1_curve.png"))

    test_metrics = _eval_cnn_loader(model, test_loader, device)
    print(f"[task2:cnn_baseline][test] macro_f1={test_metrics['macro_f1']:.4f} micro_f1={test_metrics['micro_f1']:.4f}")

    return {
        "history": history, "test": test_metrics, "checkpoint": ckpt,
        "n_tracks_with_mel": len(train_ds) + (len(val_ds) if val_ds else 0) + len(test_ds),
        "source": "real FMA log-mel spectrograms" if args.loader == "fma"
                  else "synthetic placeholder log-mel spectrograms (use --loader fma for a real B2 result)",
    }


def _eval_task2_loader(model, loader, device):
    model.eval()
    all_probs, all_tags = [], []
    with torch.no_grad():
        for graph_batch, targets in loader:
            graph_batch = graph_batch.to(device)
            logits, _ = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            all_probs.append(torch.sigmoid(logits).cpu())
            all_tags.append(targets)
    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_tags).numpy()
    return multilabel_classification_metrics(y_true, y_prob)


# ---------------------------------------------------------------------------
# Task 3: GNN-BERT fusion
# ---------------------------------------------------------------------------
def run_task3(cfg, args, train, val, test, tag_vocab, node_in_dim, device):
    fusion_mode = args.fusion_mode
    model = GNNBERTFusionModel(
        node_in_dim=node_in_dim, num_tags=len(tag_vocab),
        bert_model_name=cfg["text"]["bert_model_name"], gnn_type=cfg["model"]["gnn_type"],
        gnn_hidden_dim=cfg["model"]["gnn_hidden_dim"], gnn_out_dim=cfg["model"]["gnn_out_dim"],
        gnn_num_layers=cfg["model"]["gnn_num_layers"], gnn_dropout=cfg["model"]["gnn_dropout"],
        fusion_hidden_dim=cfg["model"]["fusion_hidden_dim"], max_length=cfg["text"]["max_length"],
        freeze_bert=cfg["text"]["freeze_bert"], fusion_mode=fusion_mode,
    ).to(device)

    sim_thresh = cfg["graph"]["similarity_threshold"]
    tokenizer = model.text_encoder.tokenizer
    collate = partial(collate_fusion, tokenizer=tokenizer, max_length=cfg["text"]["max_length"])
    train_loader = DataLoader(MusicFusionDataset(train, sim_thresh), batch_size=cfg["train"]["batch_size"],
                               shuffle=True, collate_fn=collate)
    val_loader = DataLoader(MusicFusionDataset(val, sim_thresh), batch_size=cfg["train"]["batch_size"],
                             shuffle=False, collate_fn=collate)
    test_loader = DataLoader(MusicFusionDataset(test, sim_thresh), batch_size=cfg["train"]["batch_size"],
                              shuffle=False, collate_fn=collate)

    bert_params = list(model.text_encoder.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("text_encoder")]
    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": cfg["train"]["learning_rate"]},
        {"params": other_params, "lr": cfg["train"]["gnn_learning_rate"]},
    ], weight_decay=cfg["train"]["weight_decay"])

    alpha, beta = cfg["train"]["alpha_valence"], cfg["train"]["beta_arousal"]
    history = {"epoch": [], "macro_f1": [], "micro_f1": []}
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            graph_batch = batch["graph_batch"].to(device)
            input_ids, attn_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            tags, valence, arousal = batch["tags"].to(device), batch["valence"].to(device), batch["arousal"].to(device)

            optimizer.zero_grad()
            out = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, input_ids, attn_mask)
            loss, _ = multi_task_loss(out, tags, valence, arousal, alpha=alpha, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * tags.size(0)

        m, _, _, _ = _eval_task3_loader(model, val_loader, device)
        history["epoch"].append(epoch)
        history["macro_f1"].append(m["macro_f1"])
        history["micro_f1"].append(m["micro_f1"])
        print(f"[task3:{fusion_mode}][epoch {epoch}] loss={total_loss/len(train):.4f} "
              f"macro_f1={m['macro_f1']:.4f} micro_f1={m['micro_f1']:.4f}")

    ckpt = os.path.join(cfg["paths"]["checkpoints_dir"], f"task3_fusion_{fusion_mode}.pt")
    torch.save(model.state_dict(), ckpt)
    plot_f1_curve(history, os.path.join(cfg["paths"]["results_dir"], "plots", f"task3_{fusion_mode}_f1_curve.png"))

    test_metrics, z_all, tags_all, track_ids = _eval_task3_loader(model, test_loader, device)
    v_true, v_pred, a_true, a_pred = _collect_emotion(model, test_loader, device)
    reg_v = regression_metrics(v_true, v_pred)
    reg_a = regression_metrics(a_true, a_pred)

    
    dominant = [tag_vocab[int(np.argmax(t))] if t.sum() > 0 else "none" for t in tags_all]
    plot_tsne(z_all, dominant, os.path.join(cfg["paths"]["results_dir"], "plots", f"task3_{fusion_mode}_tsne.png"))

    
    case_studies = _build_case_studies(model, test[:3], tokenizer, cfg, device, tag_vocab)
    with open(os.path.join(cfg["paths"]["results_dir"], "retrieval_examples", f"task3_{fusion_mode}_case_studies.json"), "w") as f:
        json.dump(case_studies, f, indent=2)

    save_metrics(cfg, f"task3_fusion_{fusion_mode}", {
        "history": history, "test_tags": test_metrics,
        "test_valence": reg_v, "test_arousal": reg_a,
        "checkpoint": ckpt,
    })


def _eval_task3_loader(model, loader, device):
    model.eval()
    all_probs, all_tags, all_z, all_ids = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            graph_batch = batch["graph_batch"].to(device)
            input_ids, attn_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            out = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, input_ids, attn_mask)
            all_probs.append(torch.sigmoid(out["tag_logits"]).cpu())
            all_tags.append(batch["tags"])
            all_z.append(out["z"].cpu())
    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_tags).numpy()
    z_all = torch.cat(all_z).numpy()
    m = multilabel_classification_metrics(y_true, y_prob)
    return m, z_all, y_true, all_ids


def _collect_emotion(model, loader, device):
    model.eval()
    v_true, v_pred, a_true, a_pred = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            graph_batch = batch["graph_batch"].to(device)
            input_ids, attn_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            out = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, input_ids, attn_mask)
            v_pred.append(out["valence_hat"].cpu().numpy())
            a_pred.append(out["arousal_hat"].cpu().numpy())
            v_true.append(batch["valence"].numpy())
            a_true.append(batch["arousal"].numpy())
    return (np.concatenate(v_true), np.concatenate(v_pred),
            np.concatenate(a_true), np.concatenate(a_pred))


def _build_case_studies(model, items, tokenizer, cfg, device, tag_vocab):
    from dataset import MusicFusionDataset, collate_fusion
    ds = MusicFusionDataset(items, cfg["graph"]["similarity_threshold"])
    collate = partial(collate_fusion, tokenizer=tokenizer, max_length=cfg["text"]["max_length"])
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
    model.eval()
    cases = []
    with torch.no_grad():
        for item, batch in zip(items, loader):
            graph_batch = batch["graph_batch"].to(device)
            input_ids, attn_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            out = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, input_ids, attn_mask)
            probs = torch.sigmoid(out["tag_logits"])[0].cpu().numpy()
            predicted_tags = [tag_vocab[i] for i, p in enumerate(probs) if p > 0.5]
            true_tags = [tag_vocab[i] for i, v in enumerate(item["tags"]) if v == 1.0]
            cases.append({
                "track_id": item["track_id"],
                "caption": item["caption"],
                "num_graph_nodes": len(item["segment_features"]),
                "true_tags": true_tags,
                "predicted_tags": predicted_tags,
                "predicted_valence": float(out["valence_hat"][0].cpu()),
                "predicted_arousal": float(out["arousal_hat"][0].cpu()),
                "true_valence": item["valence"],
                "true_arousal": item["arousal"],
            })
    return cases


# ---------------------------------------------------------------------------
# Task 4: Contrastive dual-encoder (InfoNCE)
# ---------------------------------------------------------------------------
def run_task4(cfg, args, train, val, test, tag_vocab, node_in_dim, device):
    model = ContrastiveGNNBERT(
        node_in_dim=node_in_dim, proj_dim=cfg["model"]["gnn_out_dim"],
        bert_model_name=cfg["text"]["bert_model_name"], gnn_type=cfg["model"]["gnn_type"],
        gnn_hidden_dim=cfg["model"]["gnn_hidden_dim"], gnn_out_dim=cfg["model"]["gnn_out_dim"],
        gnn_num_layers=cfg["model"]["gnn_num_layers"], gnn_dropout=cfg["model"]["gnn_dropout"],
        max_length=cfg["text"]["max_length"], freeze_bert=cfg["text"]["freeze_bert"],
    ).to(device)

    sim_thresh = cfg["graph"]["similarity_threshold"]
    tokenizer = model.text_encoder.tokenizer
    collate = partial(collate_pairs, tokenizer=tokenizer, max_length=cfg["text"]["max_length"])
    train_loader = DataLoader(MusicCaptionPairDataset(train, sim_thresh), batch_size=cfg["train"]["batch_size"],
                               shuffle=True, collate_fn=collate, drop_last=True)
    test_loader = DataLoader(MusicCaptionPairDataset(test, sim_thresh), batch_size=len(test),
                              shuffle=False, collate_fn=collate)

    bert_params = list(model.text_encoder.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("text_encoder")]
    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": cfg["train"]["learning_rate"]},
        {"params": other_params, "lr": cfg["train"]["gnn_learning_rate"]},
    ], weight_decay=cfg["train"]["weight_decay"])

    temperature = cfg["model"]["contrastive_temperature"]
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for graph_batch, input_ids, attn_mask, track_ids, captions in train_loader:
            graph_batch = graph_batch.to(device)
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            optimizer.zero_grad()
            g, t = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, input_ids, attn_mask)
            loss = info_nce_loss(g, t, temperature=temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"[task4][epoch {epoch}] infonce_loss={total_loss/max(n_batches,1):.4f}")

    ckpt = os.path.join(cfg["paths"]["checkpoints_dir"], "task4_contrastive.pt")
    torch.save(model.state_dict(), ckpt)

    
    model.eval()
    with torch.no_grad():
        graph_batch, input_ids, attn_mask, track_ids, captions = next(iter(test_loader))
        graph_batch = graph_batch.to(device)
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
        g, t = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, input_ids, attn_mask)

    retrieval = retrieval_eval(g, t, k_values=tuple(cfg["eval"]["k_values"]))
    plot_retrieval_bar(retrieval, os.path.join(cfg["paths"]["results_dir"], "plots", "task4_retrieval.png"))

    
    sim = (g @ t.t()).cpu().numpy()  
    examples = []
    n_show = min(10, len(track_ids))
    for i in range(n_show):
        top3_idx = np.argsort(-sim[:, i])[:3]  
        examples.append({
            "query_caption": captions[i],
            "top3_matched_track_ids": [track_ids[j] for j in top3_idx],
            "top3_scores": [float(sim[j, i]) for j in top3_idx],
        })
    with open(os.path.join(cfg["paths"]["results_dir"], "retrieval_examples", "task4_examples.json"), "w") as f:
        json.dump(examples, f, indent=2)

    save_metrics(cfg, "task4_contrastive", {"retrieval": retrieval, "checkpoint": ckpt})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--task", type=int, default=None, choices=[1, 2, 3, 4])
    parser.add_argument("--loader", type=str, default="synthetic", choices=["synthetic", "fma"])
    parser.add_argument("--fusion_mode", type=str, default="cross_attention",
                         choices=["cross_attention", "concat", "bert_only", "gnn_only"])
    parser.add_argument("--n_synthetic", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.task is not None:
        cfg["task"] = args.task
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    set_seed(cfg["train"]["seed"])
    ensure_dirs(cfg)
    device = get_device(cfg)
    print(f"[setup] task={cfg['task']} loader={args.loader} device={device}")

    train, val, test, tag_vocab, node_in_dim = prepare_corpus(cfg, args.loader, args.n_synthetic)

    task = cfg["task"]
    if task == 1:
        run_task1(cfg, args, train, val, test, tag_vocab, device)
    elif task == 2:
        run_task2(cfg, args, train, val, test, tag_vocab, node_in_dim, device)
    elif task == 3:
        run_task3(cfg, args, train, val, test, tag_vocab, node_in_dim, device)
    elif task == 4:
        run_task4(cfg, args, train, val, test, tag_vocab, node_in_dim, device)
    else:
        raise ValueError(f"Unknown task {task}")

    print("[done]")


if __name__ == "__main__":
    main()
