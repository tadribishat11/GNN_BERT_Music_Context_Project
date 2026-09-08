import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn_model import build_gnn_encoder, readout_mean_pool
from bert_encoder import BERTTextEncoder


class ContrastiveGNNBERT(nn.Module):

    def __init__(self, node_in_dim: int, proj_dim: int = 128,
                 bert_model_name: str = "distilbert-base-uncased",
                 gnn_type: str = "sage", gnn_hidden_dim: int = 128, gnn_out_dim: int = 128,
                 gnn_num_layers: int = 2, gnn_dropout: float = 0.3,
                 max_length: int = 128, freeze_bert: bool = False):
        super().__init__()
        self.gnn_encoder = build_gnn_encoder(gnn_type, node_in_dim, gnn_hidden_dim,
                                              gnn_out_dim, gnn_num_layers, gnn_dropout)
        self.text_encoder = BERTTextEncoder(bert_model_name, max_length, freeze_bert)

        self.graph_proj = nn.Linear(gnn_out_dim, proj_dim)
        self.text_proj = nn.Linear(self.text_encoder.hidden_size, proj_dim)

    def encode_graph(self, node_x, edge_index, graph_batch):
        h = self.gnn_encoder(node_x, edge_index)
        g = readout_mean_pool(h, graph_batch)
        g = self.graph_proj(g)
        return F.normalize(g, dim=-1)   

    def encode_text(self, input_ids, attention_mask):
        t, _ = self.text_encoder(input_ids, attention_mask)
        t = self.text_proj(t)
        return F.normalize(t, dim=-1)   

    def forward(self, node_x, edge_index, graph_batch, input_ids, attention_mask):
        g = self.encode_graph(node_x, edge_index, graph_batch)
        t = self.encode_text(input_ids, attention_mask)
        return g, t


def info_nce_loss(g: torch.Tensor, t: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = g @ t.t() / temperature                     
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_g2t + loss_t2g)


@torch.no_grad()
def retrieval_eval(g: torch.Tensor, t: torch.Tensor, k_values=(1, 5, 10)):
    sim = g @ t.t()          
    n = sim.size(0)
    targets = torch.arange(n, device=sim.device)

    results = {}
    for direction, matrix in [("audio_to_caption", sim), ("caption_to_audio", sim.t())]:
        ranks = matrix.argsort(dim=-1, descending=True)   # (N, N) indices sorted by similarity
        correct_rank = (ranks == targets.unsqueeze(1)).float().argmax(dim=1)  # position of true match
        for k in k_values:
            recall_k = (correct_rank < k).float().mean().item()
            results[f"{direction}_R@{k}"] = recall_k
    return results
