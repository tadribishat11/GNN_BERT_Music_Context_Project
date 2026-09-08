import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn_model import build_gnn_encoder, readout_mean_pool
from bert_encoder import BERTTextEncoder


class CrossAttentionFusion(nn.Module):

    def __init__(self, graph_dim: int, text_dim: int, attn_dim: int = 128):
        super().__init__()
        self.W_Q = nn.Linear(graph_dim, attn_dim, bias=False)
        self.W_K = nn.Linear(text_dim, attn_dim, bias=False)
        self.W_V = nn.Linear(text_dim, attn_dim, bias=False)
        self.scale = attn_dim ** 0.5

    def forward(self, g: torch.Tensor, H_text: torch.Tensor, attention_mask: torch.Tensor = None):
        Q = self.W_Q(g).unsqueeze(1)              
        K = self.W_K(H_text)                       
        V = self.W_V(H_text)                       

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale   
        if attention_mask is not None:
            mask = (1.0 - attention_mask.unsqueeze(1).float()) * -1e9
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)            
        context = torch.bmm(attn, V).squeeze(1)      
        return context, attn.squeeze(1)


class GNNBERTFusionModel(nn.Module):
    def __init__(self, node_in_dim: int, num_tags: int,
                 bert_model_name: str = "distilbert-base-uncased",
                 gnn_type: str = "sage", gnn_hidden_dim: int = 128, gnn_out_dim: int = 128,
                 gnn_num_layers: int = 2, gnn_dropout: float = 0.3,
                 fusion_hidden_dim: int = 256, max_length: int = 128,
                 freeze_bert: bool = False, fusion_mode: str = "cross_attention"):
        super().__init__()
        assert fusion_mode in ("cross_attention", "concat", "bert_only", "gnn_only")
        self.fusion_mode = fusion_mode

        self.gnn_encoder = build_gnn_encoder(gnn_type, node_in_dim, gnn_hidden_dim,
                                              gnn_out_dim, gnn_num_layers, gnn_dropout)
        self.text_encoder = BERTTextEncoder(bert_model_name, max_length, freeze_bert)

        text_dim = self.text_encoder.hidden_size
        self.cross_attn = CrossAttentionFusion(gnn_out_dim, text_dim, attn_dim=fusion_hidden_dim)

        if fusion_mode == "cross_attention":
            z_dim = gnn_out_dim + fusion_hidden_dim
        elif fusion_mode == "concat":
            z_dim = gnn_out_dim + text_dim
        elif fusion_mode == "bert_only":
            z_dim = text_dim
        else:  
            z_dim = gnn_out_dim

        self.tag_head = nn.Linear(z_dim, num_tags)
        self.valence_head = nn.Linear(z_dim, 1)
        self.arousal_head = nn.Linear(z_dim, 1)

    def forward(self, node_x, edge_index, graph_batch, input_ids, attention_mask):
        h = self.gnn_encoder(node_x, edge_index)
        g = readout_mean_pool(h, graph_batch)                     
        t, H_text = self.text_encoder(input_ids, attention_mask)   

        if self.fusion_mode == "cross_attention":
            context, attn_weights = self.cross_attn(g, H_text, attention_mask)
            z = torch.cat([g, context], dim=-1)
        elif self.fusion_mode == "concat":
            z = torch.cat([g, t], dim=-1)
            attn_weights = None
        elif self.fusion_mode == "bert_only":
            z = t
            attn_weights = None
        else:  
            z = g
            attn_weights = None

        tag_logits = self.tag_head(z)
        valence_hat = self.valence_head(z).squeeze(-1)
        arousal_hat = self.arousal_head(z).squeeze(-1)
        return {
            "tag_logits": tag_logits,
            "valence_hat": valence_hat,
            "arousal_hat": arousal_hat,
            "z": z,
            "attn_weights": attn_weights,
        }


def multi_task_loss(outputs: dict, tag_targets: torch.Tensor,
                     valence_targets: torch.Tensor = None, arousal_targets: torch.Tensor = None,
                     alpha: float = 0.5, beta: float = 0.5):
    """L = L_tags + alpha * ||v - v_hat||^2 + beta * ||a - a_hat||^2"""
    l_tags = F.binary_cross_entropy_with_logits(outputs["tag_logits"], tag_targets)
    loss = l_tags
    aux = {}
    if valence_targets is not None:
        l_v = F.mse_loss(outputs["valence_hat"], valence_targets)
        loss = loss + alpha * l_v
        aux["valence_loss"] = l_v.item()
    if arousal_targets is not None:
        l_a = F.mse_loss(outputs["arousal_hat"], arousal_targets)
        loss = loss + beta * l_a
        aux["arousal_loss"] = l_a.item()
    aux["tag_loss"] = l_tags.item()
    return loss, aux
