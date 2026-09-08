import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BERTTextEncoder(nn.Module):
   

    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128, freeze_bert: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.hidden_size = self.bert.config.hidden_size

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def tokenize(self, texts, device=None):
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return enc

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H_text = out.last_hidden_state           
        cls_emb = H_text[:, 0, :]                 
        return cls_emb, H_text

    def encode_texts(self, texts, device):
        enc = self.tokenize(texts, device=device)
        return self.forward(enc["input_ids"], enc["attention_mask"])


class BERTTagClassifier(nn.Module):

    def __init__(self, num_tags: int, model_name: str = "distilbert-base-uncased", max_length: int = 128, freeze_bert: bool = False, dropout: float = 0.1):
        super().__init__()
        self.encoder = BERTTextEncoder(model_name, max_length, freeze_bert)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.hidden_size, num_tags)  # W in R^{K x d}, b in R^K

    def forward(self, input_ids, attention_mask):
        cls_emb, _ = self.encoder(input_ids, attention_mask)
        logits = self.head(self.dropout(cls_emb))   # pre-sigmoid logits, K per sample
        return logits

    def predict_texts(self, texts, device):
        self.eval()
        with torch.no_grad():
            enc = self.encoder.tokenize(texts, device=device)
            logits = self.forward(enc["input_ids"], enc["attention_mask"])
            probs = torch.sigmoid(logits)
        return probs


def bce_tag_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
