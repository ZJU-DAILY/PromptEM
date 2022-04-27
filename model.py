import torch.nn as nn
from transformers import AutoModel


class LMNet(nn.Module):
    def __init__(self, lm='roberta-base'):
        super().__init__()
        self.lm = lm
        self.model = AutoModel.from_pretrained(lm)
        hidden_size = 768
        hidden_dropout_prob = 0.1
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, x):
        """Forward function of the models for classification."""
        output = self.model(x)
        cls = output[0][:, 0, :]
        cls = self.dropout(cls)
        logits = self.linear(cls)
        return logits
