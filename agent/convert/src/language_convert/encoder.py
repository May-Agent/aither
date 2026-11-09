"""The traced torch module: the language-detection BERT at one fixed length.

Attention is written out (matmul, additive mask, softmax, matmul) so the graph is
the plain ops the Neural Engine compiles. Position embeddings, the token-type-0
embedding and the padding bias are constants of the length. Output is the logits
of the classification head over the pooled first token.
"""

import math

import torch

from .spec import NEG


class Classifier(torch.nn.Module):
    def __init__(self, model, length):
        super().__init__()
        self.bert = model.bert
        self.classifier = model.classifier
        cfg = model.config
        self.length = length
        self.heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        emb = self.bert.embeddings
        self.register_buffer(
            "offset",
            (emb.position_embeddings.weight[:length] + emb.token_type_embeddings.weight[0]).unsqueeze(0),
        )

    def split(self, x):
        return x.view(1, self.length, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, input_ids, attention_mask):
        emb = self.bert.embeddings
        h = emb.LayerNorm(emb.word_embeddings(input_ids) + self.offset)
        bias = ((1.0 - attention_mask) * NEG).view(1, 1, 1, self.length)
        scale = 1.0 / math.sqrt(self.head_dim)
        for layer in self.bert.encoder.layer:
            a = layer.attention
            q, k, v = (self.split(p(h)) for p in (a.self.query, a.self.key, a.self.value))
            w = torch.softmax(torch.matmul(q, k.transpose(2, 3)) * scale + bias, dim=-1)
            ctx = torch.matmul(w, v).transpose(1, 2).reshape(1, self.length, -1)
            h = a.output.LayerNorm(a.output.dense(ctx) + h)
            inner = layer.intermediate.intermediate_act_fn(layer.intermediate.dense(h))
            h = layer.output.LayerNorm(layer.output.dense(inner) + h)
        pooled = self.bert.pooler.activation(self.bert.pooler.dense(h[:, 0]))
        return self.classifier(pooled)
