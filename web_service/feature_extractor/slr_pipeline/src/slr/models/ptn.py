"""Pose Transformer Network, implemented with PyTorch classes."""

import torch
from torch import nn

from .common import KeypointEmbeddingDense, PositionalEncoding, KeypointEmbeddingAttention


class PTN(nn.Module):
    def __init__(self, num_attention_layers: int, num_attention_heads: int, d_hidden: int, num_classes: int, **kwargs):
        super(PTN, self).__init__()

        # Hyperparameters / arguments.
        self.in_features = kwargs['d_pose']
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.d_hidden = d_hidden
        self.num_classes = num_classes
        self.residual_pose_embedding = kwargs['residual_pose_embedding']
        self.variable_length_sequences = kwargs['variable_length_sequences']
        self.use_pose_embedding = not kwargs['no_pose_embedding']
        self.pose_embedding_kind = kwargs['pose_embedding_kind']

        # Model architecture.
        if self.use_pose_embedding:
            if self.pose_embedding_kind == 'dense':
                self.pose_embedding = KeypointEmbeddingDense(self.in_features, self.d_hidden, self.residual_pose_embedding)
            else:
                self.pose_embedding = KeypointEmbeddingAttention()
        else:
            self.pose_embedding = nn.Linear(self.in_features, self.d_hidden, bias=False)
        self.pos_enc = PositionalEncoding(self.d_hidden)
        encoder_layer = nn.TransformerEncoderLayer(self.d_hidden, self.num_attention_heads, 2 * self.d_hidden,
                                                   dropout=0.2)
        self.self_attention = nn.TransformerEncoder(encoder_layer, self.num_attention_layers,
                                                    nn.LayerNorm(self.d_hidden))
        self.classifier = nn.Linear(self.d_hidden, self.num_classes)

        self.cls_emb = nn.Embedding(1, self.d_hidden)

        # For inference.
        self.hooks = dict()

    def forward(self, batch) -> torch.Tensor:
        """Perform a forward pass.

        :param pose_clip: Tensor of shape (Sequence Length, Batch Size, Features).
        :return: Logits."""
        pose_clip = batch.inputs
        pose_embedded = self.pose_embedding(pose_clip)

        pose_embedded = self.pos_enc(pose_embedded)

        # Add CLS token.
        cls_inputs = torch.zeros((batch.inputs.size(1), 1), dtype=torch.long).to(pose_embedded.device)
        cls_embedded = self.cls_emb(cls_inputs).permute(1, 0, 2)
        self_attention_inputs = torch.cat((cls_embedded, pose_embedded), dim=0)

        self_attention_outputs = self.self_attention(self_attention_inputs,
                                                     src_key_padding_mask=self.get_src_key_padding_mask(batch))

        classifier_inputs = self_attention_outputs[0]  # CLS token output.
        logits = self.classifier(classifier_inputs)

        return logits

    def get_src_key_padding_mask(self, batch):
        if not self.variable_length_sequences:
            return None

        # [N, S] padding mask: TRUE = masked; FALSE = not masked.
        # The + 1 for the second axis is because we add the CLS token.
        padding_mask = torch.zeros(batch.inputs.size(1), batch.inputs.size(0) + 1, dtype=torch.bool)
        for i, length in enumerate(batch.lengths):
            padding_mask[i, length + 1:] = 1  # Again: + 1 is for the CLS token.
        return padding_mask.to(batch.inputs.device)

    def init_output_bias(self, class_weights):
        self.classifier.bias.data = torch.log(class_weights)

    def setup_inference_hook(self, embedding_kind: str, hook):
        if embedding_kind == 'spatial':
            self.hooks[embedding_kind] = self.pose_embedding.register_forward_hook(hook)
        elif embedding_kind == 'temporal' or embedding_kind == 'CLS':
            self.hooks[embedding_kind] = self.self_attention.register_forward_hook(hook)
        else:
            raise ValueError(f'Unsupported embedding kind {embedding_kind} for inference hook.')

    def reset_inference_hook(self, embedding_kind: str):
        if embedding_kind in self.hooks:
            self.hooks[embedding_kind].remove()
        else:
            raise ValueError(f'Unsupported embedding kind {embedding_kind} for inference hook.')