import torch
import torch.nn as nn
from components.layers import HierarchicalTransformerLayer


class HierarchicalBERTBase(nn.Module):
    """Base hierarchical BERT model without task-specific heads"""

    def __init__(self, num_layers=2, vocab_size=50368, max_length=512, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

        # Hierarchical transformer layers
        self.hierarchical_layers = nn.ModuleList(
            [HierarchicalTransformerLayer() for _ in range(num_layers)]
        )

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        # Create position ids [0, 1, 2, ..., seq_len-1]
        position_ids = (
            torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        )

        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        # Combine embeddings
        embeddings = token_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Pass through hierarchical layers
        hidden_states = embeddings
        for layer in self.hierarchical_layers:
            hidden_states = layer(hidden_states, attention_mask)

        return {
            "last_hidden_state": hidden_states,
            "pooler_output": hidden_states[:, 0],  # [CLS] token for classification tasks
        }