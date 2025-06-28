import torch
import torch.nn as nn
from components.layers import HierarchicalTransformerLayer


class MinimalHierarchicalBERT(nn.Module):
    def __init__(self, num_layers=2, vocab_size=50368, max_length=512, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

        self.hierarchical_layers = nn.ModuleList(
            [HierarchicalTransformerLayer() for _ in range(num_layers)]
        )

        # QA head for SQuAD (add at the end of __init__)
        self.qa_outputs = nn.Linear(hidden_size, 2)  # 2 outputs: start_logits, end_logits

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

        hidden_states = embeddings
        for layer in self.hierarchical_layers:
            hidden_states = layer(hidden_states, attention_mask)

        # QA head
        qa_logits = self.qa_outputs(hidden_states)  # [batch, seq_len, 2]
        start_logits = qa_logits[:, :, 0]  # [batch, seq_len]
        end_logits = qa_logits[:, :, 1]  # [batch, seq_len]

        return {
            "last_hidden_state": hidden_states,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }