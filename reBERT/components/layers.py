from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer


def split_768_to_384(hidden_states):
    # Split 768D into two 384D branches
    mid_point = hidden_states.size(-1) // 2  # 384
    branch_A = hidden_states[:, :, :mid_point]
    branch_B = hidden_states[:, :, mid_point:]
    return branch_A, branch_B


def split_384_to_192(hidden_states_384):
    """
    Split 384D hidden states into two 192D branches

    Args:
        hidden_states_384: tensor of shape (batch_size, seq_len, 384)

    Returns:
        tuple: (sub_branch_1, sub_branch_2) each of shape (batch_size, seq_len, 192)
    """
    mid_point = hidden_states_384.size(-1) // 2  # 192
    sub_branch_1 = hidden_states_384[:, :, :mid_point]  # First 192 dimensions
    sub_branch_2 = hidden_states_384[:, :, mid_point:]  # Second 192 dimensions

    return sub_branch_1, sub_branch_2


def create_192d_transformer():
    """Create a BERT layer configured for 192D processing"""

    # Step 1: Create BertConfig for 192D
    config_192d = BertConfig(
        hidden_size=192,
        num_attention_heads=4,  # 192/4 = 48 per head
        intermediate_size=768,  # 4x hidden size
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    # Step 2: Create the layer
    layer_192d = BertLayer(config_192d)

    return layer_192d


def process_all_192d_branches(sub_A1, sub_A2, sub_B1, sub_B2, attention_mask=None):
    """
    Process all four 192D branches independently with BERT layers

    Args:
        sub_A1, sub_A2, sub_B1, sub_B2: tensors of shape (batch_size, seq_len, 192)
        attention_mask: optional attention mask

    Returns:
        tuple: (processed_A1, processed_A2, processed_B1, processed_B2)
    """
    # Create transformer layers for each branch
    transformer_A1 = create_192d_transformer()
    transformer_A2 = create_192d_transformer()
    transformer_B1 = create_192d_transformer()
    transformer_B2 = create_192d_transformer()

    # Process each branch independently
    processed_A1 = transformer_A1(sub_A1, attention_mask)[0]  # [0] gets hidden states
    processed_A2 = transformer_A2(sub_A2, attention_mask)[0]
    processed_B1 = transformer_B1(sub_B1, attention_mask)[0]
    processed_B2 = transformer_B2(sub_B2, attention_mask)[0]

    return processed_A1, processed_A2, processed_B1, processed_B2